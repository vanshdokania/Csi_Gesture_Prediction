#!/usr/bin/env python3
"""
pose_snn.py - stage 2 of the cross-person experiment: take the pose that SiWiS predicted from the Wi-Fi
signal and let a spiking neural network decide which gesture it shows.

    CSI --(SiWiS, trained once with video labels)--> predicted body mask per frame --> SNN --> bhu / lhu / rhu

The video is not used at this stage. The input is only what SiWiS predicted from the radio signal. The
predictions come from the leave-one-person-out run of the pose model (train_percom.py or
train_percom_v2.py): every person's frames were predicted by a SiWiS model that never saw that person,
so nothing leaks from the test person into the input.

Each window of T consecutive frames (default 10 frames = 5 s at 2 fps) is a small "pose spectrogram":
time x features. The features are the body outline pooled to 12 x 16 (192 values), the arm joints
(36 values), or both (228 values). --velocity adds the frame-to-frame change, i.e. how fast the pose
moves, and doubles the number of values.

    python pose_snn.py --results ~/percom/siwis_results --data ~/percom/siwis_data         # outline only, no speed
    python pose_snn.py ... --feat kp --velocity                                            # arm joints + speed
    python pose_snn.py ... --feat both --velocity                                          # outline + joints + speed
    python pose_snn.py ... --source gt                                                     # control: poses taken from the video

--source gt feeds the poses extracted from the video instead of the predicted ones. It is not a real result,
because it uses the video. It is a control experiment: it tells us whether a weak accuracy comes from the
SNN or from the quality of the predicted poses. In our case it gave 67-80 % (window accuracy 67 % without
speed, 75 % with speed), so the SNN is fine and the pose model is the limit.

SNN settings (the same as in the report): 256 -> 128 -> 3, decay 0.9, threshold 1.0, soft reset,
surrogate gradient slope 10, Adam with lr 0.002, batch size 64, 40 epochs, class-weighted
cross-entropy, no dropout, no augmentation, no validation set. Four runs, each person is the test person once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ACTS = ["bhu", "lhu", "rhu"]


# ----------------------------------------------------------------------------- data
JOINTS = [5, 6, 7, 8, 9, 10]          # L/R shoulder, L/R elbow, L/R wrist (COCO order)


def kp_features(heat, chunk=256):
    """Turn the predicted keypoint heatmaps (n, 17, H, W) into 36 numbers per frame.

    Only the shoulders, elbows and wrists are used. For these 6 joints we take the row and column of the
    heatmap peak, a soft (weighted) row and column around the peak, and the peak height. On top of that
    come the wrist-minus-elbow offsets (row and column) and the wrist-minus-shoulder height of both arms.
    """
    out = []
    for s in range(0, len(heat), chunk):
        h = heat[s:s + chunk][:, JOINTS].astype(np.float32)
        n, J, H, W = h.shape
        mx = h.max((2, 3))
        am = h.reshape(n, J, -1).argmax(2)
        r, c = am // W / H, am % W / W
        w = h * (h >= 0.5 * mx[..., None, None])
        sw = w.sum((2, 3)) + 1e-9
        sr = (w.sum(3) * (np.arange(H) / H)).sum(2) / sw
        sc = (w.sum(2) * (np.arange(W) / W)).sum(2) / sw
        rel = [sr[:, 4] - sr[:, 2], sc[:, 4] - sc[:, 2], sr[:, 5] - sr[:, 3], sc[:, 5] - sc[:, 3],
               sr[:, 4] - sr[:, 0], sr[:, 5] - sr[:, 1]]
        out.append(np.concatenate([r, c, sr, sc, mx, np.stack(rel, 1)], 1))
    return np.concatenate(out).astype(np.float32)


def load_frames(results, data, source, persons_exclude, feat="mask"):
    """Returns one dict per recording: person, act, rec and X (n, D), the features of each frame in time order.
    feat: mask (outline pooled to 12x16), kp (the joint features above) or both."""
    results, data = Path(results).expanduser(), Path(data).expanduser()
    metas = []
    for m in sorted((data / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta["person"].lower() in {e.lower() for e in persons_exclude}:
            continue
        meta["dir"] = m.parent
        meta["n"] = len(np.load(m.parent / "frame_timestamps.npy"))
        metas.append(meta)
    recs = []
    for pf in sorted(results.glob("pred_person_*.npz")):
        z = np.load(pf, allow_pickle=False)
        person = pf.stem.replace("pred_person_", "")
        mine = [m for m in metas if m["person"] == person]
        arr = z["masks"] if source == "pred" else z["gt_masks"]
        arr = arr.astype(np.float32)
        kp = None
        if feat != "mask":
            if source != "pred":
                raise SystemExit("--feat kp / both needs the predicted heatmaps (--source pred)")
            kp = kp_features(z["heatmaps"])
        if sum(m["n"] for m in mine) != len(arr):
            raise SystemExit(f"{pf.name}: {len(arr)} predicted frames but the recordings of {person} have "
                             f"{sum(m['n'] for m in mine)} - the order or the count does not match, check that --results and --data belong together")
        i = 0
        for m in mine:
            a = arr[i:i + m["n"]]
            i += m["n"]
            n, h, w = a.shape
            pooled = a[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).mean((2, 4))
            parts = []
            if feat in ("mask", "both"):
                parts.append(pooled.reshape(n, -1))
            if feat in ("kp", "both"):
                parts.append(kp[i - m["n"]:i])
            recs.append(dict(person=person, act=m["activity"], rec=m["dir"].name, X=np.concatenate(parts, 1)))
    if not recs:
        raise SystemExit(f"no pred_person_*.npz files found in {results}")
    return recs


def make_windows(recs, T, hop, velocity, center):
    X, y, person, rec = [], [], [], []
    for r in recs:
        a = r["X"].reshape(len(r["X"]), -1)
        if center:
            a = a - a.mean(0, keepdims=True)        # subtract the recording's own average, no labels needed
        if velocity:
            v = np.diff(a, axis=0, prepend=a[:1])
            a = np.concatenate([a, v], axis=1)
        for s in range(0, len(a) - T + 1, hop):
            X.append(a[s:s + T])
            y.append(ACTS.index(r["act"]))
            person.append(r["person"])
            rec.append(r["rec"])
    return np.array(X, np.float32), np.array(y), np.array(person), np.array(rec)


# ----------------------------------------------------------------------------- spiking network
class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2          # fast-sigmoid surrogate gradient, slope 10


class SNN(nn.Module):
    def __init__(self, n_in, h1=256, h2=128, n_out=3, beta=0.9, thr=1.0):
        super().__init__()
        self.f1, self.f2, self.f3 = nn.Linear(n_in, h1), nn.Linear(h1, h2), nn.Linear(h2, n_out)
        self.beta, self.thr = beta, thr
        self.rate2 = 0.0

    def forward(self, x):                           # x: (B, T, F), fed as a current at every time step
        B, T, _ = x.shape
        v1 = x.new_zeros(B, self.f1.out_features)
        v2 = x.new_zeros(B, self.f2.out_features)
        out = x.new_zeros(B, self.f3.out_features)
        s2_sum = 0.0
        for t in range(T):
            v1 = self.beta * v1 + self.f1(x[:, t])
            s1 = Spike.apply(v1 - self.thr)
            v1 = v1 - s1 * self.thr
            v2 = self.beta * v2 + self.f2(s1)
            s2 = Spike.apply(v2 - self.thr)
            v2 = v2 - s2 * self.thr
            out = self.beta * out + self.f3(s2)     # leaky readout
            s2_sum = s2_sum + s2.mean().item()
        self.rate2 = s2_sum / T
        return out


def train_eval(Xtr, ytr, Xte, args, dev):
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(args.seed)
    net = SNN(Xtr.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), args.batch):
            b = perm[i:i + args.batch]
            loss = lossf(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        tr_acc = np.mean(np.concatenate([net(Xt[i:i + 256]).argmax(1).cpu().numpy() for i in range(0, len(Xt), 256)]) == ytr)
        pred = np.concatenate([net(torch.tensor(Xte[i:i + 256], device=dev)).argmax(1).cpu().numpy()
                               for i in range(0, len(Xte), 256)])
    return pred, tr_acc, net.rate2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="~/percom/siwis_results")
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--source", choices=["pred", "gt"], default="pred")
    ap.add_argument("--velocity", action="store_true")
    ap.add_argument("--feat", choices=["mask", "kp", "both"], default="mask",
                    help="mask = pooled body outline, kp = arm joints from the predicted heatmaps, both = the two together")
    ap.add_argument("--no_center", action="store_true")
    ap.add_argument("--window", type=int, default=10, help="frames per window (2 fps -> 10 = 5 s)")
    ap.add_argument("--hop", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    recs = load_frames(args.results, args.data, args.source, args.exclude, args.feat)
    X, y, person, rec = make_windows(recs, args.window, args.hop, args.velocity, not args.no_center)
    persons = sorted(set(person))
    tag = f"source={args.source}{' (control: poses taken from the video, not a real result)' if args.source == 'gt' else ' (Wi-Fi only)'}" \
          f" | features={args.feat} | velocity={'on' if args.velocity else 'off'} | {len(X)} windows of {args.window} frames | persons {persons}"
    print(tag)
    P = np.zeros(len(y), int)
    for p in persons:
        te = person == p
        pred, tr_acc, rate = train_eval(X[~te], y[~te], X[te], args, dev)
        P[te] = pred
        print(f"  test {p:10s} window acc {np.mean(pred == y[te]) * 100:5.1f}%   (train acc {tr_acc * 100:5.1f}%, layer-2 spike rate {rate * 100:4.1f}%)",
              flush=True)
    ok = tot = 0
    for r in sorted(set(rec)):
        s = rec == r
        ok += np.bincount(P[s], minlength=3).argmax() == y[s][0]
        tot += 1
    print("\n=== leave-one-person-out, pooled over the four test people ===")
    print(f"window accuracy     {np.mean(P == y) * 100:5.1f}%   chance 33.3%")
    print(f"recording accuracy  {ok}/{tot} = {ok / tot * 100:5.1f}%   chance 33.3%")
    print("\nconfusion matrix, row % (true -> predicted)")
    print("            " + " ".join(f"{a:>7s}" for a in ACTS))
    for i, a in enumerate(ACTS):
        s = y == i
        print(f"true {a:5s}  " + " ".join(f"{np.mean(P[s] == j) * 100:7.1f}" for j in range(3)))


if __name__ == "__main__":
    main()
