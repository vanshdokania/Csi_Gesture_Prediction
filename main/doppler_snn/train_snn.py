#!/usr/bin/env python3
"""
train_snn.py - the first classifier we tried: a spiking neural network (SNN) that looks at
Doppler heatmaps and decides between bhu / lhu / rhu.

    python3 doppler_snn/train_snn.py              # train on 2 people, validate on 1, test on 1 person never seen before
    python3 doppler_snn/train_snn.py --mode loso  # leave-one-person-out: every person is the test person once, then average

The report numbers (about 35 % per frame, 44 % per recording) come from the --mode loso run.

What it reads (everything comes from make_dataset.py):
    ~/percom/dataset/index.csv, labels_all.csv         which frames exist, their label, valid flag, person
    ~/percom/work/output/doppler/<source>.npz           the Doppler spectrogram (doppler_db) of each recording
A frame = 1 s = 100 time steps x Doppler bins. The frame is fed to the network one time step at a time
(10 ms each), every step being the Doppler spectrum at that moment, so the spikes carry the timing
of the movement.

Model (pure PyTorch, no extra SNN library):
    input spectrum -> Linear -> LIF neurons -> Linear -> LIF neurons -> Linear -> leaky read-out (class scores)
    LIF = leaky integrate-and-fire neuron. A fast-sigmoid surrogate gradient (slope 25) lets it train
    with normal back-propagation.

Evaluation: frames of the same recording overlap and look alike, so a random frame split would be far
too optimistic. The split is therefore by person. We report the frame accuracy, the whole-recording
accuracy (majority vote over the frames of one recording), the confusion matrix and the average spike
rate (how sparse, i.e. how cheap, the network is).

People: Divij and Nikita are left out by default (see --exclude), which leaves the four people used
in the report.

Needs: pip install torch numpy        (GPU is used automatically if available)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- data
def load_data(args):
    ds = Path(args.dataset).expanduser()
    pre = Path(args.pre).expanduser()
    index = {r["sample_id"]: r for r in csv.DictReader(open(ds / "index.csv"))}
    classes = json.loads((ds / "classes.json").read_text())
    names = [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]
    rows_by_sample = {}
    for r in csv.DictReader(open(ds / "labels_all.csv")):
        if args.use_invalid or r["valid"] == "1":
            rows_by_sample.setdefault(r["sample_id"], []).append(r)

    X, y, person, sample = [], [], [], []
    for sid, rows in rows_by_sample.items():
        src = index[sid]["source"]
        f = pre / "doppler" / f"{src}.npz"
        if not f.exists():
            print(f"  Warning: {f} is missing, skipping {sid}")
            continue
        D = np.load(f, allow_pickle=False)
        tw = D["t_windows"].astype(float)
        db = D["doppler_db"].astype(np.float32)
        freqs = D["freqs_hz"].astype(float)
        fs_w = 1.0 / float(np.median(np.diff(tw)))
        sel = np.abs(freqs) <= args.fmax
        for r in rows:
            n_t = int(round((float(r["t_end"]) - float(r["t_start"])) * fs_w)) + 1
            s = int(np.searchsorted(tw, float(r["t_start"]) - 1e-6))
            if s + n_t > len(tw):
                continue
            X.append(db[s:s + n_t][:, sel])
            y.append(int(r["label_id"]))
            person.append(r["person"])
            sample.append(sid)
    if not X:
        raise SystemExit("no frames loaded - check --dataset / --pre paths")
    T = min(x.shape[0] for x in X)
    X = np.stack([x[:T] for x in X])
    X = np.clip((X + 35.0) / 35.0, 0.0, 1.0).astype(np.float32)        # dB in [-35, 0] -> [0, 1]
    return X, np.array(y), np.array(person), np.array(sample), names


# ----------------------------------------------------------------------------- SNN
class SpikeFn(torch.autograd.Function):
    """Heaviside spike forward, fast-sigmoid surrogate gradient backward."""
    slope = 25.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (SpikeFn.slope * v.abs() + 1.0) ** 2


spike = SpikeFn.apply


class LIF(nn.Module):
    def __init__(self, n, beta=0.9, thr=1.0, learn_beta=True):
        super().__init__()
        b = torch.full((n,), float(beta))
        if learn_beta:
            self.beta = nn.Parameter(b)
        else:
            self.register_buffer("beta", b)
        self.thr = thr

    def forward(self, cur, mem):
        beta = self.beta.clamp(0.0, 0.99)
        mem = beta * mem + cur
        s = spike(mem - self.thr)
        mem = mem - s * self.thr                         # soft reset
        return s, mem


class SNN(nn.Module):
    def __init__(self, n_in, n_cls, h1=256, h2=128, drop=0.3, beta=0.9):
        super().__init__()
        self.fc1, self.lif1 = nn.Linear(n_in, h1), LIF(h1, beta)
        self.fc2, self.lif2 = nn.Linear(h1, h2), LIF(h2, beta)
        self.fc3 = nn.Linear(h2, n_cls)
        self.beta_out = 0.9
        self.drop = nn.Dropout(drop)

    def forward(self, x):                                # x: (B, T, F)
        B, T, _ = x.shape
        m1 = x.new_zeros(B, self.fc1.out_features)
        m2 = x.new_zeros(B, self.fc2.out_features)
        out = x.new_zeros(B, self.fc3.out_features)
        acc = x.new_zeros(B, self.fc3.out_features)
        r1 = r2 = 0.0
        for t in range(T):
            s1, m1 = self.lif1(self.fc1(x[:, t]), m1)
            s2, m2 = self.lif2(self.fc2(self.drop(s1)), m2)
            out = self.beta_out * out + self.fc3(self.drop(s2))      # non-spiking leaky read-out
            acc = acc + out
            r1 = r1 + s1.detach().mean()
            r2 = r2 + s2.detach().mean()
        return acc / T, (float(r1) / T, float(r2) / T)


# ----------------------------------------------------------------------------- train / eval
def predict(model, X, dev, bs=256):
    model.eval()
    logits, rates = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lo, r = model(torch.from_numpy(X[i:i + bs]).to(dev))
            logits.append(lo.cpu())
            rates.append(r)
    return torch.cat(logits).numpy(), np.mean(rates, axis=0)


def confusion(y, p, k):
    m = np.zeros((k, k), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def vote(sample, p, y, k):
    """whole-recording prediction = most frequent frame prediction."""
    ok, tot = 0, 0
    wrong = []
    for s in np.unique(sample):
        i = sample == s
        pred = np.bincount(p[i], minlength=k).argmax()
        tot += 1
        ok += int(pred == y[i][0])
        if pred != y[i][0]:
            wrong.append((s, int(pred)))
    return ok / max(tot, 1), wrong


def run_fold(X, y, person, sample, names, test_p, val_p, args, dev, tag):
    k = len(names)
    tr = ~np.isin(person, [test_p, val_p])
    va, te = person == val_p, person == test_p
    print(f"\n=== {tag}: test person {test_p} | val person {val_p} | "
          f"train {tr.sum()} / val {va.sum()} / test {te.sum()} frames ===")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SNN(X.shape[2], k, args.h1, args.h2, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr, ytr = X[tr], y[tr]
    best, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tl, tc = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = torch.from_numpy(Xtr[b]).to(dev)
            if args.augment:                                         # small gain change + noise (no frequency flip!)
                xb = xb + 0.05 * torch.randn(xb.shape[0], 1, 1, device=dev) + 0.02 * torch.randn_like(xb)
            yb = torch.from_numpy(ytr[b]).to(dev)
            lo, _ = model(xb)
            loss = F.cross_entropy(lo, yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(b)
            tc += int((lo.argmax(1) == yb).sum())
        sched.step()
        lv, _ = predict(model, X[va], dev)
        vacc = float((lv.argmax(1) == y[va]).mean())
        if vacc > best:
            best, best_state = vacc, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"  epoch {ep:3d}  train loss {tl / len(perm):.3f} acc {tc / len(perm):.3f}   val acc {vacc:.3f}")
    model.load_state_dict(best_state)
    lt, rate = predict(model, X[te], dev)
    pt = lt.argmax(1)
    facc = float((pt == y[te]).mean())
    racc, wrong = vote(sample[te], pt, y[te], k)
    cm = confusion(y[te], pt, k)
    print(f"  test ({test_p}): frame accuracy {facc:.3f} | whole-recording accuracy {racc:.3f} "
          f"| best val {best:.3f} | spike rate L1 {rate[0]:.3f} L2 {rate[1]:.3f}")
    print("  confusion (rows = true, cols = predicted)  " + "  ".join(names))
    for n, row in zip(names, cm):
        print(f"    {n:5s} {row}")
    if wrong:
        print("  misclassified recordings:", ", ".join(f"{s}->{names[p]}" for s, p in wrong))
    return model, dict(test_person=test_p, val_person=val_p, frame_acc=facc, recording_acc=racc, best_val=best,
                       spike_rate=[float(rate[0]), float(rate[1])], confusion=cm.tolist())


def main():
    ap = argparse.ArgumentParser(description="SNN classifier for bhu / lhu / rhu Doppler frames")
    ap.add_argument("--dataset", default="~/percom/dataset", help="folder made by make_dataset.py")
    ap.add_argument("--pre", default="~/percom/work/output", help="folder with doppler/ (preprocess output)")
    ap.add_argument("--out", default="~/percom/snn_results", help="where models and results.json go")
    ap.add_argument("--mode", choices=["holdout", "loso"], default="holdout")
    ap.add_argument("--test-person", default="", help="holdout: person kept for testing (default: last alphabetical)")
    ap.add_argument("--val-person", default="", help="holdout: person used to pick the best epoch")
    ap.add_argument("--fmax", type=float, default=40.0, help="use Doppler bins with |f| <= this, Hz (default 40)")
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"],
                    help="people left out of the experiment (default: Divij Nikita). Pass --exclude with no names to keep everyone")
    ap.add_argument("--use-invalid", action="store_true", help="also train on frames with valid=0 (dead signal)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    X, y, person, sample, names = load_data(args)
    keep = ~np.isin(person, args.exclude)
    X, y, person, sample = X[keep], y[keep], person[keep], sample[keep]
    persons = sorted(set(person))
    print(f"device {dev} | frames {X.shape} (frames, time-steps, Doppler bins) | classes {names}")
    print("frames per class:", {n: int((y == i).sum()) for i, n in enumerate(names)})
    print("persons:", {p: int((person == p).sum()) for p in persons})
    if len(persons) < 3:
        raise SystemExit("need at least 3 persons for a train / val / test split by person")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    results = []
    if args.mode == "holdout":
        test_p = args.test_person or persons[-1]
        val_p = args.val_person or [p for p in persons if p != test_p][0]
        model, res = run_fold(X, y, person, sample, names, test_p, val_p, args, dev, "holdout")
        torch.save(dict(state=model.state_dict(), names=names, args=vars(args), n_in=X.shape[2]), out / "snn_holdout.pt")
        results.append(res)
    else:
        for i, tp in enumerate(persons):
            vp = persons[(i + 1) % len(persons)]
            model, res = run_fold(X, y, person, sample, names, tp, vp, args, dev, f"fold {i + 1}/{len(persons)}")
            torch.save(dict(state=model.state_dict(), names=names, args=vars(args), n_in=X.shape[2]),
                       out / f"snn_test_{tp}.pt")
            results.append(res)
        print("\n=== leave-one-person-out summary ===")
        for r in results:
            print(f"  test {r['test_person']:10s} frame {r['frame_acc']:.3f}  recording {r['recording_acc']:.3f}")
        print(f"  MEAN       frame {np.mean([r['frame_acc'] for r in results]):.3f}  "
              f"recording {np.mean([r['recording_acc'] for r in results]):.3f}")
    (out / "results.json").write_text(json.dumps(dict(classes=names, runs=results), indent=2))
    print(f"\nSaved models + results.json in {out}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
