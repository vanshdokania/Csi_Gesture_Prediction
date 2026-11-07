#!/usr/bin/env python3
"""
segment_pipeline.py - the final experiment: within-person, one complete gesture at a time.

    Wi-Fi --(SiWiS)--> body mask + 17 joint heatmaps per frame --> compact pose features --(SNN)--> bhu / lhu / rhu,
    one answer per gesture

A segment is one complete repetition of the gesture (video_segments.py cuts them, using the video). For each
person separately, in each of their recordings, the segments are taken in time order:
    the first ~80 % of the segments  -> training   (then one segment is thrown away as a gap)
    the last  ~20 % of the segments  -> testing
So a test segment is complete, was never seen during training, and shares no frame and no CSI window with any
training segment. Accuracy = correctly classified test segments / test segments.

How we keep the test honest (and where this is done in the code):
  1. Segments are split as whole units, in time order, with one segment dropped between train and test.
  2. The pose model (SiWiS) is trained only on frames of training segments. The poses of the test segments come
     from a model that never saw them. The poses of the training segments are out-of-fold predictions: groups of
     training segments are held out one after the other, and the segments next to a held-out group are also left
     out of that fold's training.
  3. The statistics of the CSI input (the z-score and the angle-of-arrival offset) only use CSI recorded before the
     first test window.
  4. The statistics of the SNN input (the average pose that is subtracted per recording) only use training segments.
  5. Epochs, learning rates and so on were fixed in advance. Nothing was chosen by looking at the test result.
The video is used (a) as the label of the pose model on training segments and (b) by video_segments.py to decide
where to cut. It is never an input of the final classifier.

Outputs in --out: poses_<Person>/ (predicted pose features, split.json), seg_siwis.json, seg_results.json, seg_predictions.csv

    python segment_pipeline.py --persons Vansh --epochs 3 --oof_folds 2 --workers 0 --seeds 1 --snn_epochs 5    # quick test
    python segment_pipeline.py                                   # everything (about 2 h on one H200)
    python segment_pipeline.py --skip_siwis                      # only the SNN again, on the saved poses (minutes)
    python segment_pipeline.py --control                         # control: poses taken from the video (99-100 % in the report)

Run it from the SiWiS repository folder (scripts/setup_siwis.sh sets it up) after video_segments.py has written
segments.json.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import config
import percom_common as pc
import pose_snn as ps
import train_percom as tp
import train_percom_v2 as v2
import within_pipeline as wp

ACTS = ps.ACTS


# ----------------------------------------------------------------------------- the split
def split_segments(n, test_frac):
    """n segments in time order -> dict(train=[...], guard=[i], test=[...]); None if too few."""
    if n < 3:
        return None
    n_test = max(1, int(round(test_frac * n)))
    n_train = n - n_test - 1
    if n_train < 1:
        n_test, n_train = 1, n - 2
    return dict(train=list(range(n_train)), guard=[n_train], test=list(range(n_train + 1, n)))


def frames_of(segs, idx):
    out = []
    for i in idx:
        out += list(range(segs[i][0], segs[i][1]))
    return out


def load_all_seg(args, cols, segs_all):
    recs = []
    for m in sorted((Path(args.data).expanduser() / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta["person"].lower() in {e.lower() for e in args.exclude}:
            continue
        if args.persons and meta["person"].lower() not in {p.lower() for p in args.persons}:
            continue
        sid = meta["sid"]
        if sid not in segs_all:
            print(f"  skip {sid}: not in segments.json (run video_segments.py first)")
            continue
        S = segs_all[sid]["segments"]
        sp = split_segments(len(S), args.test_frac)
        if sp is None:
            print(f"  skip {sid}: only {len(S)} segments (need at least 3: train, gap, test)")
            continue
        ts = np.load(m.parent / "frame_timestamps.npy")
        if len(ts) != segs_all[sid]["n_frames"]:
            raise SystemExit(f"{sid}: segments.json was made for {segs_all[sid]['n_frames']} frames but the recording has "
                             f"{len(ts)} - run video_segments.py again")
        first_test = S[sp["test"][0]][0]
        ref_until = float(ts[first_test]) - config.SIGNAL_DURATION_SEC      # CSI statistics are computed from before the test windows only
        print(f"  features {sid}: {len(S)} segments -> train {len(sp['train'])}, gap 1, test {len(sp['test'])}", flush=True)
        meta.update(ds=v2.load_recording_v2(m.parent, meta, args, cols, ref_until), seg=S, split=sp, ts=ts)
        recs.append(meta)
    if not recs:
        raise SystemExit("no usable recordings")
    return recs


# ----------------------------------------------------------------------------- stage 1: SiWiS per person
def siwis_for_person(person, recs, args, dev, n_ch, out):
    mine = [r for r in recs if r["person"] == person]
    K = args.oof_folds
    store = {r["sid"]: dict(mask=np.zeros((len(r["ds"]), 192), np.float32), kp=np.zeros((len(r["ds"]), 36), np.float32))
             for r in mine}

    def put(pieces, pred):
        pooled, kp = wp.frame_features(pred)               # pooled 12x16 mask (192) and 36 joint features
        i = 0
        for r, _, idx in pieces:
            store[r["sid"]]["mask"][idx] = pooled[i:i + len(idx)]
            store[r["sid"]]["kp"][idx] = kp[i:i + len(idx)]
            i += len(idx)

    for k in range(K):                                          # ---- out-of-fold poses of the training segments
        tr, ho = [], []
        for r in mine:
            tri = np.array(r["split"]["train"])
            g = np.array_split(tri, K)[k]
            if len(g) == 0:
                continue
            banned = set(g.tolist()) | {int(g.min()) - 1, int(g.max()) + 1}   # the neighbours of the held-out group are left out as well
            tr.append((r, r["ds"], frames_of(r["seg"], [i for i in tri.tolist() if i not in banned])))
            ho.append((r, r["ds"], frames_of(r["seg"], g.tolist())))
        if not ho:
            continue
        if sum(len(x) for _, _, x in tr) == 0:
            raise SystemExit(f"{person}: too few training segments for --oof_folds {K}; use a smaller --oof_folds")
        model = wp.fit(tr, args, dev, n_ch, f"{person} oof {k + 1}/{K}")
        put(ho, tp.predict(model, ho, args.batch_size, dev))
        del model

    tr = [(r, r["ds"], frames_of(r["seg"], r["split"]["train"])) for r in mine]        # ---- final model -> test segments
    te = [(r, r["ds"], frames_of(r["seg"], r["split"]["test"])) for r in mine]
    model = wp.fit(tr, args, dev, n_ch, f"{person} final")
    pred = tp.predict(model, te, args.batch_size, dev)
    put(te, pred)
    pts = pc.argmax_points(pred["pp"])
    res = dict(person=person, n_train=sum(len(x) for _, _, x in tr), n_test=sum(len(x) for _, _, x in te),
               model=tp.score(pred["pm"], pred["gm"], pts, pred["gc"], pred["gv"]), baseline=tp.baseline(tr, pred))
    print(f"  {person}: SiWiS on the test SEGMENTS ({res['n_test']} frames) [trained on {res['n_train']} frames of {person} only]")
    print("    MODEL     :", tp.fmt(res["model"]))
    print("    MEAN-POSE :", tp.fmt(res["baseline"]), " <- the person's own average pose, no CSI")

    d = out / f"poses_{person}"
    d.mkdir(parents=True, exist_ok=True)
    info = []
    for r in mine:
        np.savez(d / f"{r['sid']}_feat.npz", mask=store[r["sid"]]["mask"], kp=store[r["sid"]]["kp"], ts=r["ts"])
        info.append(dict(sid=r["sid"], act=r["activity"], segments=r["seg"], **r["split"]))
    (d / "split.json").write_text(json.dumps(info))
    return res


# ----------------------------------------------------------------------------- stage 2: SNN on whole segments
class SegSNN(nn.Module):
    """Per frame: Linear -> 256 LIF neurons -> 128 LIF neurons -> leaky read-out (does not spike).
    The read-out is averaged over all real frames of the segment, which gives one answer per gesture."""

    def __init__(self, d_in, h1=256, h2=128, beta=0.9, thr=1.0, drop=0.3):
        super().__init__()
        self.beta, self.thr = beta, thr
        self.drop = nn.Dropout(drop)
        self.f1, self.f2, self.f3 = nn.Linear(d_in, h1), nn.Linear(h1, h2), nn.Linear(h2, 3)

    def forward(self, x, m):                    # x (B, T, F), m (B, T) True = real frame (segments differ in length)
        B, T, _ = x.shape
        cur = self.f1(self.drop(x))
        v1 = cur.new_zeros(B, self.f1.out_features)
        v2 = cur.new_zeros(B, self.f2.out_features)
        out = cur.new_zeros(B, 3)
        acc = cur.new_zeros(B, 3)
        for t in range(T):
            mt = m[:, t:t + 1].float()
            n1 = self.beta * v1 + cur[:, t]
            s1 = ps.Spike.apply(n1 - self.thr)
            n1 = n1 - s1 * self.thr
            n2 = self.beta * v2 + self.f2(s1)
            s2 = ps.Spike.apply(n2 - self.thr)
            n2 = n2 - s2 * self.thr
            no = self.beta * out + self.f3(s2)
            v1, v2, out = mt * n1 + (1 - mt) * v1, mt * n2 + (1 - mt) * v2, mt * no + (1 - mt) * out
            acc = acc + mt * out
        return acc / m.sum(1, keepdim=True).clamp(min=1)


def gt_feats(rec_dir):
    """Control experiment: the same two feature sets, but computed from the video labels instead of the Wi-Fi prediction."""
    mask = np.load(rec_dir / "mask_labels.npy").astype(np.float32)
    lz = pc.LazyHeatmaps(np.load(rec_dir / "kp_xy.npy"), np.load(rec_dir / "kp_vis.npy"))
    heat = np.stack([lz[i] for i in range(len(mask))]).astype(np.float32)
    n, h, w = mask.shape
    pooled = mask[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).mean((2, 4)).reshape(n, -1)
    return pooled, ps.kp_features(heat)


def load_person(person, out, source, data):
    d = out / f"poses_{person}"
    recs = []
    for e in json.loads((d / "split.json").read_text()):
        z = np.load(d / f"{e['sid']}_feat.npz")
        e["ts"] = z["ts"]
        e["mask"], e["kp"] = (z["mask"], z["kp"]) if source == "pred" else gt_feats(Path(data).expanduser() / "rec" / e["sid"])
        recs.append(e)
    return recs


def build_inputs(recs, kind, args, dev):
    """-> per-recording tensors, train items, test items (ri, start, end, label, segment no.), scale.
    The per-recording mean pose and the feature scale come from the TRAINING segments only."""
    tens, tr_items, te_items, tr_rows = [], [], [], []
    for ri, e in enumerate(recs):
        tr_f = frames_of(e["segments"], e["train"])
        X = {"kp": e["kp"], "mask": e["mask"], "both": np.concatenate([e["mask"], e["kp"]], 1)}[kind]
        if not args.no_center:
            X = X - X[tr_f].mean(0, keepdims=True)
        tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
        tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
        y = ACTS.index(e["act"])
        for i in e["train"]:
            tr_items.append((ri, *e["segments"][i], y, i))
        for i in e["test"]:
            te_items.append((ri, *e["segments"][i], y, i))
    sd = np.concatenate(tr_rows).std(0) + 1e-6                # z-score with training statistics
    D = sd.shape[0] // 2
    scale = torch.from_numpy((sd if not args.no_velocity else sd[:D]).astype(np.float32)).to(dev)
    return tens, tr_items, te_items, scale


def batch_of(items, tens, args, dev, train, rng, scale):
    seqs = []
    for ri, s, e, y, _ in items:
        if train and args.augment:
            a, b = rng.integers(0, 3), rng.integers(0, 3)           # trim up to 1 s of rest at each end
            if e - s - a - b >= 6:
                s, e = s + a, e - b
        x = tens[ri][s:e]
        if not args.no_velocity:
            x = torch.cat([x, x - torch.cat([x[:1], x[:-1]], 0)], 1)
        x = x / scale
        if train and args.augment:
            x = x + 0.05 * torch.randn_like(x)
        seqs.append(x)
    T = max(len(x) for x in seqs)
    X = torch.zeros(len(seqs), T, seqs[0].shape[1], device=dev)
    M = torch.zeros(len(seqs), T, dtype=torch.bool, device=dev)
    for i, x in enumerate(seqs):
        X[i, :len(x)] = x
        M[i, :len(x)] = True
    return X, M, torch.tensor([it[3] for it in items], device=dev)


def run_snn_once(tens, tr_items, te_items, scale, args, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SegSNN(int(scale.numel())).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.snn_lr, weight_decay=1e-4)
    ytr = np.array([it[3] for it in tr_items])
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(args.snn_epochs):
        net.train()
        perm = rng.permutation(len(tr_items))
        for i in range(0, len(perm), args.snn_batch):
            X, M, y = batch_of([tr_items[j] for j in perm[i:i + args.snn_batch]], tens, args, dev, True, rng, scale)
            loss = lossf(net(X, M), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    pred, tr_pred = [], []
    with torch.no_grad():
        for items, store in ((te_items, pred), (tr_items, tr_pred)):
            for i in range(0, len(items), 16):
                X, M, y = batch_of(items[i:i + 16], tens, args, dev, False, rng, scale)
                store += net(X, M).argmax(1).cpu().tolist()
    return np.array(pred), float(np.mean(np.array(tr_pred) == ytr))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


LABELS = {"kp": "6 arm joints (36 numbers per frame)", "mask": "body mask pooled to 12x16 (192 numbers per frame)",
          "both": "mask + joints (228 numbers per frame)"}


def run_snn(persons, args, out, dev):
    sources = ["pred"] + (["gt"] if args.control else [])
    rows, summary = [], {}
    for source in sources:
        print("\n" + "=" * 100)
        print("SNN, SAME PERSON, whole segments: trained on the first ~80% of the segments, tested on the last ~20%" +
              ("   [CSI only]" if source == "pred" else "\nCONTROL: poses from the VIDEO (not Wi-Fi). Not a result - it only shows what the SNN can do with a correct pose."))
        print("=" * 100)
        data = {p: load_person(p, out, source, args.data) for p in persons}
        for kind in args.snn_models:
            per_seed, cm = [], np.zeros((3, 3))
            per_person, counts, tr_accs = {p: [] for p in persons}, {}, []
            for seed in range(args.seeds):
                tot_ok = tot_n = 0
                for p in persons:
                    tens, tr_items, te_items, scale = build_inputs(data[p], kind, args, dev)
                    if not te_items or not tr_items:
                        continue
                    pred, tr_acc = run_snn_once(tens, tr_items, te_items, scale, args, dev, seed)
                    tr_accs.append(tr_acc)
                    y = np.array([it[3] for it in te_items])
                    per_person[p].append(float(np.mean(pred == y)))
                    counts[p] = (len(tr_items), len(te_items))
                    tot_ok += int((pred == y).sum())
                    tot_n += len(y)
                    for i in range(3):
                        for j in range(3):
                            cm[i, j] += np.sum((y == i) & (pred == j))
                    if seed == 0 and source == "pred":
                        for it, pr in zip(te_items, pred):
                            e = data[p][it[0]]
                            rows.append([p, e["sid"], it[4] + 1, round(float(e["ts"][it[1]] - e["ts"][0]), 1),
                                         round(float(e["ts"][min(it[2], len(e["ts"])) - 1] - e["ts"][0]), 1),
                                         ACTS[it[3]], ACTS[pr], kind])
                per_seed.append(tot_ok / max(tot_n, 1))
            N = sum(c[1] for c in counts.values())
            m = float(np.mean(per_seed))
            lo, hi = wilson(m * N, N)
            print(f"\ninput: {LABELS[kind]}   + velocity: {'no' if args.no_velocity else 'yes'}   (accuracy on its own training segments: {np.mean(tr_accs) * 100:.0f}%)")
            for p in persons:
                if per_person[p]:
                    print(f"   {p:10s} train {counts[p][0]:3d} / test {counts[p][1]:3d} segments   accuracy {np.mean(per_person[p]) * 100:5.1f}%  "
                          f"({np.mean(per_person[p]) * counts[p][1]:.1f} of {counts[p][1]})")
            print(f"   ALL        {N} test segments   accuracy {m * 100:5.1f}% +- {np.std(per_seed) * 100:.1f} over {args.seeds} seed(s)"
                  f"   (about 95% range {lo * 100:.0f}-{hi * 100:.0f}%, chance 33.3%)")
            print("   confusion matrix, row % (true -> predicted)")
            print("                " + " ".join(f"{a:>7s}" for a in ACTS))
            for i, a in enumerate(ACTS):
                print(f"   true {a:5s}    " + " ".join(f"{cm[i, j] / max(cm[i].sum(), 1) * 100:7.1f}" for j in range(3)))
            summary[f"{source}_{kind}"] = dict(accuracy=m, std=float(np.std(per_seed)), n_test=N, ci95=[lo, hi],
                                               per_person={p: float(np.mean(v)) for p, v in per_person.items() if v})
    (out / "seg_results.json").write_text(json.dumps(summary, indent=2))
    with open(out / "seg_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "recording", "segment", "start_s", "end_s", "true", "predicted", "input"])
        w.writerows(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz")
    ap.add_argument("--segments", default="", help="segments.json (default: <data>/segments.json)")
    ap.add_argument("--out", default="~/percom/siwis_results_seg")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase", "aoa", "vel"], choices=list(v2.GROUPS))
    ap.add_argument("--persons", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60, help="SiWiS epochs")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose_weight", type=float, default=500.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--oof_folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_siwis", action="store_true", help="reuse the saved poses, run only the SNN")
    ap.add_argument("--skip_snn", action="store_true")
    ap.add_argument("--snn_models", nargs="+", default=["kp", "both", "mask"], choices=["kp", "mask", "both"],
                    help="kp = 6 arm-joint features, mask = pooled body mask, both")
    ap.add_argument("--control", action="store_true", help="also run the SNN on VIDEO-derived poses (a control, not a result)")
    ap.add_argument("--snn_epochs", type=int, default=60)
    ap.add_argument("--snn_lr", type=float, default=1e-3)
    ap.add_argument("--snn_batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no_velocity", action="store_true")
    ap.add_argument("--no_center", action="store_true")
    ap.add_argument("--no_augment", dest="augment", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cols = sorted({c for g in args.channels for c in v2.GROUPS[g]})
    print(f"device {dev} | CSI channels {[v2.CHANNELS[c] for c in cols]} | SiWiS epochs {args.epochs} | oof folds {args.oof_folds}")
    if not args.skip_siwis:
        sp = Path(args.segments or Path(args.data) / "segments.json").expanduser()
        if not sp.exists():
            raise SystemExit(f"{sp} not found - run video_segments.py first")
        recs = load_all_seg(args, cols, json.loads(sp.read_text()))
        persons = sorted({r["person"] for r in recs})
        print(f"{len(recs)} recordings | persons {persons}")
        results = []
        for p in persons:
            print(f"\n######## {p}: first ~80% of the segments of every recording train, last ~20% test")
            results.append(siwis_for_person(p, recs, args, dev, len(cols), out))
        if len(results) > 1:
            print("\n=== SiWiS, test segments, mean over persons ===")
            keys = results[0]["model"].keys()
            for name in ("model", "baseline"):
                print(f"  {name:9s}:", tp.fmt({k: float(np.nanmean([r[name][k] for r in results])) for k in keys}))
        pc.save_json(out / "seg_siwis.json", results)
    else:
        persons = sorted(p.name.replace("poses_", "") for p in out.glob("poses_*"))
        if args.persons:
            persons = [p for p in persons if p.lower() in {q.lower() for q in args.persons}]
    if not args.skip_snn and persons:
        run_snn(persons, args, out, dev)
    print(f"\nSaved in {out}")


if __name__ == "__main__":
    main()
