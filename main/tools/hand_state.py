#!/usr/bin/env python3
"""
hand_state.py - a side experiment: can the Wi-Fi signal tell which hand is up? (CPU only, a few minutes)
It uses a gradient-boosting classifier on hand-made signal features instead of the SiWiS model, and was
our check of how much information about the hand is in the raw signal at all.

Task per 1.7 s window (centred on every labelled video frame):
    0 none | 1 left hand up | 2 right hand up | 3 both
The ground truth comes from the video keypoints (wrist above elbow, the same rule as label_check.py) and not
from the recording's gesture name, so rest frames inside a recording are labelled correctly.

Features (from the converted npz, both RX antennas): per antenna amplitude level and spread, the phase DIFFERENCE
between the antennas (direction cue), band powers (0.5-2, 2-5, 5-10, 10-20, 20-50 Hz), antenna correlation.
Each feature is z-scored inside its own recording (no labels needed), which removes offsets between people
and sessions.

Evaluation: leave-one-person-out. Reports macro-F1 and balanced accuracy next to the majority-class guess, a
"which hand" test on single-hand frames (chance 50%), a recording-level activity vote, and the confusion matrix.

    python hand_state.py --data ~/percom/siwis_data --npz_dir ~/percom/work/npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

LE, RE, LW, RW = 7, 8, 9, 10
BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 50)]
NAMES = ["none", "left", "right", "both"]


def streams(npz_path, rate):
    """-> grid (absolute epoch s), array (K, 5): ampA, ampB, stdA, stdB, phase diff (unwrapped)."""
    z = np.load(npz_path, allow_pickle=False)
    data, t = z["data"], z["t"].astype(np.float64)
    t0 = float(z["t0_epoch"])
    o = np.argsort(t)
    data, t = data[o], t[o]
    A, B = data[:, 0::2], data[:, 1::2]
    aA, aB = np.abs(A).astype(np.float32), np.abs(B).astype(np.float32)
    mA, mB = aA.mean(1), aB.mean(1)
    ok = (mA > 0) & (mB > 0)
    stdA = aA.std(1) / np.where(mA == 0, 1, mA)
    stdB = aB.std(1) / np.where(mB == 0, 1, mB)
    pd = np.angle((A * np.conj(B)).sum(1))
    cols = [mA, mB, stdA, stdB, pd]
    tt = (t0 + t)[ok]
    pd_u = np.unwrap(pd[ok])
    cols = [mA[ok], mB[ok], stdA[ok], stdB[ok], pd_u]
    grid = np.arange(tt[0], tt[-1], 1.0 / rate)
    return grid, np.stack([np.interp(grid, tt, c) for c in cols], 1)


def window_features(seg, rate):
    """seg: (n, 5) -> feature vector."""
    f = []
    seg = seg - seg.mean(0, keepdims=True)
    d = np.diff(seg, axis=0)
    f += list(seg.std(0)) + list(np.ptp(seg, axis=0)) + list(np.abs(d).mean(0))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))[:, None], axis=0)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1.0 / rate)
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        f += list(np.log1p(spec[m].sum(0)))
    a, b = seg[:, 0], seg[:, 1]
    f.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0)
    f.append((a - b).std())
    return np.nan_to_num(np.array(f, np.float64))


def hand_labels(rec):
    xy = np.load(rec / "kp_xy.npy")[:, 0]
    vis = np.load(rec / "kp_vis.npy")[:, 0].astype(bool)
    lw = vis[:, LW] & vis[:, LE] & (xy[:, LW, 1] < xy[:, LE, 1])
    rw = vis[:, RW] & vis[:, RE] & (xy[:, RW, 1] < xy[:, RE, 1])
    return lw.astype(int) + 2 * rw.astype(int)


def build(args):
    root = Path(args.data).expanduser() / "rec"
    X, y, person, rec_id, act = [], [], [], [], []
    for m in sorted(root.glob("*/meta.json")):
        d = m.parent
        meta = json.loads(m.read_text())
        npz = Path(args.npz_dir).expanduser() / meta["source_npz"]
        if not npz.exists():
            print("skip (npz missing):", d.name)
            continue
        grid, S = streams(npz, args.rate)
        ts = np.load(d / "frame_timestamps.npy")
        lab = hand_labels(d)
        n = int(round(args.window * args.rate))
        feats, labs = [], []
        for k, tk in enumerate(ts):
            i0 = int(np.searchsorted(grid, tk - args.window / 2))
            if i0 < 0 or i0 + n > len(grid):
                continue
            feats.append(window_features(S[i0:i0 + n], args.rate))
            labs.append(lab[k])
        if len(feats) < 20:
            continue
        F = np.array(feats)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)          # per-recording z-score, no labels used
        X.append(F)
        y += labs
        person += [meta["person"]] * len(F)
        rec_id += [d.name] * len(F)
        act += [meta["activity"]] * len(F)
        print(f"{d.name:20s} {len(F):4d} windows | hand-up share {np.mean(np.array(labs) > 0) * 100:3.0f}%", flush=True)
    return np.concatenate(X), np.array(y), np.array(person), np.array(rec_id), np.array(act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz")
    ap.add_argument("--out", default="~/percom/siwis_results")
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--window", type=float, default=1.7)
    args = ap.parse_args()
    X, y, person, rec_id, act = build(args)
    persons = sorted(set(person))
    print(f"\n{len(X)} windows, {X.shape[1]} features, persons {persons}")
    print("class counts:", {NAMES[c]: int((y == c).sum()) for c in range(4)})

    P = np.zeros(len(y), int)
    prob = np.zeros((len(y), 4))
    folds = []
    for p in persons:
        te, tr = person == p, person != p
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
        clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        pr = clf.predict_proba(X[te])
        pp = np.zeros((te.sum(), 4))
        pp[:, clf.classes_] = pr
        prob[te] = pp
        P[te] = pp.argmax(1)
        maj = np.bincount(y[tr], minlength=4).argmax()
        folds.append((p, f1_score(y[te], P[te], average="macro"), balanced_accuracy_score(y[te], P[te]),
                      f1_score(y[te], np.full(te.sum(), maj), average="macro")))
        print(f"fold {p:10s} macro-F1 {folds[-1][1]:.3f}  balanced acc {folds[-1][2]:.3f}  (majority-class F1 {folds[-1][3]:.3f})", flush=True)

    print("\n=== leave-one-person-out, pooled ===")
    print(f"4-class macro-F1      model {f1_score(y, P, average='macro'):.3f}   majority guess {np.mean([f[3] for f in folds]):.3f}")
    print(f"4-class balanced acc  model {balanced_accuracy_score(y, P):.3f}   chance 0.250")
    anyup = (y > 0)
    print(f"hand up vs none       balanced acc {balanced_accuracy_score(anyup, P > 0):.3f}   chance 0.500")
    one = (y == 1) | (y == 2)
    if one.any():
        side = np.where(prob[one][:, 1] >= prob[one][:, 2], 1, 2)
        print(f"left vs right (only frames with exactly one hand up)  acc {np.mean(side == y[one]):.3f}   chance 0.500")
    cm = confusion_matrix(y, P, labels=range(4), normalize="true") * 100
    print("\nconfusion matrix, row % (true -> predicted)")
    print("            " + " ".join(f"{n:>7s}" for n in NAMES))
    for i, n in enumerate(NAMES):
        print(f"true {n:6s} " + " ".join(f"{v:7.1f}" for v in cm[i]))

    # recording-level vote: which activity does the share of predicted frames point to?
    ok, tot = 0, 0
    for r in sorted(set(rec_id)):
        s = rec_id == r
        share = {"bhu": np.mean(P[s] == 3), "lhu": np.mean(P[s] == 1), "rhu": np.mean(P[s] == 2)}
        guess = max(share, key=share.get)
        ok += guess == act[s][0]
        tot += 1
    print(f"\nrecording-level activity vote: {ok}/{tot} correct = {ok / tot * 100:.1f}%   chance 33.3%")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "hand_state.json").write_text(json.dumps(dict(
        macro_f1=float(f1_score(y, P, average="macro")), balanced_acc=float(balanced_accuracy_score(y, P)),
        recording_vote=f"{ok}/{tot}", folds=[dict(person=f[0], macro_f1=f[1], bal_acc=f[2]) for f in folds]), indent=2))


if __name__ == "__main__":
    main()
