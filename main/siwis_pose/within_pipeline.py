#!/usr/bin/env python3
"""
within_pipeline.py - the within-person experiment on fixed 5-second windows. The whole chain runs
from the Wi-Fi signal only, and the model is trained and tested on the same person:

    CSI (AoA, velocity, amplitude, phase ...) --SiWiS--> body mask + joints per frame --SNN--> bhu / lhu / rhu

For each person separately (nobody else's data is used):
    train part = the first 80 % of each of their recordings
    test part  = the last 20 % of each recording
    3 label frames (about 1.5 s) are dropped in between, so no 1.7 s CSI window touches both parts.

Stage 1  SiWiS (pose from the Wi-Fi signal). The video is only used as the training label of this stage.
    * One final model is trained on the whole train part and predicts the test part. These are the
      poses the SNN is tested on.
    * The train part is predicted "out-of-fold": it is cut into --oof_folds contiguous blocks (4 by default),
      and each block is predicted by a model that was trained on the other blocks. That way every pose the
      SNN is trained on comes from a model that never saw that frame, so the SNN meets the same kind of
      imperfect poses in training and in testing. (If the SNN were trained on poses from a model that had
      seen those frames, they would be almost perfect, and the SNN would learn from poses it never gets
      to see again at test time.)
    This costs 4 extra pose models per person.
Stage 2  SNN: windows of --window frames (2 fps, so 10 frames = 5 s) made of the pooled body mask and/or the joint
    positions, always together with their frame-to-frame change (speed). The SNN is trained on windows every
    --train_hop frames (2) and tested on windows every --test_hop frames (5). It is run with 3 seeds.
    The script reports the window accuracy and the recording accuracy (one vote per recording's test part)
    for each feature set as mean +- standard deviation.

    python within_pipeline.py                                  # everything, all 4 people (about 1-1.5 h on one H200)
    python within_pipeline.py --persons Vansh --epochs 3 --oof_folds 2 --workers 0 --seeds 1     # quick test
    python within_pipeline.py --skip_siwis                     # run only the SNN again on the saved poses (seconds)

Run it from the SiWiS repository folder (it needs config.py, models/, losses/, and the files percom_common.py,
train_percom.py, train_percom_v2.py and pose_snn.py next to it; scripts/setup_siwis.sh sets this up).
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
import percom_common as pc
import pose_snn as ps
import train_percom as tp
import train_percom_v2 as v2
from losses.losses import mask_loss, pose_loss
from models.signal_encoder import SignalEncoder
from models.siwis_net import SiWiSNet

ACTS = ps.ACTS


# ----------------------------------------------------------------------------- stage 1: SiWiS
def fit(pieces, args, dev, n_ch, tag):
    """Train SiWiSNet (same recipe as train_percom_v2.py) on pieces = [(meta, dataset, indices)]."""
    n = sum(len(i) for _, _, i in pieces)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SiWiSNet()
    model.signal_encoder = SignalEncoder(in_channels=n_ch)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = DataLoader(ConcatDataset([Subset(d, idx) for _, d, idx in pieces if len(idx)]), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0,
                        drop_last=n >= args.batch_size)
    npix = config.HEATMAP_H * config.HEATMAP_W
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        lm_s = lp_s = 0.0
        nb = 0
        for b in loader:
            o = model(b["signal"].to(dev))
            B = o["mask"].shape[0]
            lm = mask_loss(torch.sigmoid(o["mask"]), b["mask_label"].to(dev)) / (B * npix)
            lp = pose_loss(o["pose_heatmaps"], b["keypoint_label"].to(dev), o["pose_tags"],
                           b["keypoint_coords"].to(dev), b["keypoint_visible"].to(dev))
            loss = lm + args.pose_weight * lp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lm_s += float(lm.detach())
            lp_s += float(lp.detach())
            nb += 1
        sched.step()
        if ep == 0 or (ep + 1) % max(args.epochs // 4, 1) == 0 or ep == args.epochs - 1:
            print(f"    [{tag}] epoch {ep + 1:3d}/{args.epochs}  mask {lm_s / max(nb, 1):.4f}  pose {lp_s / max(nb, 1):.5f}"
                  f"  ({time.time() - t0:.0f} s, {n} frames)", flush=True)
    return model


def frame_features(pred):
    """predict() output -> per-frame vectors: pooled 12x16 mask (192) and joint features (36)."""
    pm = pred["pm"].astype(np.float32)
    n, h, w = pm.shape
    pooled = pm[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).mean((2, 4)).reshape(n, -1)
    return pooled, ps.kp_features(pred["pp"])


def split_regions(n, args):
    cut = int(n * config.TRAIN_FRACTION)
    return cut, min(cut + args.gap, n)


def siwis_for_person(person, recs, args, dev, n_ch, out):
    mine = [r for r in recs if r["person"] == person]
    info = {r["sid"]: split_regions(len(r["ds"]), args) for r in mine}
    store = {r["sid"]: dict(mask=np.zeros((len(r["ds"]), 192), np.float32), kp=np.zeros((len(r["ds"]), 36), np.float32))
             for r in mine}
    K = args.oof_folds

    # ---- out-of-fold poses for the train part
    for k in range(K):
        tr, ho = [], []
        for r in mine:
            cut = info[r["sid"]][0]
            e = [round(j * cut / K) for j in range(K + 1)]
            idx_tr = list(range(0, max(0, e[k] - args.gap))) + list(range(min(cut, e[k + 1] + args.gap), cut))
            tr.append((r, r["ds"], idx_tr))
            ho.append((r, r["ds"], list(range(e[k], e[k + 1]))))
        model = fit(tr, args, dev, n_ch, f"{person} oof {k + 1}/{K}")
        pred = tp.predict(model, ho, args.batch_size, dev)
        pooled, kp = frame_features(pred)
        i = 0
        for r, _, idx in ho:
            store[r["sid"]]["mask"][idx] = pooled[i:i + len(idx)]
            store[r["sid"]]["kp"][idx] = kp[i:i + len(idx)]
            i += len(idx)
        del model

    # ---- final model: whole train part -> test part
    tr = [(r, r["ds"], list(range(info[r["sid"]][0]))) for r in mine]
    te = [(r, r["ds"], list(range(info[r["sid"]][1], len(r["ds"])))) for r in mine]
    model = fit(tr, args, dev, n_ch, f"{person} final")
    pred = tp.predict(model, te, args.batch_size, dev)
    pooled, kp = frame_features(pred)
    i = 0
    for r, _, idx in te:
        store[r["sid"]]["mask"][idx] = pooled[i:i + len(idx)]
        store[r["sid"]]["kp"][idx] = kp[i:i + len(idx)]
        i += len(idx)
    pts = pc.argmax_points(pred["pp"])
    res = dict(person=person, n_train=sum(len(x) for _, _, x in tr), n_test=sum(len(x) for _, _, x in te),
               model=tp.score(pred["pm"], pred["gm"], pts, pred["gc"], pred["gv"]), baseline=tp.baseline(tr, pred))
    print(f"  {person}: SiWiS test part ({res['n_test']} frames)   [model trained on {res['n_train']} frames of {person} only]")
    print("    MODEL     :", tp.fmt(res["model"]))
    print("    MEAN-POSE :", tp.fmt(res["baseline"]), " <- the person's own average pose, no CSI")
    np.savez_compressed(out / f"within_feats_{person}.npz", names=np.array([r["sid"] for r in mine]),
                        acts=np.array([r["activity"] for r in mine]),
                        cut=np.array([info[r["sid"]][0] for r in mine]), start_test=np.array([info[r["sid"]][1] for r in mine]),
                        **{f"mask_{r['sid']}": store[r["sid"]]["mask"] for r in mine},
                        **{f"kp_{r['sid']}": store[r["sid"]]["kp"] for r in mine})
    return res


# ----------------------------------------------------------------------------- stage 2: SNN
def load_feats(person, out):
    z = np.load(out / f"within_feats_{person}.npz", allow_pickle=False)
    recs = []
    for j, name in enumerate(z["names"]):
        name = str(name)
        recs.append(dict(sid=name, act=str(z["acts"][j]), cut=int(z["cut"][j]), start_test=int(z["start_test"][j]),
                         mask=z[f"mask_{name}"], kp=z[f"kp_{name}"]))
    return recs


def windows_of(a, T, hop, velocity):
    if velocity:
        a = np.concatenate([a, np.diff(a, axis=0, prepend=a[:1])], 1)
    return [a[s:s + T] for s in range(0, len(a) - T + 1, hop)]


def make_sets(recs, feat, args):
    Xtr, ytr, Xte, yte, rte = [], [], [], [], []
    for r in recs:
        parts = []
        if feat in ("mask", "both"):
            parts.append(r["mask"])
        if feat in ("kp", "both"):
            parts.append(r["kp"])
        a = np.concatenate(parts, 1)
        used = np.r_[0:r["cut"], r["start_test"]:len(a)]
        a = a - a[used].mean(0, keepdims=True)                     # per-recording centring, no labels used
        y = ACTS.index(r["act"])
        for w in windows_of(a[: r["cut"]], args.window, args.train_hop, True):
            Xtr.append(w)
            ytr.append(y)
        for w in windows_of(a[r["start_test"]:], args.window, args.test_hop, True):
            Xte.append(w)
            yte.append(y)
            rte.append(r["sid"])
    return (np.array(Xtr, np.float32), np.array(ytr), np.array(Xte, np.float32), np.array(yte), np.array(rte))


def run_snn(persons, args, out, dev):
    print("\n" + "=" * 100)
    print("SNN on the CSI-predicted poses - SAME PERSON: train on the first 80% of the person, test on the last 20%")
    print("=" * 100)
    summary = {}
    for feat in args.feats:
        accs_w, accs_r = [], []
        cm = np.zeros((3, 3))
        per_person = {p: [] for p in persons}
        for seed in range(args.seeds):
            snn_args = SimpleNamespace(seed=seed, epochs=args.snn_epochs, lr=args.snn_lr, batch=args.snn_batch)
            Y, P, R = [], [], []
            for p in persons:
                recs = load_feats(p, out)
                Xtr, ytr, Xte, yte, rte = make_sets(recs, feat, args)
                if len(Xte) == 0 or len(Xtr) == 0:
                    continue
                pred, tr_acc, _ = ps.train_eval(Xtr, ytr, Xte, snn_args, dev)
                per_person[p].append(float(np.mean(pred == yte)))
                Y.append(yte)
                P.append(pred)
                R.append(rte)
            Y, P, R = np.concatenate(Y), np.concatenate(P), np.concatenate(R)
            ok = tot = 0
            for r in sorted(set(R)):
                s = R == r
                ok += np.bincount(P[s], minlength=3).argmax() == Y[s][0]
                tot += 1
            accs_w.append(float(np.mean(P == Y)))
            accs_r.append(ok / tot)
            for i in range(3):
                for j in range(3):
                    cm[i, j] += np.sum((Y == i) & (P == j))
        print(f"\nfeatures = {feat} + velocity   ({len(Y)} test windows, {tot} recordings, {args.seeds} seed(s))")
        for p in persons:
            if per_person[p]:
                print(f"   test {p:10s} window acc {np.mean(per_person[p]) * 100:5.1f}%")
        print(f"   pooled window acc    {np.mean(accs_w) * 100:5.1f}% +- {np.std(accs_w) * 100:.1f}   (chance 33.3%)")
        print(f"   pooled recording acc {np.mean(accs_r) * 100:5.1f}% +- {np.std(accs_r) * 100:.1f}   ({tot} recordings, chance 33.3%)")
        print("   confusion matrix, row % (true -> predicted)")
        print("               " + " ".join(f"{a:>7s}" for a in ACTS))
        for i, a in enumerate(ACTS):
            print(f"   true {a:5s}   " + " ".join(f"{cm[i, j] / max(cm[i].sum(), 1) * 100:7.1f}" for j in range(3)))
        summary[feat] = dict(window_acc=float(np.mean(accs_w)), window_std=float(np.std(accs_w)),
                             recording_acc=float(np.mean(accs_r)), recording_std=float(np.std(accs_r)))
    (out / "within_snn.json").write_text(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz")
    ap.add_argument("--out", default="~/percom/siwis_results_within_full")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase", "aoa", "vel"], choices=list(v2.GROUPS))
    ap.add_argument("--persons", nargs="*", default=[], help="only these persons (default: all)")
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose_weight", type=float, default=500.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--oof_folds", type=int, default=4)
    ap.add_argument("--gap", type=int, default=3, help="frames dropped between train and test blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_siwis", action="store_true", help="reuse the saved poses, run only the SNN")
    ap.add_argument("--skip_snn", action="store_true")
    ap.add_argument("--feats", nargs="+", default=["both", "kp", "mask"], choices=["mask", "kp", "both"])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--train_hop", type=int, default=2)
    ap.add_argument("--test_hop", type=int, default=5)
    ap.add_argument("--snn_epochs", type=int, default=40)
    ap.add_argument("--snn_lr", type=float, default=2e-3)
    ap.add_argument("--snn_batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3, help="SNN seeds (mean +- std reported)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cols = sorted({c for g in args.channels for c in v2.GROUPS[g]})
    print(f"device {dev} | input channels {[v2.CHANNELS[c] for c in cols]} | epochs {args.epochs} | oof folds {args.oof_folds}")

    if not args.skip_siwis:
        recs = v2.load_all_v2(args, cols)
        persons = sorted({r["person"] for r in recs})
        if args.persons:
            persons = [p for p in persons if p.lower() in {q.lower() for q in args.persons}]
        print(f"{len(recs)} recordings | persons {persons}")
        results = []
        for p in persons:
            print(f"\n######## {p}: first 80% of his/her recordings train, last 20% test")
            results.append(siwis_for_person(p, recs, args, dev, len(cols), out))
        if len(results) > 1:
            print("\n=== SiWiS within-person (80/20) mean over persons ===")
            keys = results[0]["model"].keys()
            for name in ("model", "baseline"):
                print(f"  {name:9s}:", tp.fmt({k: float(np.nanmean([r[name][k] for r in results])) for k in keys}))
        pc.save_json(out / "within_siwis.json", results)
    else:
        persons = sorted(p.stem.replace("within_feats_", "") for p in out.glob("within_feats_*.npz"))
        if args.persons:
            persons = [p for p in persons if p.lower() in {q.lower() for q in args.persons}]
    if not args.skip_snn and persons:
        run_snn(persons, args, out, dev)
    print(f"\nSaved in {out}")


if __name__ == "__main__":
    main()
