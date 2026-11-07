#!/usr/bin/env python3
"""
train_percom_v2.py - the SiWiS pose model trained again, with the two problems of the first run fixed.
This is the "final cross-person model (with speed)" of the report. The instructor's repository code is
used unchanged.

What changed compared with train_percom.py
  1. Loss balance. The repository's mask loss adds up the error over every pixel (values in the thousands),
     while the keypoint loss is a mean (around 0.002), so the joints hardly influenced the training. Here the
     mask loss is divided by the number of pixels and the keypoint loss is multiplied by --pose_weight
     (default 500), so both parts matter.
  2. Richer input. Ten channels per time step instead of two (see CHANNELS below): the spread and the log level
     of each antenna, the phase difference between the antennas, an angle-of-arrival estimate with its speed
     and spread, and the speed of the signal level. Every channel is z-scored inside its own recording, which
     uses no labels and removes offsets between people and sessions. --channels picks a subset, for example
     --channels std gives the old two-channel input.
  3. Optimiser. AdamW with lr 3e-4 and cosine decay, gradient clipping at 1.0 (the first run had one
     exploding epoch), 80 epochs and parallel data loading.

It reads the labels that prepare_percom.py already wrote (nothing is labelled again) and the converted CSI (npz).
The output files have the same names as those of train_percom.py (pred_person_<Person>.npz and so on), so
pose_snn.py can read them:

    python train_percom_v2.py --split_test Vansh --epochs 3    # quick test with a single held-out person
    python train_percom_v2.py --out ~/percom/siwis_results_final   # all 4 folds (leave-one-person-out)
    python pose_snn.py --results ~/percom/siwis_results_final

Run it from the SiWiS repository folder (it needs config.py, models/ and losses/); scripts/setup_siwis.sh
prepares that folder.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
import percom_common as pc
import train_percom as tp
from data.dataset import SiWiSDataset
from losses.losses import mask_loss, pose_loss
from models.signal_encoder import SignalEncoder
from models.siwis_net import SiWiSNet

CHANNELS = ["std_a", "std_b", "logamp_a", "logamp_b", "phase_diff", "aoa", "aoa_vel", "aoa_std", "ampvel_a", "ampvel_b"]
GROUPS = {"std": [0, 1], "amp": [2, 3], "phase": [4], "aoa": [5, 6, 7], "vel": [8, 9]}
# columns 0-4 are the same as in our first run of this script, so --channels std amp phase reproduces it


# ----------------------------------------------------------------------------- features
def ma(x, n):
    """Centred moving average (edge padded), same length as x."""
    n = max(int(n), 1)
    if n == 1:
        return x.copy()
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")


def stream_features(npz_path, rate, spacing=0.5, ref_until=None):
    """-> grid (absolute epoch s), (K, 10) float32, each column z-scored over the recording.
    0-1 std of each antenna, 2-3 log level, 4 antenna phase difference (unwrapped, 2 s high-pass),
    5 AoA = arcsin(phase difference / (2 pi d/lambda)) with the per-recording offset removed (d/lambda = spacing, 0.5 = half wave),
    6 AoA velocity (rad/s), 7 AoA spread over 0.5 s (how much the direction wobbles),
    8-9 velocity of the log level of each antenna (dB-like change per second).
    ref_until (epoch seconds, optional): the AoA offset and the z-score statistics are then computed only from
    CSI recorded before this time. segment_pipeline.py uses it so that no statistic of the test part leaks
    into the input."""
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
    ph = np.angle((A * np.conj(B)).sum(1))[ok]                        # wrapped phase difference per packet
    tt = (t0 + t)[ok]
    ref = np.ones(len(tt), bool) if ref_until is None else (tt < ref_until)
    if ref.sum() < 50:
        ref[:] = True
    pd = np.unwrap(ph)                                                # unwrap on the packet series, then resample
    off = np.angle(np.exp(1j * ph[ref]).mean())                       # constant antenna/cable offset of this recording
    wr = np.angle(np.exp(1j * (ph - off)))
    aoa = np.arcsin(np.clip(wr / (2 * np.pi * spacing), -1.0, 1.0))
    grid = np.arange(tt[0], tt[-1], 1.0 / rate)
    cols = [stdA[ok], stdB[ok], np.log(mA[ok]), np.log(mB[ok]), pd, aoa]
    F = np.stack([np.interp(grid, tt, c) for c in cols], 1)
    F[:, 4] = F[:, 4] - ma(F[:, 4], 2 * rate)                         # remove the slow phase drift (2 s moving average)
    n2, n5 = 0.2 * rate, 0.5 * rate
    aoa_s = ma(F[:, 5], n2)
    aoa_vel = ma(np.gradient(aoa_s) * rate, n2)
    aoa_std = np.sqrt(np.clip(ma(F[:, 5] ** 2, n5) - ma(F[:, 5], n5) ** 2, 0, None))
    va = ma(np.gradient(ma(F[:, 2], n2)) * rate, n2)
    vb = ma(np.gradient(ma(F[:, 3], n2)) * rate, n2)
    F = np.column_stack([F[:, :5], aoa_s, aoa_vel, aoa_std, va, vb])
    gm = np.ones(len(F), bool) if ref_until is None else (grid < ref_until)
    if gm.sum() < 50:
        gm[:] = True
    F = (F - F[gm].mean(0)) / (F[gm].std(0) + 1e-9)
    return grid, F.astype(np.float32)


def get_features(rec_dir, meta, npz_dir, rate, spacing=0.5, ref_until=None):
    if ref_until is not None:                                          # split-specific cache (segment_pipeline.py)
        key = json.dumps([round(float(ref_until), 3), rate, spacing])
        fk, fg, ff = rec_dir / "features_seg_key.json", rec_dir / "features_seg_t.npy", rec_dir / "features_seg.npy"
        if fk.exists() and fg.exists() and ff.exists() and fk.read_text() == key:
            return np.load(fg), np.load(ff)
        grid, F = stream_features(Path(npz_dir).expanduser() / meta["source_npz"], rate, spacing, ref_until)
        np.save(fg, grid)
        np.save(ff, F)
        fk.write_text(key)
        return grid, F
    fg, ff = rec_dir / "features_v3_t.npy", rec_dir / "features_v3.npy"
    if fg.exists() and ff.exists():
        return np.load(fg), np.load(ff)
    grid, F = stream_features(Path(npz_dir).expanduser() / meta["source_npz"], rate, spacing)
    np.save(fg, grid)
    np.save(ff, F)
    return grid, F


def load_recording_v2(rec_dir, meta, args, cols, ref_until=None):
    grid, F = get_features(rec_dir, meta, args.npz_dir, args.rate, args.spacing, ref_until)
    feats = np.column_stack([grid, F[:, cols].astype(np.float64)])
    ts = np.load(rec_dir / "frame_timestamps.npy")
    mask = np.load(rec_dir / "mask_labels.npy")
    kp_xy, kp_vis = np.load(rec_dir / "kp_xy.npy"), np.load(rec_dir / "kp_vis.npy")
    coords = np.zeros(kp_xy.shape[:3] + (2,), np.int64)
    coords[..., 0] = np.clip(kp_xy[..., 1].astype(np.int64), 0, config.HEATMAP_H - 1)
    coords[..., 1] = np.clip(kp_xy[..., 0].astype(np.int64), 0, config.HEATMAP_W - 1)
    return SiWiSDataset(feats, ts, mask, pc.LazyHeatmaps(kp_xy, kp_vis), coords, kp_vis)


def load_all_v2(args, cols):
    recs = []
    for m in sorted((Path(args.data).expanduser() / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta["person"].lower() in {e.lower() for e in args.exclude}:
            continue
        print(f"  features {m.parent.name}", flush=True)
        meta["ds"] = load_recording_v2(m.parent, meta, args, cols)
        recs.append(meta)
    if not recs:
        raise SystemExit("no prepared recordings found, run prepare_percom.py first")
    return recs


# ----------------------------------------------------------------------------- training
def within_split(recs, person, frac=0.8, gap=3):
    """One participant only: the first 80 % of each of their recordings is used for training and the last 20 %
    for testing. A few frames are dropped between the two so that no 1.7 s window overlaps both parts.
    Nobody else's data is used."""
    tr, te = [], []
    for r in recs:
        if r["person"] != person:
            continue
        n = len(r["ds"])
        cut = int(n * frac)
        tr.append((r, r["ds"], list(range(cut))))
        te.append((r, r["ds"], list(range(min(cut + gap, n), n))))
    return tr, te


def run_fold(recs, args, test_person, dev, tag, n_ch):
    if args.within:
        tr, te = within_split(recs, test_person)
    else:
        tr, te = tp.make_split(recs, "person", test_person)
    n_tr, n_te = sum(len(i) for _, _, i in tr), sum(len(i) for _, _, i in te)
    print(f"\n=== {tag}: test person {test_person} | train {n_tr} frames ({len(tr)} recordings) / test {n_te} frames ===")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SiWiSNet()
    model.signal_encoder = SignalEncoder(in_channels=n_ch)            # the repo default is 2 channels
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = DataLoader(ConcatDataset([Subset(d, idx) for _, d, idx in tr]), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    npix = config.HEATMAP_H * config.HEATMAP_W
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = dict(mask=0.0, pose=0.0, n=0)
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
            run["mask"] += float(lm.detach())
            run["pose"] += float(lp.detach())
            run["n"] += 1
        sched.step()
        if ep == 0 or (ep + 1) % args.print_every == 0 or ep == args.epochs - 1:
            nb = max(run["n"], 1)
            print(f"  epoch {ep + 1:3d}  mask_loss(mean) {run['mask'] / nb:.4f}  pose_loss {run['pose'] / nb:.5f}"
                  f"   ({time.time() - t0:.0f} s)", flush=True)
    pred = tp.predict(model, te, args.batch_size, dev)
    pts = pc.argmax_points(pred["pp"])
    res = dict(model=tp.score(pred["pm"], pred["gm"], pts, pred["gc"], pred["gv"]), baseline=tp.baseline(tr, pred),
               by_activity={}, test_person=test_person, n_train=n_tr, n_test=n_te)
    print("  MODEL     :", tp.fmt(res["model"]))
    print("  MEAN-POSE :", tp.fmt(res["baseline"]), " <- predicts the average mask / skeleton, no CSI used")
    for a in sorted(set(pred["act"])):
        s = pred["act"] == a
        res["by_activity"][a] = tp.score(pred["pm"][s], pred["gm"][s], pts[s], pred["gc"][s], pred["gv"][s])
        print(f"    {a:4s}    :", tp.fmt(res["by_activity"][a]))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    kind = "within" if args.within else "person"
    torch.save(model.state_dict(), out / f"siwisv2_{kind}_{test_person}.pt")
    np.savez_compressed(out / f"pred_{kind}_{test_person}.npz", masks=pred["pm"].astype(np.float16),
                        heatmaps=pred["pp"].astype(np.float16), gt_masks=pred["gm"], activity=pred["act"],
                        person=pred["person"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz")
    ap.add_argument("--out", default="~/percom/siwis_results_v2")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase", "aoa", "vel"], choices=list(GROUPS),
                    help="input groups: std amp phase aoa vel (default: all five groups = the 10 channels of the report)")
    ap.add_argument("--split_test", default="", help="run only this held-out person (default: all, leave-one-person-out)")
    ap.add_argument("--within", action="store_true",
                    help="within-person mode: for each person, train on the first 80%% of their recordings and test on the last 20%% "
                         "(easier than leave-one-person-out, report it as such). within_pipeline.py is the full version of this")
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose_weight", type=float, default=500.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--spacing", type=float, default=0.5, help="antenna spacing in wavelengths for the AoA (0.5 = half wavelength)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)
    cols = sorted({c for g in args.channels for c in GROUPS[g]})
    print(f"device {dev} | input channels {[CHANNELS[c] for c in cols]} | pose_weight {args.pose_weight} | lr {args.lr}")
    recs = load_all_v2(args, cols)
    persons = sorted({r["person"] for r in recs})
    print(f"{len(recs)} recordings | persons {persons}")
    todo = [args.split_test] if args.split_test else persons
    results = [run_fold(recs, args, p, dev, f"fold {i}/{len(todo)}", len(cols)) for i, p in enumerate(todo, 1)]
    if len(results) > 1:
        print("\n=== " + ("within-person (80/20) mean" if args.within else "leave-one-person-out mean") + " ===")
        keys = results[0]["model"].keys()
        for name in ("model", "baseline"):
            print(f"  {name:9s}:", tp.fmt({k: float(np.nanmean([r[name][k] for r in results])) for k in keys}))
    pc.save_json(Path(args.out).expanduser() / ("results_within.json" if args.within else "results_v2.json"), results)
    print(f"\nSaved in {Path(args.out).expanduser()}")


if __name__ == "__main__":
    main()
