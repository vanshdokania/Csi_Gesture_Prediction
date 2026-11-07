#!/usr/bin/env python3
"""
label_frames.py - an optional, earlier way of labelling: it cuts one Doppler spectrogram into frames and
labels every frame as active, still or transition by looking at the motion curve.

The final experiments do not need it. make_dataset.py gives every frame the gesture of its recording,
because each recording contains one gesture type only. We kept this script because it is handy to
see where the person actually moves and where they stand still (see the *_segments.png picture).

    python3 label_frames.py --pre output --out labels
    python3 label_frames.py --pre output --name LHU_S1_Priyanshu --out labels

Reads   <pre>/doppler/<name>.npz     (made by preprocess_csi.py)
Writes  <out>/<name>_frames.csv       one row per frame: times, label, how sure, features
        <out>/<name>_frames.npz       X (frames, time, doppler_bins) in dB + y + label names
        <out>/<name>_segments.png     motion curve with the detected active/still segments (worth a look)
        <out>/dataset.npz / dataset.csv   all files together (what you feed the SNN)

How a frame gets its label (nothing is trained; it relies on the recording protocol
"do the activity, stand still, repeat"):
  1. motion curve (energy above 1.5 Hz, already in the Doppler file) is smoothed;
  2. a threshold (automatic Otsu on a log scale - splits 'low' from 'high' level - or --thr) separates ACTIVE from STILL;
  3. hysteresis + minimum length remove flicker (--min-active, --min-still seconds);
  4. a frame is labelled with the activity name (from the file, e.g. walking) when >= 50 % of it is active,
     'still' when <= 20 % is active, and 'transition' in between (ignored for training by default).
Labels are only as good as the threshold: look at *_segments.png and fix with --thr / --min-active,
or hand-correct the CSV. Optional --video-offset-from NPZDIR adds a 'video_t' column for the side-by-side video.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def otsu(x, bins=64):
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return c[int(np.argmax(var))]


def smooth(x, n):
    n = max(1, int(n))
    return np.convolve(x, np.ones(n) / n, mode="same")


def segments_from_mask(mask, dt, min_active, min_still):
    """Boolean per-window mask -> cleaned mask (remove short still gaps, then short active blips)."""
    m = mask.copy()

    def runs(v):
        edges = np.flatnonzero(np.diff(np.r_[False, v, False].astype(int)))
        return list(zip(edges[::2], edges[1::2]))

    for s, e in runs(~m):                               # short gaps inside activity -> active
        if (e - s) * dt < min_still and s > 0 and e < len(m):
            m[s:e] = True
    for s, e in runs(m):                                # short active blips -> still
        if (e - s) * dt < min_active:
            m[s:e] = False
    return m


def load_offset(npz_dir, name, video_start):
    """Seconds to add to CSI time to get video time, or None."""
    if not npz_dir or video_start is None:
        return None
    p = Path(npz_dir).expanduser() / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "t0_epoch" in z.files and float(z["t0_epoch"]) > 0:
        return float(z["t0_epoch"]) - video_start
    return None


def process(path, args, out):
    name = path.stem
    D = np.load(path, allow_pickle=False)
    tw = D["t_windows"].astype(float)
    motion = D["motion"].astype(float)
    db = D["doppler_db"].astype(np.float32)
    raw = D["raw_power"].astype(np.float64)
    freqs = D["freqs_hz"].astype(float)
    activity = str(D["activity"]) if "activity" in D.files else "activity"
    activity = args.activity or activity
    dt = float(np.median(np.diff(tw)))
    fs_w = 1.0 / dt

    # ---- 1-3: active / still per window
    ms = smooth(motion, args.smooth * fs_w)
    lo, hi = np.percentile(ms, 5), np.percentile(ms, 95)
    if args.thr is not None:
        thr = args.thr
    else:                                   # Otsu on a log scale: weak activity (clapping) still separates from noise floor
        lg = np.log10(ms + 1e-3)
        thr = float(10 ** otsu(lg) - 1e-3)
    bimodal = (hi - lo) > 0.15
    if not bimodal:
        print(f"  WARNING: motion barely varies (5-95% range {hi - lo:.2f}); the file may be all-still or "
              f"all-active. Check the PNG and set --thr by hand.")
    mask = segments_from_mask(ms > thr, dt, args.min_active, args.min_still)

    # ---- 4: frames
    n_t = int(round(args.frame * fs_w))
    hop = int(round(args.hop * fs_w))
    starts = np.arange(0, len(tw) - n_t + 1, hop)
    if len(starts) == 0:
        raise RuntimeError("recording shorter than one frame")
    absmax = max(np.abs(freqs).max(), 1e-9)
    hi_f = np.abs(freqs) > 1.5
    rows, X, y = [], [], []
    for k, s in enumerate(starts):
        sl = slice(s, s + n_t)
        frac = float(mask[sl].mean())
        if frac >= args.active_frac:
            lab = activity
        elif frac <= 1 - args.active_frac or frac <= 0.2:
            lab = "still"
        else:
            lab = "transition"
        p = raw[sl].sum(axis=0)
        p = p / (p.sum() + 1e-30)
        centroid = float((np.abs(freqs) * p).sum())
        spread = float(np.sqrt((((np.abs(freqs) - centroid) ** 2) * p).sum()))
        rows.append(dict(frame=k, t_start=round(tw[s], 3), t_end=round(tw[s + n_t - 1], 3),
                         label=lab, active_frac=round(frac, 3),
                         motion_mean=round(float(motion[sl].mean()), 4), motion_max=round(float(motion[sl].max()), 4),
                         doppler_centroid_hz=round(centroid, 3), doppler_spread_hz=round(spread, 3),
                         high_freq_ratio=round(float(p[hi_f].sum()), 4)))
        X.append(db[sl])
        y.append(lab)
    X = np.stack(X).astype(np.float16)

    vid_off = load_offset(args.npz_dir, name, args.video_start)
    if vid_off is not None:
        for r in rows:
            r["video_t_start"] = round(r["t_start"] + vid_off, 3)
            r["video_t_end"] = round(r["t_end"] + vid_off, 3)

    names = sorted(set(y))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / f"{name}_frames.npz", X=X, y=np.array(y), t_start=np.array([r["t_start"] for r in rows]),
                        freqs_hz=freqs, source=name, activity=activity,
                        frame_s=args.frame, hop_s=args.hop, windows_per_s=fs_w)

    # ---- picture to check the segmentation
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 2]})
    ax[0].imshow(db.T, aspect="auto", origin="lower", cmap="magma", vmin=-35, vmax=0,
                 extent=[tw[0], tw[-1], freqs[0], freqs[-1]])
    ax[0].set_ylim(-30, 30)
    ax[0].set_ylabel("Doppler (Hz)")
    ax[0].set_title(f"{name}   activity = {activity}", fontsize=10, loc="left")
    ax[1].plot(tw, motion, color="tab:red", lw=0.8, alpha=0.5, label="motion")
    ax[1].plot(tw, ms, color="k", lw=1.2, label="smoothed")
    ax[1].axhline(thr, color="tab:blue", ls="--", label=f"threshold {thr:.2f}")
    colors = {"still": "#d0d0d0", "transition": "#ffd27f"}
    for r in rows:
        c = colors.get(r["label"], "#7fc97f")
        ax[1].axvspan(r["t_start"], r["t_start"] + args.hop, ymin=0, ymax=0.06, color=c, lw=0)
    ax[1].fill_between(tw, 0, 1.05, where=mask, color="tab:green", alpha=0.15, label=f"ACTIVE = {activity}")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("Time (s) from first packet")
    ax[1].set_ylabel("Motion")
    ax[1].legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out / f"{name}_segments.png", dpi=110)
    plt.close(fig)

    cnt = {n: y.count(n) for n in names}
    print(f"  {len(rows)} frames of {args.frame:.1f} s (hop {args.hop:.1f} s)  threshold {thr:.2f}  -> {cnt}")
    if vid_off is not None:
        print(f"  video_t columns added (CSI + {vid_off:+.2f} s)")
    return X, np.array(y), rows, name


def main():
    ap = argparse.ArgumentParser(description="Cut Doppler spectrograms into frames and label them still/activity")
    ap.add_argument("--pre", default="output", help="folder given as --output to preprocess_csi.py (default ./output)")
    ap.add_argument("--out", default="labels", help="output folder (default ./labels)")
    ap.add_argument("--name", default="", help="only files whose name contains this text (e.g. LHU_S1_Priyanshu)")
    ap.add_argument("--activity", default="", help="override the activity name stored in the file")
    ap.add_argument("--frame", type=float, default=1.0, help="frame length in seconds (default 1.0)")
    ap.add_argument("--hop", type=float, default=0.5, help="frame step in seconds (default 0.5)")
    ap.add_argument("--thr", type=float, default=None, help="motion threshold 0..1 (default: automatic Otsu)")
    ap.add_argument("--smooth", type=float, default=0.5, help="motion smoothing in seconds (default 0.5)")
    ap.add_argument("--min-active", type=float, default=1.0, help="drop active bursts shorter than this, s (default 1.0)")
    ap.add_argument("--min-still", type=float, default=1.0, help="fill still gaps shorter than this, s (default 1.0)")
    ap.add_argument("--active-frac", type=float, default=0.5, help="share of a frame that must be active (default 0.5)")
    ap.add_argument("--npz-dir", default="", help="csi_to_npz.py folder (to add video_t columns)")
    ap.add_argument("--video-start", type=float, default=None, help="video_start_epoch from timing.txt (adds video_t)")
    ap.add_argument("--keep-transition", action="store_true", help="keep 'transition' frames in dataset.npz")
    args = ap.parse_args()

    pre = Path(args.pre).expanduser()
    files = sorted(p for p in (pre / "doppler").glob("*.npz") if args.name in p.stem)
    if not files:
        raise SystemExit(f"Error: no Doppler files in {pre / 'doppler'} matching '{args.name}'")
    out = Path(args.out).expanduser()
    allX, ally, allrows = [], [], []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.stem}")
        try:
            X, y, rows, name = process(f, args, out)
        except Exception as e:  # noqa: BLE001
            print(f"  Error: {e}")
            continue
        keep = np.ones(len(y), bool) if args.keep_transition else (y != "transition")
        allX.append(X[keep])
        ally.append(y[keep])
        for r, k in zip(rows, keep):
            if k:
                allrows.append(dict(source=name, **r))
    if not allX:
        raise SystemExit("nothing processed")
    X = np.concatenate(allX)
    y = np.concatenate(ally)
    classes = sorted(set(y), key=lambda c: (c != "still", c))
    ids = np.array([classes.index(v) for v in y])
    np.savez_compressed(out / "dataset.npz", X=X, y=ids, labels=np.array(classes), label_text=y,
                        source=np.array([r["source"] for r in allrows]),
                        t_start=np.array([r["t_start"] for r in allrows]))
    keys = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["label_id"])
        w.writeheader()
        for r, i in zip(allrows, ids):
            w.writerow({**r, "label_id": int(i)})
    print(f"\nDataset: {X.shape} (frames, time-windows, Doppler bins)   classes {dict(zip(classes, np.bincount(ids)))}")
    print(f"Saved in {out}/ : dataset.npz, dataset.csv, per-file *_frames.csv/npz and *_segments.png (check these)")


if __name__ == "__main__":
    main()
