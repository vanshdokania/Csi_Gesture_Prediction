#!/usr/bin/env python3
"""
find_segments.py - our first gesture cutter (superseded by siwis_pose/video_segments.py, kept for reference).
It cuts every recording into complete repetitions ("segments"): hand down -> up -> held -> down.

    python find_segments.py --data ~/percom/siwis_data

How: the video poses made by prepare_percom.py tell, for every labelled frame, whether a wrist is above its elbow
(same rule as label_check.py / hand_state.py). Runs of "hand up" frames are the repetitions. A segment goes from the
middle of the rest before a repetition to the middle of the rest after it, so the segments tile the recording and
every segment holds exactly one complete repetition. A repetition that touches the start or the end of the recording
is cut off, so it is dropped.

The video is only used to decide where to cut. Nothing here touches the Wi-Fi input of the later stages.

Why we moved on: it works on the 2-frames-per-second labels with a hard yes/no rule, which glues fast
repetitions together and splits slow ones. video_segments.py reads the video at 10 fps instead.

Output (in --data):
    segments.json            the cuts (frame indices). You may edit this file by hand and re-run the later steps.
    segments_plots/<id>.png  one picture per recording: the wrist-up signal with the cuts (worth a look)
    segments_summary.txt     the table printed below
If a recording gives fewer than --min_segments segments (e.g. the hand stays up), it falls back to fixed-length pieces
(--fallback_len seconds) and says so; or force fixed pieces for everything with --fixed SECONDS.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10          # COCO ids (left = the person's own left)


def up_signals(rec, margin):
    xy = np.load(rec / "kp_xy.npy")[:, 0]                  # (n, 17, 2) x, y in the 48x64 label grid
    vis = np.load(rec / "kp_vis.npy")[:, 0].astype(bool)
    lw = vis[:, L_WRIST] & vis[:, L_ELBOW] & (xy[:, L_WRIST, 1] < xy[:, L_ELBOW, 1] - margin)
    rw = vis[:, R_WRIST] & vis[:, R_ELBOW] & (xy[:, R_WRIST, 1] < xy[:, R_ELBOW, 1] - margin)
    return lw, rw


def runs_of(b):
    """[(start, end_exclusive)] of the True runs of a boolean vector."""
    d = np.diff(np.r_[0, b.astype(int), 0])
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def clean_up(up, min_up, max_gap):
    up = median_filter(up.astype(int), size=3, mode="nearest").astype(bool)
    for s, e in runs_of(~up):                              # close short gaps inside a repetition
        if s > 0 and e < len(up) and e - s <= max_gap:
            up[s:e] = True
    for s, e in runs_of(up):                               # drop flicker
        if e - s < min_up:
            up[s:e] = False
    return up


def tile_by_runs(up_runs, n):
    """segments = from the middle of one rest to the middle of the next; one run each."""
    if len(up_runs) < 2:
        return [[0, n]] if up_runs else []
    cuts = [0] + [(up_runs[i][1] + up_runs[i + 1][0]) // 2 for i in range(len(up_runs) - 1)] + [n]
    return [[int(cuts[i]), int(cuts[i + 1])] for i in range(len(cuts) - 1)]


def tile_fixed(n, length):
    k = max(n // length, 1)
    cuts = [round(i * n / k) for i in range(k + 1)]
    return [[int(cuts[i]), int(cuts[i + 1])] for i in range(k)]


def plot(path, sid, lw, rw, up, segs, fps, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(up)) / fps
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.step(t, lw * 0.9 + 1.2, where="post", color="tab:blue", label="left wrist above elbow")
    ax.step(t, rw * 0.9 + 0.1, where="post", color="tab:orange", label="right wrist above elbow")
    ax.fill_between(t, 0, 2.2, where=up, step="post", color="green", alpha=0.12, label="repetition (hand up)")
    for i, (s, e) in enumerate(segs):
        ax.axvline(s / fps, color="k", lw=1.2)
        ax.text((s + e) / 2 / fps, 2.3, str(i + 1), ha="center", fontsize=8)
    ax.axvline(len(up) / fps, color="k", lw=1.2)
    ax.set_ylim(0, 2.6)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{sid}: {len(segs)} segments  {note}", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--margin", type=float, default=1.0, help="wrist must be this many label-grid pixels above the elbow")
    ap.add_argument("--min_up", type=float, default=1.5, help="shortest hand-up period that counts (s)")
    ap.add_argument("--max_gap", type=float, default=1.0, help="a brief drop shorter than this inside a repetition is ignored (s)")
    ap.add_argument("--min_segments", type=int, default=6, help="fewer than this -> fall back to fixed pieces")
    ap.add_argument("--fallback_len", type=float, default=10.0)
    ap.add_argument("--fixed", type=float, default=0.0, help="force fixed-length segments of this many seconds everywhere")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    recs = sorted((data / "rec").glob("*/meta.json"))
    if not recs:
        raise SystemExit(f"no prepared recordings in {data}/rec - run prepare_percom.py first")
    (data / "segments_plots").mkdir(exist_ok=True)
    out, lines = {}, []
    hdr = f"{'recording':22s} {'person':10s} {'act':4s} {'frames':>6s} {'segs':>4s} {'len s (min/med/max)':>22s}  method"
    lines.append(hdr)
    for m in recs:
        d = m.parent
        meta = json.loads(m.read_text())
        fps = float(meta.get("label_fps", 2.0))
        lw, rw = up_signals(d, args.margin)
        n = len(lw)
        up = clean_up(lw | rw, max(int(round(args.min_up * fps)), 1), int(round(args.max_gap * fps)))
        runs = [(s, e) for s, e in runs_of(up) if s > 0 and e < n]      # drop cut-off repetitions
        method = "video"
        if args.fixed > 0:
            segs, method = tile_fixed(n, max(int(round(args.fixed * fps)), 2)), f"fixed {args.fixed:g} s"
        else:
            segs = tile_by_runs(runs, n)
            if len(segs) < args.min_segments:
                segs = tile_fixed(n, max(int(round(args.fallback_len * fps)), 2))
                method = f"FALLBACK fixed {args.fallback_len:g} s (only {len(runs)} repetitions found - check the plot)"
        out[meta["sid"]] = dict(person=meta["person"], act=meta["activity"], n_frames=n, fps=fps, method=method,
                                segments=segs, up_runs=[[int(s), int(e)] for s, e in runs])
        ln = np.array([(e - s) / fps for s, e in segs])
        lines.append(f"{meta['sid']:22s} {meta['person']:10s} {meta['activity']:4s} {n:6d} {len(segs):4d} "
                     f"{ln.min():7.1f}/{np.median(ln):5.1f}/{ln.max():5.1f}  {method}")
        plot(data / "segments_plots" / f"{meta['sid']}.png", meta["sid"], lw, rw, up, segs, fps, f"[{method}]")
    lines.append("")
    lines.append("per person: segments in total, and how many go to test with an 80/20 split (last 20% of every recording)")
    for p in sorted({v["person"] for v in out.values()}):
        tot = [len(v["segments"]) for v in out.values() if v["person"] == p]
        te = [max(1, int(round(0.2 * k))) for k in tot]
        lines.append(f"  {p:10s} {sum(tot):4d} segments in {len(tot)} recordings -> about {sum(te)} test segments")
    warn = [k for k, v in out.items() if v["method"].startswith("FALLBACK") or len(v["segments"]) < 3]
    if warn:
        lines.append("\nLOOK AT THESE PLOTS FIRST (fallback or fewer than 3 segments): " + ", ".join(warn))
    text = "\n".join(lines)
    print(text)
    (data / "segments_summary.txt").write_text(text + "\n")
    (data / "segments.json").write_text(json.dumps(out, indent=1))
    print(f"\nSaved {data / 'segments.json'} and plots in {data / 'segments_plots'}")


if __name__ == "__main__":
    main()
