#!/usr/bin/env python3
"""
pose_video.py - makes a video of the body pose that SiWiS predicts from the Wi-Fi signal alone (no camera in the input).

Uses the predictions saved by train_percom.py (pred_person_<Person>.npz: predicted mask + 17 keypoint heatmaps per
frame, made by a model that never saw that person).  Left panel: predicted mask with the 6 upper-body joints
(shoulders, elbows, wrists). The optional right panel (--with_gt) shows the mask from the video, only so you can
judge the quality.

    python pose_video.py --rec rhu_Vansh_S3 --out ~/percom/pose_rhu_Vansh_S3.mp4
    python pose_video.py --rec rhu_Vansh_S3 --out ... --with_gt

Needs ffmpeg on the PATH (export PATH=$HOME/bin:$PATH on the server).  Frames are the labelled frames, 2 per second.
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINTS = {5: "L shoulder", 6: "R shoulder", 7: "L elbow", 8: "R elbow", 9: "L wrist", 10: "R wrist"}
COL = {5: "#4FC3F7", 7: "#4FC3F7", 9: "#4FC3F7", 6: "#FFB74D", 8: "#FFB74D", 10: "#FFB74D"}   # blue = person's left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="recording id, e.g. rhu_Vansh_S3")
    ap.add_argument("--results", default="~/percom/siwis_results")
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--with_gt", action="store_true")
    args = ap.parse_args()
    results, data = Path(args.results).expanduser(), Path(args.data).expanduser()
    meta = json.loads((data / "rec" / args.rec / "meta.json").read_text())
    person = meta["person"]
    z = np.load(results / f"pred_person_{person}.npz", allow_pickle=False)
    # frames of this person are stored recording after recording, in sorted recording order
    offs, i = {}, 0
    for m in sorted((data / "rec").glob("*/meta.json")):
        mm = json.loads(m.read_text())
        if mm["person"] != person:
            continue
        n = len(np.load(m.parent / "frame_timestamps.npy"))
        offs[m.parent.name] = (i, n)
        i += n
    if i != len(z["masks"]):
        raise SystemExit(f"{i} frames in the recordings of {person} but {len(z['masks'])} predicted - order mismatch")
    s, n = offs[args.rec]
    masks = z["masks"][s:s + n].astype(np.float32)
    heat = z["heatmaps"][s:s + n].astype(np.float32)
    gt = z["gt_masks"][s:s + n].astype(np.float32) if args.with_gt else None
    out = Path(args.out or f"~/percom/pose_{args.rec}.mp4").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    ncol = 2 if args.with_gt else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.2), dpi=100)
    axes = np.atleast_1d(axes)
    ims, dots = [], {}
    for k, ax in enumerate(axes):
        ims.append(ax.imshow(masks[0] if k == 0 else gt[0], vmin=0, vmax=1, cmap="viridis"))
        ax.axis("off")
    axes[0].set_title("predicted from CSI only", fontsize=11)
    if args.with_gt:
        axes[1].set_title("video-derived mask (for comparison)", fontsize=11)
    for j in JOINTS:
        dots[j], = axes[0].plot([], [], "o", ms=9, mfc=COL[j], mec="white", mew=1.2)
    txt = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(args.fps),
           "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        ims[0].set_data(masks[k])
        if args.with_gt:
            ims[1].set_data(gt[k])
        for j in JOINTS:
            r, c = np.unravel_index(heat[k, j].argmax(), heat[k, j].shape)
            dots[j].set_data([c], [r])
        txt.set_text(f"{args.rec}   t = {k / args.fps:5.1f} s   (blue = person's left, orange = right)")
        fig.canvas.draw()
        p.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    p.stdin.close()
    p.wait()
    print(f"saved {out}  ({n} frames, {n / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
