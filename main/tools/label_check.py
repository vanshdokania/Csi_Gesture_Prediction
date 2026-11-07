"""
label_check.py - quick sanity check of the video labels made by prepare_percom.py.

For every gesture (bhu / lhu / rhu) it prints how often each wrist is above its elbow and how often each wrist
was detected at all. In a good labelling the left-hand recordings show a high "L wrist up" value and a low
"R wrist up" value, and the other way round for the right hand.

    python tools/label_check.py ~/percom/siwis_data
"""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/percom/siwis_data").expanduser() / "rec"
LS, RS, LW, RW = 7, 8, 9, 10
rows = {}
for d in sorted(root.glob("*/kp_xy.npy")):
    d = d.parent
    act = d.name.split("_")[0]
    xy, vis = np.load(d / "kp_xy.npy")[:, 0], np.load(d / "kp_vis.npy")[:, 0].astype(bool)
    lw = vis[:, LW] & vis[:, LS] & (xy[:, LW, 1] < xy[:, LS, 1])
    rw = vis[:, RW] & vis[:, RS] & (xy[:, RW, 1] < xy[:, RS, 1])
    rows.setdefault(act, []).append((lw.mean(), rw.mean(), (lw & rw).mean(), vis[:, LW].mean(), vis[:, RW].mean()))
print("share of frames with the wrist above the elbow (COCO left = person's own left)")
print(f"{'act':5s} {'recs':>4s} {'L wrist up':>11s} {'R wrist up':>11s} {'both up':>8s} {'L wrist seen':>13s} {'R wrist seen':>13s}")
for a, v in sorted(rows.items()):
    m = np.mean(v, axis=0) * 100
    print(f"{a:5s} {len(v):4d} {m[0]:10.0f}% {m[1]:10.0f}% {m[2]:7.0f}% {m[3]:12.0f}% {m[4]:12.0f}%")
