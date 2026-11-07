#!/usr/bin/env python3
"""
video_segments.py - cuts every recording into single gestures, using the video.

    gpurun python video_segments.py --src <folder with BHU/ LHU/ RHU/ videos> --data ~/percom/siwis_data

Each recording contains about 48 repetitions of one gesture, and the final SNN classifies one complete
repetition at a time, so we first need to know where each repetition starts and ends.

Why a new cutter: our first version (legacy/find_segments.py) only had the labels at 2 frames per second
and a hard yes/no rule (wrist above elbow). That glued fast repetitions together or split slow ones. This
script instead

  1. runs the person-pose network (Keypoint R-CNN, the same model that made the labels) on the raw video at 10 fps,
  2. turns each frame into a smooth "how high is the hand" number: (elbow_y - wrist_y) / torso height, per hand,
     so it does not depend on how far the person stands from the camera,
  3. finds every hand-up peak on that curve (adaptive: the threshold comes from the recording's own low and high
     level, so every person and every speed works),
  4. cuts in the middle of the rest between two peaks. A segment = rest -> hand up -> held -> hand down -> rest,
     which is exactly one complete repetition. Repetitions that are cut off at the start or end of the
     recording are dropped,
  5. maps the cut times onto the existing label frames, so everything after this step stays unchanged.

Only the video decides where to cut. The Wi-Fi input of the later stages is not touched. The cut times are
transferred to the CSI through the time stamps (timing.txt of every recording).

Output (in --data):
    video_signal/<id>.npz        the hand-height curves (cached: re-cutting with other settings takes seconds,
                                 add --recompute to run the network again)
    segments_ai.json             same format as segments.json. After checking the plots:  cp segments_ai.json segments.json
    segments_ai_plots/<id>.png   one picture per recording: the curve, the peaks, the cuts
    segments_ai_summary.txt      table with the number of segments and the length of each
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HP, R_HP = 5, 6, 7, 8, 9, 10, 11, 12


# ----------------------------------------------------------------------------- 1./2. video -> curves
def hand_heights(kp, ks, kp_thresh):
    """kp (17, 2) pixels, ks (17,) scores -> (left, right) height in torso lengths; nan if not visible."""
    ok = ks > kp_thresh
    def pt(i):
        return kp[i] if ok[i] else None
    shs = [pt(L_SH), pt(R_SH)]
    hps = [pt(L_HP), pt(R_HP)]
    sh = [p for p in shs if p is not None]
    hp = [p for p in hps if p is not None]
    if sh and hp:
        torso = abs(np.mean([p[1] for p in hp]) - np.mean([p[1] for p in sh]))
    elif len(sh) == 2:
        torso = 1.6 * abs(sh[0][0] - sh[1][0])                      # no hips visible: shoulder width x 1.6
    else:
        return np.nan, np.nan
    if torso < 5:
        return np.nan, np.nan
    out = []
    for el, wr in ((L_EL, L_WR), (R_EL, R_WR)):
        out.append((kp[el][1] - kp[wr][1]) / torso if ok[el] and ok[wr] else np.nan)
    return out[0], out[1]


def run_network(video, fps, max_h, device, kp_thresh, batch=8):
    import torch
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
    import percom_common as pc
    net = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
    sig, buf = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div(255.0).to(device)
        with torch.no_grad():
            outs = net(list(x))
        for o in outs:
            best, best_area = None, 0.0
            for sc, box, kp, ks in zip(o["scores"], o["boxes"], o["keypoints"], o["keypoints_scores"]):
                if float(sc) < 0.7:
                    continue
                a = float((box[2] - box[0]) * (box[3] - box[1]))
                if a > best_area:
                    best_area, best = a, (kp[:, :2].cpu().numpy(), ks.cpu().numpy())
            sig.append(hand_heights(best[0], best[1], kp_thresh) if best else (np.nan, np.nan))
        buf.clear()

    for frame in pc.decode_frames(video, fps, out_h=max_h):
        buf.append(frame.copy())
        if len(buf) == batch:
            flush()
    flush()
    return np.array(sig, np.float32)                               # (n, 2): left, right


# ----------------------------------------------------------------------------- 3./4. curve -> segments
def fill_nan(x, max_gap):
    """linear interpolation over gaps up to max_gap samples; longer gaps stay nan."""
    x = x.copy()
    bad = np.isnan(x)
    if bad.all():
        return x
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    d = np.diff(np.r_[0, bad.astype(int), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s > max_gap:
            x[s:e] = np.nan
    return x


def segment_signal(sig, fps, smooth_s=0.2, min_period_s=1.5, prom=0.35, rest_frac=0.30, min_range=0.15):
    """
    sig (n, 2) hand height (left, right) at fps -> dict(curve, peaks, cuts, segments (time in s), low, high).
    The curve is max(left, right): either hand counts (also fixes mirrored cameras).
    """
    L = fill_nan(sig[:, 0], int(fps))
    R = fill_nan(sig[:, 1], int(fps))
    both = np.stack([L, R], 1)
    s = np.nanmax(np.where(np.isnan(both), -np.inf, both), axis=1)
    s[~np.isfinite(s)] = np.nan
    s = fill_nan(s, int(2 * fps))
    ok = ~np.isnan(s)
    if ok.sum() < 5 * fps:
        return dict(curve=s, peaks=[], cuts=[], segments=[], low=np.nan, high=np.nan, reason="person not found")
    s0 = np.where(ok, s, np.nanmedian(s))
    s0 = gaussian_filter1d(s0, max(smooth_s * fps, 0.5))
    low, high = np.percentile(s0[ok], 15), np.percentile(s0[ok], 92)
    rng = high - low
    if rng < min_range:
        return dict(curve=s0, peaks=[], cuts=[], segments=[], low=low, high=high, reason="no hand movement")
    # Cut at the valleys (the lowest points, or the middle of a flat rest). One segment = valley -> peak -> next valley.
    # This works whether the person rests between repetitions or keeps moving continuously, and it does not need
    # a height threshold for the peaks. A valley must be a clear dip (prominence) and valleys are at least
    # min_period apart.
    valleys, _ = find_peaks(-s0, prominence=prom * rng, distance=max(int(min_period_s * fps), 1))
    segs, peaks, cuts = [], [], [int(v) for v in valleys]
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = a + int(np.argmax(s0[a:b + 1]))
        if s0[k] - max(s0[a], s0[b]) >= prom * rng:                 # a real hand-up between the two valleys
            peaks.append(k)
            segs.append((a / fps, b / fps))
    return dict(curve=s0, peaks=[int(p) for p in peaks], cuts=cuts, segments=segs, low=low, high=high, reason="")


# ----------------------------------------------------------------------------- mapping + plots
def to_frames(segs_s, vt):
    """segment times (video seconds) -> index ranges [a, b) on the label frames whose video times are vt."""
    out = []
    for a, b in segs_s:
        i, j = int(np.searchsorted(vt, a, "left")), int(np.searchsorted(vt, b, "left"))
        if j - i >= 3:
            out.append([i, j])
    return out


def plot(path, sid, sig, fps, res, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(sig)) / fps
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, sig[:, 0], color="tab:blue", lw=0.6, alpha=0.5, label="left hand height")
    ax.plot(t, sig[:, 1], color="tab:orange", lw=0.6, alpha=0.5, label="right hand height")
    ax.plot(t, res["curve"], color="k", lw=1.2, label="used curve (max of both, smoothed)")
    if not np.isnan(res["low"]):
        ax.axhline(res["low"], color="gray", ls=":", lw=0.8)
        ax.axhline(res["high"], color="gray", ls=":", lw=0.8)
    for p in res["peaks"]:
        ax.plot(p / fps, res["curve"][p], "v", color="green", ms=5)
    for a, b in res["segments"]:
        ax.axvline(a, color="red", lw=1.0)
        ax.axvline(b, color="red", lw=1.0)
    ymax = ax.get_ylim()[1]
    for i, (a, b) in enumerate(res["segments"]):
        ax.text((a + b) / 2, ymax, str(i + 1), ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hand height (torso lengths above elbow)")
    ax.set_title(f"{sid}: {len(res['segments'])} segments  {note}", fontsize=10, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with BHU/ LHU/ RHU/ (the same --src as prepare_percom.py)")
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--fps", type=float, default=10.0, help="video frames per second given to the network")
    ap.add_argument("--max_h", type=int, default=480)
    ap.add_argument("--kp_thresh", type=float, default=1.0)
    ap.add_argument("--device", default="")
    ap.add_argument("--only", nargs="*", default=[], help="only these recording ids (test)")
    ap.add_argument("--recompute", action="store_true", help="run the network again even if cached")
    ap.add_argument("--smooth", type=float, default=0.2, help="smoothing of the curve (s)")
    ap.add_argument("--min_period", type=float, default=1.5, help="two repetitions are at least this far apart (s)")
    ap.add_argument("--prom", type=float, default=0.35, help="a peak must rise this fraction of the hand range")
    ap.add_argument("--rest_frac", type=float, default=0.30, help="curve below this fraction of the range = at rest")
    ap.add_argument("--expect", type=int, default=0, help="roughly how many repetitions per recording you did (flags outliers)")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    metas = sorted((data / "rec").glob("*/meta.json"))
    if not metas:
        sys.exit(f"no prepared recordings in {data}/rec")
    import percom_common as pc
    recs = {r["sid"]: r for r in pc.find_recordings(Path(args.src).expanduser(), exclude_persons=["Divij", "Nikita"])}
    (data / "video_signal").mkdir(exist_ok=True)
    (data / "segments_ai_plots").mkdir(exist_ok=True)
    old_p = data / "segments.json"
    old = json.loads(old_p.read_text()) if old_p.exists() else {}
    dev = args.device
    if not dev and any(not (data / "video_signal" / f"{m.parent.name}.npz").exists() for m in metas) or args.recompute:
        import torch
        dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"network on {dev}", flush=True)
    out, lines = {}, []
    lines.append(f"{'recording':22s} {'person':10s} {'act':4s} {'segs':>4s} {'len s (min/med/max)':>22s}  note")
    for m in metas:
        meta = json.loads(m.read_text())
        sid = meta["sid"]
        if args.only and sid not in args.only:
            continue
        d = m.parent
        ts = np.load(d / "frame_timestamps.npy")
        cache = data / "video_signal" / f"{sid}.npz"
        if cache.exists() and not args.recompute:
            z = np.load(cache)
            sig, fps = z["sig"], float(z["fps"])
        else:
            rec = recs.get(sid)
            if rec is None or not (rec["folder"] / "video.mp4").exists():
                print(f"{sid}: no video found under --src - keeping the old cuts")
                if sid in old:
                    out[sid] = old[sid]
                continue
            vstart = pc.read_timing(rec["folder"]).get("video_start_epoch")
            print(f"{sid}: running the network on the video ...", flush=True)
            sig, fps = run_network(rec["folder"] / "video.mp4", args.fps, args.max_h, dev, args.kp_thresh), args.fps
            np.savez(cache, sig=sig, fps=fps, vstart=vstart)
        z = np.load(cache)
        t_off = float(z["vstart"]) + float(meta.get("sync_shift_s", 0.0))
        res = segment_signal(sig, fps, args.smooth, args.min_period, args.prom, args.rest_frac)
        vt = ts - t_off                                             # video time of every label frame
        frames = to_frames(res["segments"], vt)
        note = "video-ai"
        if args.expect and not (0.6 * args.expect <= len(frames) <= 1.5 * args.expect):
            note += f"  (expected about {args.expect}, please check the plot)"
        if res["reason"]:
            note = f"failed ({res['reason']})"
        if len(frames) < 6 and sid in old:
            note += "  -> too few, kept the old cuts"
            out[sid] = dict(old[sid], method="old cuts (video-ai found too few)")
            frames = out[sid]["segments"]
        else:
            out[sid] = dict(person=meta["person"], act=meta["activity"], n_frames=int(len(ts)),
                            fps=float(meta.get("label_fps", 2.0)), method="video-ai", segments=frames,
                            up_runs=[[a, b] for a, b in frames])
        ln = np.array([(b - a) / out[sid]["fps"] for a, b in frames]) if frames else np.zeros(1)
        lines.append(f"{sid:22s} {meta['person']:10s} {meta['activity']:4s} {len(frames):4d} "
                     f"{ln.min():7.1f}/{np.median(ln):5.1f}/{ln.max():5.1f}  {note}")
        plot(data / "segments_ai_plots" / f"{sid}.png", sid, sig, fps, res, f"[{note}]")
        print(lines[-1], flush=True)
    lines.append("")
    lines.append("per person: segments in total, and how many go to test with an 80/20 split")
    for p in sorted({v["person"] for v in out.values()}):
        tot = [len(v["segments"]) for v in out.values() if v["person"] == p]
        te = [max(1, int(round(0.2 * k))) for k in tot]
        lines.append(f"  {p:10s} {sum(tot):4d} segments in {len(tot)} recordings -> about {sum(te)} test segments")
    text = "\n".join(lines)
    print("\n" + text)
    (data / "segments_ai_summary.txt").write_text(text + "\n")
    (data / "segments_ai.json").write_text(json.dumps(out, indent=1))
    print(f"\nSaved {data / 'segments_ai.json'} and plots in {data / 'segments_ai_plots'}")
    print("Please look at the plots. If the cuts are right, copy the file:  cp segments_ai.json segments.json   (in the data folder)")


if __name__ == "__main__":
    main()
