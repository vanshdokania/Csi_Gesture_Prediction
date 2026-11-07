#!/usr/bin/env python3
"""
csi_video_sync.py - play the preprocessed Wi-Fi CSI (double ratio + Doppler spectrogram)
next to the recorded video, time-aligned. We used it to check by eye that the motion in the
spectrogram really lines up with what the person does in the video.

    python3 csi_video_sync.py ~/percom/experiment_20261007_152659 --pre ~/percom/preprocessed

Pipeline this file belongs to:
    record.sh -> csi_to_npz.py -> preprocess_csi.py -> csi_video_sync.py   (this file)

It does no signal processing of its own. It only reads what preprocess_csi.py wrote:
    <pre>/double_ratio/<name>.npz   cleaned double-ratio CSI   -> heatmap panel
    <pre>/doppler/<name>.npz        Doppler spectrogram+motion -> spectrogram + motion panels
and the converter file <name>.npz (from csi_to_npz.py) for the absolute start time `t0_epoch`.

Time alignment (video time = seconds since the first video frame):
    video_time = csi_time + (t0_epoch - video_start_epoch) + --shift
    where video_start_epoch comes from <experiment>/timing.txt.
If t0_epoch or timing.txt is missing/implausible, the CSI END is aligned to the video end
(both stop at Ctrl+C) and a warning is printed.

Output (inside the experiment folder):
    csi_video_sync.mp4    left: video | right: heatmap, Doppler spectrogram, motion curve + moving cursor
    csi_overview.png      the right-hand panel as one still picture

The helpers `find_key` and `frame_offsets` are also imported by csi_to_npz.py, so they live here.
Needs: numpy, matplotlib, ffmpeg/ffprobe on PATH.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOISE_LEVEL = -2   # same log10 floor as preprocess_csi.py (normalised view)


# ----------------------------------------------------------------------------- helpers
def die(msg):
    sys.exit(f"Error: {msg}")


def find_key(d, names):
    """Case-insensitive lookup of the first existing, non-None key in dict d."""
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        v = low.get(n.lower())
        if v is not None:
            return v
    return None


def frame_offsets(path):
    """Byte offset of every complete frame (.csi = repeated [uint32 length][frame bytes])."""
    size = os.path.getsize(path)
    offs, pos = [], 0
    with open(path, "rb") as fh:
        while pos + 4 <= size:
            fh.seek(pos)
            h = fh.read(4)
            if len(h) < 4:
                break
            L = struct.unpack("<I", h)[0]
            if pos + 4 + L > size:      # truncated last frame (recording was interrupted)
                break
            offs.append(pos)
            pos += 4 + L
    return offs


def read_timing(folder):
    out = {}
    p = folder / "timing.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    out[k.strip()] = v.strip()
    return out


def probe_video(path, fps):
    """Return (width, height, duration_seconds)."""
    def run(args):
        r = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
                           capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {}

    j = run(["-select_streams", "v:0", "-show_entries", "stream=width,height,duration:format=duration"])
    st = (j.get("streams") or [{}])[0]
    w, h = st.get("width"), st.get("height")
    if not w or not h:
        die(f"cannot read video stream from {path} (is video.mp4 playable?)")
    for cand in ((j.get("format") or {}).get("duration"), st.get("duration")):
        try:
            d = float(cand)
            if d > 0:
                return int(w), int(h), d
        except (TypeError, ValueError):
            pass
    j = run(["-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets"])
    try:
        return int(w), int(h), int(j["streams"][0]["nb_read_packets"]) / fps
    except Exception:  # noqa: BLE001
        die("cannot determine video duration")


# ----------------------------------------------------------------------------- locating preprocessed files
def locate(folder, args):
    """Find doppler npz, double_ratio npz and (optionally) converter npz for this experiment."""
    pre = Path(args.pre).expanduser() if args.pre else folder.parent / "preprocessed"
    dop_dir, dr_dir = pre / "doppler", pre / "double_ratio"
    if not dop_dir.is_dir():
        die(f"no 'doppler' folder in {pre}\n  Give the folder you used as --output for preprocess_csi.py:"
            f"  --pre /path/to/that/folder")
    key = args.name or folder.name
    cands = sorted(p for p in dop_dir.glob("*.npz") if key in p.stem)
    if not cands:
        have = "\n    ".join(p.stem for p in sorted(dop_dir.glob("*.npz"))[:20]) or "(empty)"
        die(f"no Doppler file in {dop_dir} matching '{key}'.\n  Available:\n    {have}\n"
            f"  Use --name TEXT to pick one (any part of the file name).")
    if len(cands) > 1:
        print(f"Note: {len(cands)} files match; using {cands[0].stem}  (others: "
              f"{', '.join(c.stem for c in cands[1:])}).  Use --name to choose.")
    dop = cands[0]
    dr = dr_dir / dop.name
    if not dr.exists():
        die(f"{dr} not found (double_ratio file missing - re-run preprocess_csi.py)")

    conv = None
    search = [Path(args.npz_dir).expanduser()] if args.npz_dir else [folder.parent / "npz", folder.parent, folder]
    for d in search:
        if d.is_dir() and (d / dop.name).exists():
            conv = d / dop.name
            break
    if conv is None and not args.npz_dir:                      # last resort: look below the parent folder
        for p in folder.parent.rglob(dop.name):
            if p.parent.name not in ("doppler", "double_ratio", "png"):
                conv = p
                break
    return dop, dr, conv


def get_offset(folder, conv, args, csi_dur, vid_dur):
    """Seconds to ADD to CSI time to obtain video time. Returns (offset, description)."""
    t0, how_t0 = None, ""
    if args.t0 is not None:
        t0, how_t0 = args.t0, "--t0"
    elif conv is not None:
        try:
            z = np.load(conv, allow_pickle=False)
            if "t0_epoch" in z.files and float(z["t0_epoch"]) > 0:
                t0, how_t0 = float(z["t0_epoch"]), f"t0_epoch in {conv.name}"
        except Exception:  # noqa: BLE001
            pass
    vs = read_timing(folder).get("video_start_epoch")
    if t0 is not None and isinstance(vs, float):
        off = t0 - vs
        end_err = (csi_dur + off) - vid_dur
        if -90 < off < 90 and abs(end_err) < 20:
            return off + args.shift, (f"absolute clocks: CSI first packet is {off:+.2f} s relative to first video "
                                      f"frame ({how_t0}, timing.txt); CSI end vs video end {end_err:+.2f} s")
        print(f"WARNING: clock-based offset {off:+.1f} s looks implausible (CSI would end {end_err:+.1f} s from "
              f"the video end); falling back to end alignment.")
    else:
        missing = []
        if t0 is None:
            missing.append("t0_epoch (re-run csi_to_npz.py, or pass --npz-dir / --t0)")
        if not isinstance(vs, float):
            missing.append("video_start_epoch in timing.txt")
        print("WARNING: missing " + " and ".join(missing) + "; falling back to end alignment.")
    off = vid_dur + args.end_lag - csi_dur
    return off + args.shift, f"end alignment (CSI end = video end + {args.end_lag:.1f} s): offset {off:+.2f} s"


# ----------------------------------------------------------------------------- panel
def heat_from_double_ratio(t_vid, dr, rate, vid_dur):
    """Time-bin |double ratio| (packets, K) to `rate` columns/s, z-score per column. Returns (tb, heat)."""
    amp = np.abs(dr).astype(np.float32)
    if amp.ndim == 1:
        amp = amp[:, None]
    nb = max(int(np.ceil(vid_dur * rate)), 2)
    idx = np.floor(t_vid * rate).astype(int)
    ok = (idx >= 0) & (idx < nb)
    heat = np.full((nb, amp.shape[1]), np.nan, np.float32)
    cnt = np.bincount(idx[ok], minlength=nb).astype(float)
    have = cnt > 0
    xs = np.arange(nb)
    for k in range(amp.shape[1]):
        s = np.bincount(idx[ok], weights=amp[ok, k], minlength=nb)
        col = np.full(nb, np.nan)
        col[have] = s[have] / cnt[have]
        if have.sum() >= 2:
            col = np.interp(xs, xs[have], col[have], left=np.nan, right=np.nan)
        heat[:, k] = col
    mu = np.nanmedian(heat, axis=0)
    sd = np.nanstd(heat, axis=0) + 1e-9
    heat = np.clip((heat - mu) / sd, -2.5, 2.5)
    return (np.arange(nb) + 0.5) / rate, heat


def build_panel(vid_dur, heat, t_win, freqs, spec, view, motion, fmax, size_px, overview_png):
    W, H = size_px
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    L, R_ = 0.085, 0.89
    ax1 = fig.add_axes([L, 0.69, R_ - L, 0.26])
    ax2 = fig.add_axes([L, 0.37, R_ - L, 0.26])
    ax3 = fig.add_axes([L, 0.08, R_ - L, 0.20])
    cax1 = fig.add_axes([0.915, 0.69, 0.016, 0.26])
    cax2 = fig.add_axes([0.915, 0.37, 0.016, 0.26])

    im1 = ax1.imshow(np.nan_to_num(heat).T, aspect="auto", origin="lower", cmap="viridis", vmin=-2.5, vmax=2.5,
                     extent=[0, vid_dur, 0, heat.shape[1]], interpolation="nearest")
    ax1.set_ylabel("Stream #")
    ax1.set_title("Preprocessed double-ratio CSI |amplitude| (z-score)", fontsize=10, loc="left")
    fig.colorbar(im1, cax=cax1)

    ext = [t_win[0], t_win[-1], freqs[0], freqs[-1]]
    if view == "norm":
        im2 = ax2.imshow(np.log10(spec).T, aspect="auto", origin="lower", extent=ext, cmap="jet",
                         vmin=NOISE_LEVEL, vmax=0, interpolation="bilinear")
        ax2.set_title("Doppler spectrogram (each window normalised, log10)", fontsize=10, loc="left")
    else:
        im2 = ax2.imshow(spec.T, aspect="auto", origin="lower", extent=ext, cmap="magma",
                         vmin=-35, vmax=0, interpolation="bilinear")
        ax2.set_title("Doppler spectrogram (dB, absolute scale)", fontsize=10, loc="left")
    ax2.set_ylim(-fmax, fmax)
    ax2.set_ylabel("Doppler (Hz)")
    fig.colorbar(im2, cax=cax2)

    ax3.plot(t_win, motion, color="tab:red", lw=1.2)
    ax3.fill_between(t_win, 0, motion, color="tab:red", alpha=0.25)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Motion")
    ax3.set_xlabel("Time (s) from video start")
    ax3.grid(alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, vid_dur)
    for ax in (ax1, ax2):
        ax.tick_params(labelbottom=False)

    if overview_png:
        fig.savefig(overview_png, dpi=150)
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    Hh = base.shape[0]
    regions = []
    for ax in (ax1, ax2, ax3):
        bb = ax.get_window_extent()
        regions.append((int(round(bb.x0)), int(round(bb.x1)), int(round(Hh - bb.y1)), int(round(Hh - bb.y0))))
    plt.close(fig)
    return base, regions


# ----------------------------------------------------------------------------- main
def run(args):
    folder = Path(args.folder).expanduser()
    video = folder / "video.mp4"
    if not video.exists():
        die(f"{video} not found")
    vw0, vh0, vid_dur = probe_video(video, args.fps)
    print(f"Video: {vw0}x{vh0}, {vid_dur:.1f} s")

    dop_f, dr_f, conv_f = locate(folder, args)
    print(f"Doppler file : {dop_f}")
    print(f"Double ratio : {dr_f}")
    print(f"Converter npz: {conv_f if conv_f else '(not found)'}")

    D = np.load(dop_f, allow_pickle=False)
    R = np.load(dr_f, allow_pickle=False)
    t_win = D["t_windows"].astype(float)
    freqs = D["freqs_hz"].astype(float)
    spec = D["doppler"] if args.view == "norm" else D["doppler_db"]
    motion = D["motion"].astype(float)
    t_pk = R["t"].astype(float)
    dr = R["double_ratio"]
    activity = str(D["activity"]) if "activity" in D.files else ""
    csi_dur = float(t_pk[-1])

    off, how = get_offset(folder, conv_f, args, csi_dur, vid_dur)
    print(f"Alignment: {how}\n  total shift applied to CSI: {off:+.2f} s (includes --shift {args.shift:+.2f})")

    t_win_v = t_win + off
    t_pk_v = t_pk + off
    if ((t_win_v > 0) & (t_win_v < vid_dur)).sum() < 5:
        die("CSI and video barely overlap in time - check timing.txt, or use --shift / --t0")
    cover = 100.0 * (min(t_win_v[-1], vid_dur) - max(t_win_v[0], 0)) / vid_dur
    print(f"CSI covers {cover:.0f}% of the video ({max(t_win_v[0], 0):.1f}s .. {min(t_win_v[-1], vid_dur):.1f}s)")

    heat = heat_from_double_ratio(t_pk_v, dr, args.vis_rate, vid_dur)[1]

    vh = 720
    vw = int(round(vw0 * vh / vh0 / 2) * 2)
    panel_w = 960
    overview = None if args.no_png else folder / "csi_overview.png"
    base, regions = build_panel(vid_dur, heat, t_win_v, freqs, spec, args.view, motion,
                                args.fmax, (panel_w, vh), overview)
    Hp, Wp = base.shape[:2]
    if Hp != vh:
        die(f"internal: panel height {Hp} != video height {vh}")
    base_dim = (base * 0.45).astype(np.uint8)
    Wtot, Htot = vw + Wp, vh
    print(f"Output frame: {Wtot}x{Htot}")

    out = folder / ("csi_video_sync_preview.mp4" if (args.max_seconds or args.start) else "csi_video_sync.mp4")
    start = args.start
    dec_cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        dec_cmd += ["-ss", str(start)]
    dec_cmd += ["-i", str(video)]
    if args.max_seconds:
        dec_cmd += ["-t", str(args.max_seconds)]
    dec_cmd += ["-vf", f"fps={args.fps},scale={vw}:{vh}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    enc_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{Wtot}x{Htot}", "-r", str(args.fps), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out)]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    fbytes = vw * vh * 3
    canvas = np.zeros((Htot, Wtot, 3), np.uint8)
    idx = 0
    total = int((args.max_seconds or (vid_dur - start)) * args.fps)
    try:
        while True:
            buf = dec.stdout.read(fbytes)
            if len(buf) < fbytes:
                break
            canvas[:, :vw] = np.frombuffer(buf, np.uint8).reshape(vh, vw, 3)
            panel = base.copy()
            now = start + idx / args.fps
            for (x0, x1, y0, y1) in regions:
                xc = int(round(x0 + (x1 - x0) * min(now / vid_dur, 1.0)))
                panel[y0:y1, xc:x1] = base_dim[y0:y1, xc:x1]          # dim the "future"
                panel[y0:y1, max(xc - 1, x0):xc + 1] = (255, 40, 40)   # cursor
            canvas[:, vw:] = panel
            enc.stdin.write(canvas.tobytes())
            idx += 1
            if idx % 150 == 0:
                print(f"  rendered {idx}/{total} frames ({100 * idx / max(total, 1):.0f}%)", flush=True)
    finally:
        try:
            enc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        dec.stdout.close()
        dec.wait()
        enc.wait()
    if idx == 0 or enc.returncode != 0:
        die("rendering failed (no frames decoded or encoder error)")

    info = [
        f"Experiment      : {folder.name}   activity: {activity or 'n/a'}",
        f"Output          : {out.name}  {Wtot}x{Htot} @ {args.fps} fps, {idx} frames ({idx / args.fps:.1f} s), "
        f"{os.path.getsize(out) / 1e6:.1f} MB",
        f"CSI data        : {dop_f.stem}",
        f"  double ratio  : {dr.shape[0]} packets x {dr.shape[1] if dr.ndim > 1 else 1} streams, "
        f"{dr.shape[0] / max(csi_dur, 1e-9):.0f} pkt/s over {csi_dur:.1f} s",
        f"  Doppler       : {spec.shape[0]} windows x {spec.shape[1]} bins, fs {float(D['fs']):.0f} Hz, "
        f"window {int(D['num_symbols'])} samples, delta_v {float(D['delta_v'])} m/s",
        f"Alignment       : {how}",
        f"  CSI coverage  : {cover:.0f}% of the video",
    ]
    text = "\n".join(info)
    print("\nDone: " + str(out) + "\n" + text)
    (folder / ("report_preview.txt" if (args.max_seconds or args.start) else "report.txt")).write_text(text + "\n")
    if overview:
        print(f"Still overview: {overview}")
    print("If the pattern looks offset from the video (e.g. the clap appears early/late), "
          "re-run with --shift SECONDS (positive = CSI later in the video).")


def main():
    ap = argparse.ArgumentParser(description="Side-by-side video + preprocessed CSI (double ratio, Doppler)")
    ap.add_argument("folder", help="experiment folder with video.mp4 and timing.txt")
    ap.add_argument("--pre", help="folder given as --output to preprocess_csi.py "
                                  "(contains doppler/ and double_ratio/). Default: <experiment parent>/preprocessed")
    ap.add_argument("--npz-dir", help="folder with the csi_to_npz.py files (for t0_epoch). "
                                      "Default: <parent>/npz, then searched under the parent")
    ap.add_argument("--name", help="part of the preprocessed file name if several match the folder name")
    ap.add_argument("--t0", type=float, default=None, help="override: epoch time of the first CSI packet")
    ap.add_argument("--view", choices=["db", "norm"], default="db",
                    help="db = absolute dB scale (default); norm = each window normalised, jet, log10")
    ap.add_argument("--fps", type=int, default=30, help="output frame rate (default 30)")
    ap.add_argument("--fmax", type=float, default=30.0, help="max |Doppler| shown in Hz (default 30)")
    ap.add_argument("--vis-rate", type=float, default=20.0, help="heatmap columns per second (default 20)")
    ap.add_argument("--shift", type=float, default=0.0, help="move CSI later (+) / earlier (-) by this many seconds")
    ap.add_argument("--end-lag", type=float, default=0.3, help="fallback alignment: CSI ends this long after video")
    ap.add_argument("--start", type=float, default=0.0, help="start rendering at this video second")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="render only N seconds (quick test)")
    ap.add_argument("--no-png", action="store_true", help="do not write csi_overview.png")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
