"""
percom_common.py - shared helpers that connect our own recordings (PicoScenes CSI + webcam video)
to the instructor's SiWiS code (KateIronic/siwis-pose-estimation). The repository itself is not modified.

What we replaced
    nexmon .pcap loader (data/csi_pcap_loader.py)  ->  csi_features(): reads the csi_to_npz.py file
    build_labels.py (frames named by epoch time)   ->  prepare_percom.py: frames come straight from video.mp4
    train.py / evaluate.py (one recording)         ->  train_percom.py: many recordings, split BY PERSON

What we kept exactly as it is: models/, losses/, data/dataset.py (SiWiSDataset), the label formats, the metrics
(mask mAP, PCK) and the config values - except RESAMPLE_RATE_HZ (see set_sampling_rate).

The scripts import the repository's config.py, so they have to be run from the repository root.
scripts/setup_siwis.sh takes care of that by copying our files into a clone of the repository.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

import config

REC = re.compile(r"^(?P<act>[A-Za-z]+)_S(?P<s>\d+)_(?P<person>[A-Za-z0-9]+)$")
RIGHT_SHOULDER, LEFT_HIP = 6, 11                 # COCO indices (same as evaluate.py)
LEFT_WRIST, RIGHT_WRIST = 9, 10


# ----------------------------------------------------------------------------- recordings
def find_recordings(src, acts=(), persons=(), exclude_persons=()):
    """<src>/<ACT>/<ACT>_S<n>_<Person>/  ->  list of dicts."""
    src = Path(src).expanduser()
    recs = []
    for adir in sorted(p for p in src.iterdir() if p.is_dir()):
        if acts and adir.name.upper() not in {a.upper() for a in acts}:
            continue
        for r in sorted(p for p in adir.iterdir() if p.is_dir()):
            m = REC.match(r.name)
            if not m:
                continue
            person = m["person"].capitalize()
            if persons and person.lower() not in {p.lower() for p in persons}:
                continue
            if person.lower() in {p.lower() for p in exclude_persons}:
                continue
            recs.append(dict(folder=r, act=m["act"].lower(), session=f"S{int(m['s'])}", person=person,
                             sid=f"{m['act'].lower()}_{person}_S{int(m['s'])}", name=r.name))
    return recs


def read_timing(folder):
    out = {}
    p = Path(folder) / "timing.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    out[k.strip()] = v.strip()
    return out


# ----------------------------------------------------------------------------- CSI -> feature sequence
def set_sampling_rate(rate_hz):
    """
    The original repository resamples to 300 Hz. Our converter (csi_to_npz.py) keeps about 150 packets
    per second, so 150 Hz is the honest rate, and the 1.7 s window then has 255 samples instead of 510.
    dataset.py reads config.SIGNAL_SEQ_LEN_SAMPLES at run time, so setting it here is enough and the
    repository does not need to be edited.
    """
    config.RESAMPLE_RATE_HZ = rate_hz
    config.SIGNAL_SEQ_LEN_SAMPLES = round(config.SIGNAL_DURATION_SEC * rate_hz)


def csi_features(npz_path, rate_hz):
    """
    Same feature as the repo's build_feature_sequence (Eqn. 11, amplitude version):
        per packet, per stream: amplitude / mean amplitude, then std across subcarriers.
    The two streams are our two receive antennas (columns 0::2 and 1::2 of the converter file) instead
    of the repository's two transmit-chain streams, which it recovers with KMeans.
    Returns (K, 1 + 2) float64: column 0 = absolute epoch seconds (t0_epoch + t), then antenna A, B.
    """
    z = np.load(npz_path, allow_pickle=False)
    data, t = z["data"], z["t"].astype(np.float64)
    t0 = float(z["t0_epoch"]) if "t0_epoch" in z.files else 0.0
    if t0 <= 0:
        raise RuntimeError(f"{Path(npz_path).name}: no t0_epoch (absolute start time) - re-run csi_to_npz.py")
    order = np.argsort(t)
    data, t = data[order], t[order]
    streams = []
    for a in (0, 1):
        amp = np.abs(data[:, a::2]).astype(np.float32)
        m = amp.mean(axis=1, keepdims=True)
        ok = (m[:, 0] > 0)
        amp = amp / np.where(m == 0, 1, m)
        streams.append(np.where(ok, amp.std(axis=1), np.nan))
    t_abs = t0 + t
    good = ~(np.isnan(streams[0]) | np.isnan(streams[1]))
    t_abs = t_abs[good]
    grid = np.arange(t_abs[0], t_abs[-1], 1.0 / rate_hz)
    cols = [np.interp(grid, t_abs, s[good]) for s in streams]
    return np.stack([grid] + cols, axis=1)


# ----------------------------------------------------------------------------- video
def probe_size(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    try:
        w, h = [int(x) for x in r.stdout.strip().split(",")[:2]]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot read {video} (is it playable?): {r.stderr[-200:]}") from e
    return w, h


def decode_frames(video, fps, out_h=None, gray=False, out_w=None):
    """Yield frames at `fps` (RGB uint8, or gray). Frame k is at video time k / fps."""
    w, h = probe_size(video)
    if out_h is None:
        out_h = h
    if out_w is None:
        out_w = int(round(w * out_h / h / 2) * 2)
    pix = "gray" if gray else "rgb24"
    nb = out_w * out_h * (1 if gray else 3)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vf", f"fps={fps},scale={out_w}:{out_h}",
           "-f", "rawvideo", "-pix_fmt", pix, "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = p.stdout.read(nb)
            if len(buf) < nb:
                break
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(out_h, out_w) if gray else a.reshape(out_h, out_w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def video_duration(video):
    """Seconds of video (ffprobe; frame count as a fallback for fragmented mp4 without a stored duration)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    try:
        d = float(r.stdout.strip())
        if d > 0:
            return d
    except ValueError:
        pass
    return sum(1 for _ in decode_frames(video, 5, out_h=36, out_w=64, gray=True)) / 5.0


def clock_check(video, feats, video_start_epoch):
    """
    Sanity check of the video and CSI clocks (the repository README warns that a 1.4 s error already
    makes every label wrong). record.sh stops the video first and the CSI a moment later, so
        end_gap = CSI end - video length - video_start_epoch(timing.txt)
    is roughly the camera start-up delay plus the stop gap. 0 to about 2 s is normal. More than 3 s or less
    than -1 s means the timing of this recording is suspect. If you know the camera really starts X s later
    than timing.txt says, pass --sync_shift X.
    """
    dur = video_duration(video)
    return dict(video_duration=float(dur), end_gap=float(feats[-1, 0] - dur - video_start_epoch))


# ----------------------------------------------------------------------------- labels -> heatmaps
class LazyHeatmaps:
    """(N, K, H, W) union-over-people Gaussian heatmaps, computed per item from stored keypoints."""

    def __init__(self, kp_xy, kp_vis, h=config.HEATMAP_H, w=config.HEATMAP_W, sigma=2.0):
        self.kp_xy, self.kp_vis, self.h, self.w, self.sigma = kp_xy, kp_vis, h, w, sigma
        self.yy, self.xx = np.mgrid[0:h, 0:w]

    def __len__(self):
        return len(self.kp_xy)

    def __getitem__(self, i):
        out = np.zeros((config.NUM_KEYPOINTS, self.h, self.w), np.float32)
        for n in range(self.kp_xy.shape[1]):
            for k in range(config.NUM_KEYPOINTS):
                if self.kp_vis[i, n, k]:
                    x, y = self.kp_xy[i, n, k]
                    g = np.exp(-((self.xx - x) ** 2 + (self.yy - y) ** 2) / (2 * self.sigma ** 2))
                    out[k] = np.maximum(out[k], g)
        return out


def load_recording(rec_dir):
    """-> SiWiSDataset for one prepared recording (labels + features cached by prepare_percom.py)."""
    from data.dataset import SiWiSDataset
    d = Path(rec_dir)
    feats = np.load(d / "features.npy")
    ts = np.load(d / "frame_timestamps.npy")
    mask = np.load(d / "mask_labels.npy")
    kp_xy = np.load(d / "kp_xy.npy")
    kp_vis = np.load(d / "kp_vis.npy")
    coords = np.zeros(kp_xy.shape[:3] + (2,), np.int64)               # (row, col)
    coords[..., 0] = np.clip(kp_xy[..., 1].astype(np.int64), 0, config.HEATMAP_H - 1)
    coords[..., 1] = np.clip(kp_xy[..., 0].astype(np.int64), 0, config.HEATMAP_W - 1)
    return SiWiSDataset(feats, ts, mask, LazyHeatmaps(kp_xy, kp_vis), coords, kp_vis)


# ----------------------------------------------------------------------------- metrics (same as evaluate.py)
def mask_map(pred_masks, gt_masks):
    pb, gb = pred_masks >= 0.5, gt_masks >= 0.5
    inter = (pb & gb).sum(axis=(1, 2)).astype(np.float64)
    union = (pb | gb).sum(axis=(1, 2)).astype(np.float64)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    ths = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap = {a: float((iou >= a).mean()) for a in ths}
    return float(np.mean(list(ap.values()))), ap


def argmax_points(heatmaps):
    N, K, H, W = heatmaps.shape
    idx = heatmaps.reshape(N, K, -1).argmax(-1)
    r, c = np.unravel_index(idx, (H, W))
    return np.stack([r, c], -1).astype(np.float64)


def pck(pred_pts, gt, vis, joints=None, thresholds=(0.1, 0.2, 0.3, 0.5)):
    """Single-person PCK (Eqn. 17, torso-normalised). joints: restrict to these COCO joint ids."""
    hits = {a: [] for a in thresholds}
    used = 0
    sel = np.zeros(vis.shape[1], bool)
    sel[list(range(vis.shape[1])) if joints is None else list(joints)] = True
    for i in range(len(gt)):
        if not (vis[i, RIGHT_SHOULDER] and vis[i, LEFT_HIP]):
            continue
        norm = np.linalg.norm(gt[i, RIGHT_SHOULDER] - gt[i, LEFT_HIP])
        if norm < 1e-6:
            continue
        v = vis[i] & sel
        if v.sum() == 0:
            continue
        used += 1
        d = np.linalg.norm(pred_pts[i] - gt[i], axis=-1) / norm
        for a in thresholds:
            hits[a].append(((d <= a) & v).sum() / v.sum())
    return {a: (float(np.mean(h)) if h else float("nan")) for a, h in hits.items()}, used


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
