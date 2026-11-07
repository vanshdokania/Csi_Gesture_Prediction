#!/usr/bin/env python3
"""
prepare_percom.py - stage 1 of the pose pipeline.
It takes our own recordings (video.mp4 + CSI) and produces what the SiWiS model needs:
CSI features on one side and body labels (mask + 17 joints) from the video on the other.
This replaces steps 1-4 of the original SiWiS repository, which expects nexmon captures.

    python prepare_percom.py --src "/media/$USER/My Passport/percom" --out ~/percom/siwis_data

For every recording folder  <src>/<ACT>/<ACT>_S<n>_<Person>/  (Divij and Nikita are excluded by default) it
  1. finds the converted CSI (csi_to_npz.py output, ~/percom/work/npz) or converts the .csi (needs the
     PicoScenes venv active) and builds the SiWiS signal feature (2 antenna streams, 150 Hz) -> features.npy
  2. decodes video.mp4 at --label_fps frames/second; time of frame k = video_start_epoch (timing.txt) + k/fps
  3. checks the video/CSI clock alignment (CSI end vs video length) and warns if a recording looks shifted
  4. runs Mask R-CNN (body mask) + a keypoint network (17 COCO joints) on every frame whose 1.7 s CSI window exists
  -> <out>/rec/<id>/{features.npy, frame_timestamps.npy, mask_labels.npy, kp_xy.npy, kp_vis.npy, meta.json, preview.png}

Finished recordings are skipped, so you can stop with Ctrl+C and start again later. Labelling is the
slow part (seconds per frame on a CPU), so we ran it on the GPU server.

Settings used for the report: 2 labelled frames per second, frames resized to 480 px height,
person detections need a score of at least 0.7, mask pixels at least 0.5, a joint counts as
visible when its keypoint score is above 1.0.

Keypoint backends:  torchvision (default, no OpenMMLab needed) | mmpose (HRNet, as in the instructor's
repository). The report uses the torchvision one: Mask R-CNN for the body outline and
Keypoint R-CNN for the 17 joints.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config
import percom_common as pc


# ----------------------------------------------------------------------------- label backends
class TorchvisionBackend:
    """Mask R-CNN (mask) + Keypoint R-CNN (17 COCO joints). Weights download automatically on first use."""

    def __init__(self, device, kp_thresh, det_thresh=0.7):
        import torch
        from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights,
                                                  keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn)
        self.torch, self.device = torch, device
        self.mask = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
        self.kp = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
        self.kp_thresh, self.det_thresh = kp_thresh, det_thresh

    def label(self, frame):
        torch = self.torch
        h, w = frame.shape[:2]
        x = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mo = self.mask(x)[0]
            ko = self.kp(x)[0]
        mask = np.zeros((h, w), np.float32)
        for lab, sc, m in zip(mo["labels"], mo["scores"], mo["masks"]):
            if int(lab) == 1 and float(sc) >= 0.5:
                mask = np.maximum(mask, m[0].cpu().numpy())
        people = []
        for sc, box, kp, ks in zip(ko["scores"], ko["boxes"], ko["keypoints"],
                                   ko.get("keypoints_scores", [None] * len(ko["scores"]))):
            if float(sc) < self.det_thresh:
                continue
            kp = kp.cpu().numpy()
            conf = ks.cpu().numpy() if ks is not None else np.full(kp.shape[0], 10.0)
            b = box.cpu().numpy()
            people.append(((b[2] - b[0]) * (b[3] - b[1]), np.concatenate([kp[:, :2], (conf > self.kp_thresh)[:, None]], 1)))
        people.sort(key=lambda p: -p[0])
        return mask, [p[1] for p in people]


class MMPoseBackend:
    """Mask R-CNN (mask + person boxes) + HRNet via mmpose, the same way the instructor's data/vision_labels.py does it. Not used for our results."""

    def __init__(self, device, hrnet_config, hrnet_checkpoint, kp_thresh=0.3):
        from data.vision_labels import KeypointLabelExtractor, MaskLabelExtractor
        self.mask = MaskLabelExtractor(device=device)
        self.kp = KeypointLabelExtractor(hrnet_config, hrnet_checkpoint, device=device)
        self.kp_thresh = kp_thresh

    def label(self, frame):
        from scipy import ndimage
        mask = self.mask.extract(frame)
        lab, n = ndimage.label(mask > 0.5)
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(ys) >= 500:
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        if not boxes:
            return mask, []
        boxes = np.array(sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])), np.float32)
        kps = self.kp.extract(frame, boxes)                                # (N, 17, 3) x, y, score
        return mask, [np.concatenate([k[:, :2], (k[:, 2:3] > self.kp_thresh)], 1) for k in kps]


class SyntheticBackend:
    """Only for testing the plumbing: finds one bright rectangle and places 17 fake joints on it. Not for real video."""
    FR = np.array([[.5, .05], [.45, .03], [.55, .03], [.4, .05], [.6, .05], [.3, .25], [.7, .25], [.15, .4], [.85, .4],
                   [.0, .15], [1., .15], [.35, .55], [.65, .55], [.35, .75], [.65, .75], [.35, 1.], [.65, 1.]])

    def label(self, frame):
        m = (frame.min(axis=2) > 200).astype(np.float32)
        ys, xs = np.where(m > 0)
        if len(ys) < 50:
            return m, []
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        kp = np.zeros((17, 3), np.float32)
        kp[:, 0] = x0 + self.FR[:, 0] * (x1 - x0)
        kp[:, 1] = y0 + self.FR[:, 1] * (y1 - y0)
        kp[:, 2] = 1
        return m, [kp]


# ----------------------------------------------------------------------------- helpers
def find_npz(npz_dir, name):
    f = sorted(Path(npz_dir).glob(f"*__{name}_rx_*.npz"))
    return f[0] if f else None


def convert_csi(rec, npz_dir, percom, python):
    print("   converting .csi -> .npz (slow, the file is about 1 GB) ...", flush=True)
    r = subprocess.run([python, str(Path(percom) / "csi_to_npz.py"), str(rec["folder"]), "--out", str(npz_dir),
                        "--activity", rec["act"]], capture_output=True, text=True)
    f = find_npz(npz_dir, rec["name"])
    if r.returncode != 0 or f is None:
        msg = (r.stdout + r.stderr)[-600:]
        if "picoscenes" in msg:
            msg += ("\n   -> the interpreter used for conversion has no 'picoscenes' module. Pass the PicoScenes toolbox "
                    "venv python:  --convert_python /path/to/.venv/bin/python")
        raise RuntimeError("conversion failed:\n" + msg)
    return f


def preview(path, frames, masks, kps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(frames)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 2.6))
    ax = np.atleast_1d(ax)
    for a, f, m, k in zip(ax, frames, masks, kps):
        a.imshow(f)
        a.imshow(m, alpha=0.35, cmap="Reds")
        for person in k:
            a.scatter(person[:, 0], person[:, 1], s=8, c=["lime" if v else "none" for v in person[:, 2]])
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def process(rec, args, backend, out_dir):
    d = out_dir / "rec" / rec["sid"]
    if (d / "meta.json").exists() and not args.force:
        print(f"{rec['sid']}: already prepared")
        return None
    d.mkdir(parents=True, exist_ok=True)
    video = rec["folder"] / "video.mp4"
    if not video.exists():
        raise RuntimeError("video.mp4 missing")
    timing = pc.read_timing(rec["folder"])
    vstart = timing.get("video_start_epoch")
    if not isinstance(vstart, float):
        raise RuntimeError("timing.txt has no video_start_epoch")

    # 1. CSI feature sequence
    npz = find_npz(args.npz_dir, rec["name"]) or convert_csi(rec, args.npz_dir, args.percom, args.convert_python)
    feats = pc.csi_features(npz, args.rate)
    np.save(d / "features.npy", feats)
    print(f"   CSI: {len(feats)} samples at {args.rate:g} Hz, {feats[-1, 0] - feats[0, 0]:.1f} s", flush=True)

    # 2./3. clock check
    sy = pc.clock_check(video, feats, vstart)
    print(f"   clock check: video {sy['video_duration']:.1f} s; CSI end - video length - timing.txt start = "
          f"{sy['end_gap']:+.2f} s (normal: 0 to 2 s)")
    if sy["end_gap"] > 3 or sy["end_gap"] < -1:
        print("   Warning: video and CSI clocks look misaligned for this recording, the labels may be shifted")
    applied = args.sync_shift
    t_off = vstart + applied

    # 4. frames -> labels
    win = config.SIGNAL_DURATION_SEC
    lo, hi = feats[0, 0] + win, feats[-1, 0]
    ts, masks, kp_xy, kp_vis, shown = [], [], [], [], ([], [], [])
    t_start = time.time()
    H, W = config.HEATMAP_H, config.HEATMAP_W
    from PIL import Image
    n_dec = 0
    for k, frame in enumerate(pc.decode_frames(video, args.label_fps, out_h=args.max_h)):
        n_dec += 1
        t = t_off + k / args.label_fps
        if t < lo or t > hi:
            continue
        if args.limit_frames and len(ts) >= args.limit_frames:
            break
        mask, people = backend.label(frame)
        if not people or mask.max() < 0.5:
            continue                                                    # nobody found, so no usable label
        h, w = frame.shape[:2]
        mrs = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H)), np.float32)
        xy = np.zeros((config.MAX_PEOPLE, config.NUM_KEYPOINTS, 2), np.float32)
        vis = np.zeros((config.MAX_PEOPLE, config.NUM_KEYPOINTS), bool)
        for n, p in enumerate(people[:config.MAX_PEOPLE]):
            xy[n, :, 0] = p[:, 0] * W / w
            xy[n, :, 1] = p[:, 1] * H / h
            vis[n] = p[:, 2] > 0.5
        ts.append(t)
        masks.append(mrs)
        kp_xy.append(xy)
        kp_vis.append(vis)
        if len(ts) in (1, 40, 90, 150):
            shown[0].append(frame.copy())
            shown[1].append(mask)
            shown[2].append([p for p in people[:2]])
        if len(ts) % 20 == 0:
            el = time.time() - t_start
            print(f"   {len(ts)} frames labelled ({el / len(ts):.1f} s/frame)", flush=True)
    if len(ts) < 20:
        raise RuntimeError(f"only {len(ts)} usable frames (of {n_dec} decoded): check the person is visible and the "
                           "video/CSI times overlap (timing.txt)")
    np.save(d / "frame_timestamps.npy", np.array(ts))
    np.save(d / "mask_labels.npy", np.stack(masks).astype(np.float16))
    np.save(d / "kp_xy.npy", np.stack(kp_xy))
    np.save(d / "kp_vis.npy", np.stack(kp_vis))
    if shown[0]:
        preview(d / "preview.png", *shown)
    pc.save_json(d / "meta.json", dict(sid=rec["sid"], person=rec["person"], activity=rec["act"], session=rec["session"],
                                       frames=len(ts), decoded=n_dec, label_fps=args.label_fps, rate_hz=args.rate,
                                       clock_check=sy, sync_shift_s=applied, backend=args.backend, source_npz=npz.name))
    print(f"{rec['sid']}: {len(ts)} labelled frames -> {d}")
    return len(ts)


def main():
    ap = argparse.ArgumentParser(description="Recordings (video + CSI) -> SiWiS labels and features")
    ap.add_argument("--src", required=True, help='folder with BHU/ LHU/ RHU/ (e.g. "/media/$USER/My Passport/percom")')
    ap.add_argument("--out", default="~/percom/siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz", help="converted CSI (made by make_dataset.py / csi_to_npz.py)")
    ap.add_argument("--percom", default=".", help="folder that contains csi_to_npz.py (only used if a .csi file still has to be converted)")
    ap.add_argument("--convert_python", default=sys.executable,
                    help="python used to convert .csi files; must be the PicoScenes toolbox venv's python "
                         "(e.g. ~/PicoscenesToolbox/.venv/bin/python). Default: the python running this script")
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"], help="people to leave out (default: Divij Nikita, as in the report)")
    ap.add_argument("--persons", nargs="*", default=[], help="only these persons")
    ap.add_argument("--acts", nargs="*", default=[], help="only these activity folders, e.g. BHU")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N recordings")
    ap.add_argument("--limit_frames", type=int, default=0, help="label at most N frames per recording (test)")
    ap.add_argument("--label_fps", type=float, default=2.0, help="video frames labelled per second (default 2; the SiWiS paper uses 10)")
    ap.add_argument("--max_h", type=int, default=480, help="frame height used for labelling (default 480)")
    ap.add_argument("--rate", type=float, default=150.0, help="CSI feature rate in Hz (default 150 = converter rate)")
    ap.add_argument("--sync_shift", type=float, default=0.0, help="manual extra shift of frame times, seconds")
    ap.add_argument("--backend", choices=["torchvision", "mmpose", "synthetic-selftest"], default="torchvision")
    ap.add_argument("--kp_thresh", type=float, default=1.0, help="torchvision: keypoint score above this counts as visible")
    ap.add_argument("--hrnet_config", default="checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py")
    ap.add_argument("--hrnet_checkpoint", default="checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth")
    ap.add_argument("--device", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.convert_python = str(Path(args.convert_python).expanduser())
    for a in ("npz_dir", "percom", "out"):
        setattr(args, a, str(Path(getattr(args, a)).expanduser()))
    pc.set_sampling_rate(args.rate)

    src = Path(args.src).expanduser()
    if not src.is_dir():
        sys.exit(f"Error: {src} not found - is the SSD mounted?")
    recs = pc.find_recordings(src, args.acts, args.persons, args.exclude)
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        sys.exit("Error: no recordings found")
    print(f"{len(recs)} recordings (excluded persons: {', '.join(args.exclude) or 'none'})")

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend == "torchvision":
        backend = TorchvisionBackend(dev, args.kp_thresh)
    elif args.backend == "mmpose":
        backend = MMPoseBackend(dev, args.hrnet_config, args.hrnet_checkpoint)
    else:
        backend = SyntheticBackend()
    print(f"labelling backend {args.backend} on {dev}\n")

    out = Path(args.out)
    failed = []
    for i, rec in enumerate(recs, 1):
        print(f"[{i}/{len(recs)}] {rec['sid']}  ({rec['name']})", flush=True)
        try:
            process(rec, args, backend, out)
        except Exception as e:  # noqa: BLE001
            print(f"   failed: {e}")
            failed.append(rec["sid"])
    done = sorted((out / "rec").glob("*/meta.json"))
    print(f"\n{len(done)} recordings prepared in {out / 'rec'}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)} - fix the cause and run again (finished ones are skipped)")


if __name__ == "__main__":
    main()
