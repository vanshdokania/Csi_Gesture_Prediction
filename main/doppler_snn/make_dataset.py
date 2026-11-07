#!/usr/bin/env python3
"""
make_dataset.py - one command that takes the raw recordings from the SSD and produces the
labelled frame dataset used by the Doppler SNN.

    python3 doppler_snn/make_dataset.py --src "/media/$USER/My Passport/percom" --out ~/percom/dataset

Expected layout of --src (this is how the recordings are stored on our SSD):
    BHU/BHU_S1_Priyanshu/rx_*.csi (+ video.mp4, timing.txt ...)
    BHU/BHU_S2_Sanchit/...      LHU/...     RHU/...          (<ACTIVITY>/<ACTIVITY>_S<n>_<Person>/)

For every recording folder it runs, skipping what is already done (so you can stop and re-run):
    1. csi_to_npz.py     .csi -> .npz                    (needs the PicoScenes toolbox venv to be active)
    2. preprocess_csi.py double ratio + Doppler spectrogram
    3. frame labelling   1 s frames every 0.5 s, label = the activity of the recording
Intermediate files go to --work (default ~/percom/work) and are reused.

Result in --out:
    dataset/
      classes.json         bhu 0, lhu 1, rhu 2 ... (append-only: new activities get the next id)
      labels_all.csv       every frame of every recording in one table (the file train_snn.py loads)
      index.csv            one row per recording (frames, valid frames, source files)
      bhu/  lhu/  rhu/     one labels file per recording:  <Person>_<S#>.csv   (+ .npz with the spectrogram
                           frames when --save-x is given)

Label rule: each recording is one gesture type repeated over and over, so every frame simply gets the
recording's gesture as its label.
Frames where the Wi-Fi motion signal is dead (mean motion < --min-motion, for example because packets
went missing) are kept in the table but get valid=0. train_snn.py skips them by default.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "csi_tools"       # csi_to_npz.py and preprocess_csi.py live here
REC = re.compile(r"^(?P<act>[A-Za-z]+)_S(?P<s>\d+)_(?P<person>[A-Za-z0-9]+)$")


def find_recordings(src, acts):
    recs = []
    for adir in sorted(p for p in src.iterdir() if p.is_dir()):
        if acts and adir.name.upper() not in acts:
            continue
        for r in sorted(p for p in adir.iterdir() if p.is_dir()):
            m = REC.match(r.name)
            if not m:
                continue
            recs.append(dict(folder=r, act=m["act"].lower(), s=f"S{int(m['s'])}", person=m["person"].capitalize()))
    return recs


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def frames_from_doppler(D, frame_s, hop_s, min_motion):
    tw = D["t_windows"].astype(float)
    db = D["doppler_db"].astype(np.float32)
    motion = D["motion"].astype(float)
    freqs = D["freqs_hz"].astype(float)
    fs_w = 1.0 / float(np.median(np.diff(tw)))
    n_t = int(round(frame_s * fs_w))
    hop = max(1, int(round(hop_s * fs_w)))
    starts = np.arange(0, len(tw) - n_t + 1, hop)
    if len(starts) == 0:
        raise RuntimeError("recording shorter than one frame")
    p_all = 10 ** (db / 10)
    absf = np.abs(freqs)
    X, rows = [], []
    for k, s in enumerate(starts):
        sl = slice(s, s + n_t)
        p = p_all[sl].sum(axis=0)
        p = p / (p.sum() + 1e-30)
        cen = float((absf * p).sum())
        spr = float(np.sqrt((((absf - cen) ** 2) * p).sum()))
        mm = float(motion[sl].mean())
        X.append(db[sl])
        rows.append(dict(frame=k, t_start=round(tw[s], 3), t_end=round(tw[s + n_t - 1], 3),
                         motion_mean=round(mm, 4), motion_max=round(float(motion[sl].max()), 4),
                         doppler_centroid_hz=round(cen, 3), doppler_spread_hz=round(spr, 3),
                         doppler_signed_hz=round(float((freqs * p).sum()), 3), valid=int(mm >= min_motion)))
    return np.stack(X).astype(np.float16), rows, freqs, fs_w


def main():
    ap = argparse.ArgumentParser(description="Raw recordings (SSD) -> labelled frame dataset")
    ap.add_argument("--src", required=True, help='folder holding BHU/ LHU/ RHU/ (e.g. "/media/$USER/My Passport/percom")')
    ap.add_argument("--out", default="~/percom/dataset", help="dataset folder (default ~/percom/dataset)")
    ap.add_argument("--work", default="~/percom/work", help="cache for converted/preprocessed files (default ~/percom/work)")
    ap.add_argument("--acts", nargs="*", default=[], help="only these activity folders, e.g. --acts LHU")
    ap.add_argument("--frame", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--min-motion", type=float, default=0.03, help="below this mean motion a frame is valid=0 (default 0.03)")
    ap.add_argument("--rate", type=float, default=150.0, help="packets/s kept by csi_to_npz (default 150)")
    ap.add_argument("--save-x", action="store_true", help="also save the spectrogram frames (X) per recording as .npz")
    ap.add_argument("--force", action="store_true", help="redo labels even if they exist (conversion is still reused)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N recordings (test)")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    if not src.is_dir():
        sys.exit(f"Error: {src} not found - is the SSD mounted? (you can check with: ls /media/$USER)")
    out, work = Path(args.out).expanduser(), Path(args.work).expanduser()
    npz_dir, pre = work / "npz", work / "output"
    for d in (out, npz_dir, pre / "doppler"):
        d.mkdir(parents=True, exist_ok=True)
    recs = find_recordings(src, {a.upper() for a in args.acts})
    if not recs:
        sys.exit(f"Error: no <ACT>_S<n>_<Person> recording folders found under {src}")
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} recordings found under {src}\n")

    cpath = out / "classes.json"
    classes = json.loads(cpath.read_text()) if cpath.exists() else {}
    index, failed = {}, []
    if (out / "index.csv").exists():
        for r in csv.DictReader(open(out / "index.csv")):
            index[r["sample_id"]] = r

    for i, r in enumerate(recs, 1):
        name = r["folder"].name
        sid = f"{r['act']}_{r['person']}_{r['s']}"
        tag = f"[{i}/{len(recs)}] {name}"
        try:
            # ---- 1. csi -> npz
            found = sorted(npz_dir.glob(f"*__{name}_rx_*.npz"))
            if not found:
                if not list(r["folder"].glob("rx_*.csi")):
                    raise RuntimeError("no rx_*.csi file in the folder")
                print(f"{tag}: converting .csi -> .npz ...", flush=True)
                code, log = run([sys.executable, str(TOOLS / "csi_to_npz.py"), str(r["folder"]), "--out", str(npz_dir),
                                 "--activity", r["act"], "--rate", str(args.rate)])
                found = sorted(npz_dir.glob(f"*__{name}_rx_*.npz"))
                if code != 0 or not found:
                    raise RuntimeError("conversion failed:\n" + log[-800:])
            npz = found[0]

            # ---- 2. preprocess
            dop = pre / "doppler" / npz.name
            if not dop.exists():
                print(f"{tag}: preprocessing ...", flush=True)
                stage = work / "stage"
                stage.mkdir(exist_ok=True)
                for old in stage.glob("*.npz"):
                    old.unlink()
                (stage / npz.name).symlink_to(npz.resolve())
                code, log = run([sys.executable, str(TOOLS / "preprocess_csi.py"), "--input", str(stage),
                                 "--output", str(pre), "--no-png"])
                if code != 0 or not dop.exists():
                    raise RuntimeError("preprocessing failed:\n" + log[-800:])

            # ---- 3. label frames
            if sid in index and not args.force and (out / r["act"] / f"{r['person']}_{r['s']}.csv").exists():
                print(f"{tag}: already labelled")
                continue
            if r["act"] not in classes:
                classes[r["act"]] = len(classes)
            cid = classes[r["act"]]
            D = np.load(dop, allow_pickle=False)
            X, rows, freqs, fs_w = frames_from_doppler(D, args.frame, args.hop, args.min_motion)
            adir = out / r["act"]
            adir.mkdir(exist_ok=True)
            base = adir / f"{r['person']}_{r['s']}"
            cols = ["sample_id", "activity", "person", "sample", "frame", "t_start", "t_end", "label", "label_id",
                    "valid", "motion_mean", "motion_max", "doppler_centroid_hz", "doppler_spread_hz", "doppler_signed_hz"]
            with open(f"{base}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for row in rows:
                    w.writerow(dict(sample_id=sid, activity=r["act"], person=r["person"], sample=r["s"],
                                    label=r["act"], label_id=cid, **row))
            if args.save_x:
                np.savez_compressed(f"{base}.npz", X=X, y=np.full(len(rows), cid, np.int64),
                                    valid=np.array([x["valid"] for x in rows], bool),
                                    t_start=np.array([x["t_start"] for x in rows]), freqs_hz=freqs,
                                    label=r["act"], person=r["person"], sample=r["s"], windows_per_s=fs_w)
            nv = sum(x["valid"] for x in rows)
            index[sid] = dict(sample_id=sid, activity=r["act"], label_id=cid, person=r["person"], sample=r["s"],
                              frames=len(rows), valid_frames=nv, source=npz.stem,
                              labels=str(Path(r["act"]) / f"{r['person']}_{r['s']}.csv"))
            print(f"{tag}: {len(rows)} frames, {nv} valid, {len(rows) - nv} dead-signal -> {base}.csv", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{tag}: failed - {e}")
            failed.append(name)

    cpath.write_text(json.dumps(classes, indent=2))
    cols = ["sample_id", "activity", "label_id", "person", "sample", "frames", "valid_frames", "source", "labels"]
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(index.values(), key=lambda r: (r["activity"], r["person"], r["sample"])):
            w.writerow({k: r.get(k, "") for k in cols})
    # combined table from all per-recording csv files
    allrows, header = [], None
    for f in sorted(out.glob("*/*.csv")):
        with open(f) as fh:
            rd = csv.reader(fh)
            h = next(rd)
            header = header or h
            allrows += list(rd)
    if header:
        with open(out / "labels_all.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(allrows)
    print(f"\nclasses: {classes}")
    print(f"{len(index)} recordings, {len(allrows)} frames in total -> {out}/labels_all.csv")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}  (fix the cause and run again, finished recordings are skipped)")


if __name__ == "__main__":
    main()
