#!/usr/bin/env python3
"""
csi_to_npz.py - turn PicoScenes .csi recordings into small .npz files that the rest of the
pipeline can read (a complex array "data" of packets x columns, plus the packet times).
The layout follows the WiLight / AxSense preprocessing code we based our Doppler step on.

    python3 csi_to_npz.py ~/percom/experiment_* --out ~/percom/npz

Every rx_*.csi found (recursively) in the given folders becomes one file
    <out>/<activity>__<name>.npz
where <activity> is read from an  activity.txt  file inside the recording's folder (for example the
word  clapping ), otherwise the folder name. Only files named rx_*.csi (the files PicoScenes
writes when receiving) are used.

Which two antennas are used: the AX200 gives us two receive antennas for the same transmitted
stream. The Doppler step divides one by the other, so only that pair is kept and everything
else is thrown away. The recordings arrive at roughly 150 packets per second.

The .npz holds
    data        complex64 (packets, 2*tones)  columns interleaved: tone0_antA, tone0_antB, tone1_antA, ...
                (so the WiLight ratio  data[:,0::2] / data[:,1::2]  = antenna A / antenna B per tone)
    t           float64 (packets,)            packet times in seconds from the first packet
    t0_epoch    float                         wall-clock time of the first packet (s), 0 if unknown
    center_freq_mhz, cbw_mhz, num_tx, num_rx, pair, tones_kept, rate_hz, activity, source

Needs the PicoScenes Python toolbox (activate its virtual environment first) and numpy.
"""
import argparse
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_video_sync import frame_offsets, find_key  # noqa: E402  (same folder)

CHUNK_FRAMES = 500   # frames parsed at a time (keeps RAM small)


def extract(raw):
    """Pull the useful fields out of PicoScenes frames.

    Returns a list of (tones, flat_csi, sender_address, system_time_s, device_time_raw, csi_info).
    """
    out = []
    for fr in raw:
        c = fr.get("CSI") if isinstance(fr, dict) else None
        if not c:
            continue
        try:
            nt = int(c.get("numTones") or 0)
            arr = np.asarray(c.get("CSI"), dtype=np.complex64)
        except Exception:  # noqa: BLE001
            continue
        if nt <= 0 or arr.size == 0 or arr.size % nt:
            continue
        addr = find_key(fr.get("StandardHeader"), ["Addr2"])
        try:
            addr = tuple(np.asarray(addr).ravel().tolist())
        except Exception:  # noqa: BLE001
            addr = str(addr)
        rx = fr.get("RxSBasic")
        sysns = find_key(rx, ["systemns"])
        t_sys = float(sysns) * 1e-9 if sysns and float(sysns) > 1e17 else None
        t_dev = find_key(rx, ["timestamp"])
        out.append((nt, arr, addr, t_sys, None if t_dev is None else float(t_dev),
                    dict(rx=dict(rx) if isinstance(rx, dict) else {},
                         numTx=c.get("numTx"), numRx=c.get("numRx"), CBW=c.get("CBW"))))
    return out


def roughness(m):
    a = np.abs(m)
    return np.abs(np.diff(a, axis=0)).mean() / (a.mean() + 1e-12)


def to_matrix(arr, nt, links, mode):
    if arr.ndim == 3:
        return arr.reshape(nt, -1)
    if arr.ndim == 2:
        return arr if arr.shape[0] == nt else arr.T
    return arr.reshape(links, nt).T if mode == "links_major" else arr.reshape(nt, links)


def rate_of(t_sys, t_dev):
    """Packets per second estimated from a chunk's timestamps (None if unknown)."""
    def r(t):
        t = np.sort(np.asarray(t, dtype=float))
        span = t[-1] - t[0]
        return len(t) / span if span > 0 else None
    if t_sys and all(v is not None for v in t_sys):
        return r(t_sys)
    if t_dev and all(v is not None for v in t_dev):
        for scale in (1e-6, 1e-9, 1e-3):
            rr = r(np.asarray(t_dev) * scale)
            if rr and 1 <= rr <= 20000:
                return rr
    return None


def final_times(ts, td, n, target_rate):
    """Packet times in seconds from the first packet, plus description and t0_epoch."""
    if ts is not None:
        return ts - ts[0], "PicoScenes system clock", float(ts[0])
    if td is not None:
        for scale, unit in ((1e-6, "us"), (1e-9, "ns"), (1e-3, "ms")):
            t = (td - td[0]) * scale
            dt = np.diff(t)
            dt = dt[dt > 0]
            if len(dt) and t[-1] > 0 and 2e-4 < np.median(dt) < 0.5:
                return t, f"device clock ({unit})", 0.0
    return np.arange(n) / float(target_rate), "no timestamps found, assuming even spacing", 0.0


def convert_file(f, out_path, activity, target_rate, pair_arg):
    from picoscenes import Picoscenes
    offs = frame_offsets(f)
    size_mb = os.path.getsize(f) / 1e6
    print(f"  {len(offs)} frames, {size_mb:.0f} MB")
    if not offs:
        raise RuntimeError("no complete frames in file")
    with open(f, "rb") as fh:
        first = fh.read(offs[1] if len(offs) > 1 else os.path.getsize(f))
    tmp = tempfile.NamedTemporaryFile(suffix=".csi", delete=False)
    tmp.write(first)
    tmp.close()
    ps = Picoscenes(tmp.name)          # the toolbox parses a whole file on creation: give it a tiny one

    best = mode = pair = None
    keep_every, counter = 1, 0
    mats, ts_l, td_l = [], [], []
    meta = {}
    n_total = n_best = 0
    try:
        for i in range(0, len(offs), CHUNK_FRAMES):
            n = min(CHUNK_FRAMES, len(offs) - i)
            ps.seek(str(f), int(offs[i]), int(n))
            frames = extract(ps.raw)
            ps.raw.clear()
            n_total += len(frames)

            if best is None:                       # work out the data layout from the first chunk
                if not frames:
                    continue
                cnt = Counter((fr[0], fr[1].size // fr[0], fr[2]) for fr in frames)
                best = cnt.most_common(1)[0][0]
                nt, links = best[0], best[1]
                same = [fr for fr in frames if (fr[0], fr[1].size // fr[0], fr[2]) == best]
                print(f"  main stream: {nt} tones x {links} links ({len(same)} of {len(frames)} frames in first chunk)")
                if links < 2:
                    raise RuntimeError("only one CSI link in this file, but the antenna ratio needs two. "
                                       "Check that both antennas of the AX200 are connected")
                mode = "tones_major"
                if same[0][1].ndim == 1:
                    sample = same[:50]
                    ra = np.mean([roughness(to_matrix(s[1], nt, links, "links_major")) for s in sample])
                    rb = np.mean([roughness(to_matrix(s[1], nt, links, "tones_major")) for s in sample])
                    mode = "links_major" if ra <= rb else "tones_major"
                    print(f"  layout: {mode} (roughness {ra:.2f} vs {rb:.2f})")
                ci = same[0][5]
                ntx, nrx = int(ci.get("numTx") or 1), int(ci.get("numRx") or 1)
                if pair_arg:
                    pair = tuple(int(x) for x in pair_arg.split(","))
                elif nrx >= 2 and ntx * nrx == links:
                    pair = (0, ntx)                # same TX stream, two different RX antennas
                else:
                    pair = (0, 1)
                print(f"  numTx={ntx} numRx={nrx} -> antenna pair used for the ratio: links {pair[0]} and {pair[1]}")
                rate = rate_of([fr[3] for fr in same], [fr[4] for fr in same])
                if rate:
                    keep_every = max(1, int(round(rate / target_rate)))
                    print(f"  packet rate ~{rate:.0f}/s -> keeping every {keep_every}th packet (target {target_rate:.0f}/s)")
                else:
                    print("  WARNING: cannot estimate the packet rate from timestamps; keeping every packet")
                rx = ci.get("rx", {})
                meta = dict(center_freq_mhz=float(rx.get("centerFreq") or 0), cbw_mhz=float(ci.get("CBW") or rx.get("CBW") or 0),
                            num_tx=ntx, num_rx=nrx, rate_hz=float(rate or 0), n_tones_total=nt)

            for fr in frames:
                if (fr[0], fr[1].size // fr[0], fr[2]) != best:
                    continue
                n_best += 1
                counter += 1
                if (counter - 1) % keep_every:
                    continue
                m = to_matrix(fr[1], best[0], best[1], mode)[:, list(pair)]
                mats.append(m.astype(np.complex64))
                ts_l.append(fr[3])
                td_l.append(fr[4])
            print(f"\r  parsed {min(i + n, len(offs))}/{len(offs)} frames, kept {len(mats)}", end="", flush=True)
        print()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if len(mats) < 50:
        raise RuntimeError(f"only {len(mats)} usable packets kept - recording too short or no CSI")

    data3 = np.stack(mats)                                     # (N, tones, 2)
    del mats
    ts = np.asarray(ts_l, dtype=float) if all(v is not None for v in ts_l) else None
    td = np.asarray(td_l, dtype=float) if all(v is not None for v in td_l) else None
    t, how, t0 = final_times(ts, td, len(data3), target_rate)
    order = np.argsort(t, kind="stable")
    t, data3 = t[order], data3[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t, data3 = t[keep], data3[keep]
    t = t - t[0]

    # drop the null / dead tones (they are zero on one of the antennas)
    mag = np.abs(data3).mean(axis=0)                           # (tones, 2)
    live = (mag > 1e-6 * mag.max()).all(axis=1)
    tones_kept = np.where(live)[0]
    data = data3[:, tones_kept, :].reshape(len(t), -1)         # interleave: tone0_A, tone0_B, tone1_A ...
    dur = t[-1] - t[0]
    print(f"  packets {len(t)} over {dur:.1f} s (~{len(t) / max(dur, 1e-9):.0f}/s), times from {how}")
    print(f"  tones kept {len(tones_kept)} of {best[0]} (null tones dropped: {best[0] - len(tones_kept)}) -> data {data.shape}")
    np.savez_compressed(out_path, data=data, t=t, t0_epoch=t0, pair=np.array(pair), tones_kept=tones_kept,
                        activity=np.array(activity), source=np.array(str(f)), **meta)
    return dict(packets=len(t), seconds=dur, rate=len(t) / max(dur, 1e-9), tones=len(tones_kept), how=how)


def read_label(folder):
    """Activity name from <folder>/activity.txt (first non-empty line), or None."""
    p = Path(folder) / "activity.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                return line.strip().replace(" ", "_").replace("/", "-")
    return None


def discover(folders, out, parent, pattern, activity_override):
    """List (csi file, output npz path, activity). Activity priority:
       1. activity.txt in the .csi file's own folder   2. --activity   3. --parent first sub-folder
       4. the folder you passed."""
    jobs = []
    for root in folders:
        root = Path(root).expanduser().resolve()
        for f in sorted(root.rglob(pattern)):
            rel = f.relative_to(root).parts
            activity = read_label(f.parent) or activity_override or (rel[0] if (parent and len(rel) > 1) else root.name)
            name = "_".join(list(rel[:-1]) + [f.stem]) if len(rel) > 1 else f"{root.name}_{f.stem}"
            jobs.append((f, out / f"{activity}__{name}.npz", activity))
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Convert PicoScenes .csi recordings to .npz files for the Doppler preprocessing")
    ap.add_argument("folders", nargs="+", help="activity folders (or parents) containing .csi files")
    ap.add_argument("--out", default="npz", help="output folder (default ./npz)")
    ap.add_argument("--rate", type=float, default=150.0, help="target packets/s; faster recordings are thinned (default 150)")
    ap.add_argument("--pair", default="", help="two CSI link numbers for the ratio, e.g. 0,2 (default: automatic)")
    ap.add_argument("--pattern", default="rx_*.csi", help="which files to convert (default rx_*.csi = PicoScenes recordings; skips copies like sample.csi)")
    ap.add_argument("--activity", default="", help="activity name for all files that have no activity.txt")
    ap.add_argument("--parent", action="store_true", help="the given folders are parents whose sub-folders are the activities")
    ap.add_argument("--force", action="store_true", help="overwrite existing .npz files")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    jobs = discover(args.folders, out, args.parent, args.pattern, args.activity)
    if not jobs:
        sys.exit(f"Error: no files matching '{args.pattern}' found in the given folders")
    print(f"{len(jobs)} .csi file(s) found -> {out}")
    unlabeled = sorted({str(f.parent) for f, _, a in jobs if a.startswith("experiment_")})
    if unlabeled:
        print("\nWARNING: these folders have no activity name yet (they will be labelled with the folder name):")
        for u in unlabeled:
            print(f"   {u}")
        print("   Label a folder with:  echo clapping > <folder>/activity.txt      (then run again)\n")

    rows = []
    for k, (f, o, activity) in enumerate(jobs, 1):
        print(f"\n[{k}/{len(jobs)}] {f}")
        if o.exists() and not args.force:
            print(f"  exists, skipped: {o.name} (use --force to redo)")
            continue
        t0 = time.time()
        try:
            r = convert_file(f, o, activity, args.rate, args.pair)
            rows.append((o.name, r))
            print(f"  saved {o.name}  ({time.time() - t0:.0f} s)")
        except Exception as e:  # noqa: BLE001
            print(f"  Error: {e}")
    print("\n==== summary ====")
    for name, r in rows:
        print(f"  [{name.split('__')[0]}] {name}: {r['packets']} packets, {r['seconds']:.1f} s, {r['rate']:.0f}/s, {r['tones']} tones, times: {r['how']}")
    if rows:
        print(f"\nNext: python3 preprocess_csi.py --input {out} --output {out.parent / 'output'}")


if __name__ == "__main__":
    main()