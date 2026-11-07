#!/usr/bin/env python3
"""
preprocess_csi.py - turns the converted CSI into a Doppler heatmap. It follows the cleaning stages
of the WiLight / AxSense batchprocessing.py pipeline, adapted to our PicoScenes recordings.

    python3 csi_tools/preprocess_csi.py --input ~/percom/npz --output ~/percom/output

Input : the .npz files written by csi_to_npz.py (keys data, t, ...)
Output: <output>/double_ratio/<name>.npz   cleaned double-ratio CSI (+ packet times)
        <output>/doppler/<name>.npz        Doppler spectrogram(s) + all parameters
        <output>/png/<name>.png            picture of the result for a quick look

Stages (the same as the original WiLight code, except the one marked "new"):
  1. single ratio   antenna A / antenna B per tone, with vote + interpolate + clip + Savitzky-Golay
  2. double ratio   ratio of neighbouring tones' ratios, same cleaning
  3. temporal clean flag and interpolate outlier packets per column
  new: resample     packets arrive at irregular times (ping traffic), so the cleaned series is
                    interpolated onto a uniform time grid (default 100 Hz) before the FFT
  4. Doppler        sliding Hann window, static part removed, FFT over time, power summed over tones
"""
import argparse
import gc
import math as mt
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.fftpack import fft, fftshift  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.signal import savgol_filter as _savgol  # noqa: E402
from scipy.signal.windows import hann  # noqa: E402

# ---- cleaning parameters (same values as the original WiLight code) --------------------------------------
THRESHOLD = 2
VOTE_PACKETS = 100
VOTE_FRACTION = 2 / 3
CLIP_MULT = 5.0
SG_WINDOW = 11
SG_POLY = 3
T_CLIP_MULT = 5.0
NOISE_LEVEL = -2          # log10 noise floor of the normalised spectrogram
SMOOTH_SIGMA_T = 0.5
SMOOTH_SIGMA_V = 0.5
V_LIGHT = 3e8


def ratio_stage(data_2d, threshold, vote_packets, vote_fraction, clip_mult, sg_window, sg_poly):
    """One ratio stage: (n_packets, N) complex -> (n_packets, N//2) complex (spatial cleaning)."""
    n_packets, N = data_2d.shape
    n_pair = N // 2
    check_upto = min(vote_packets, n_packets)
    x = np.arange(n_pair, dtype=float)

    num = data_2d[:, 0::2][:, :n_pair]
    den = data_2d[:, 1::2][:, :n_pair]
    nz = den != 0
    raw_all = np.where(nz, num / np.where(nz, den, 1 + 0j), np.nan + 0j)

    # A. vote on phase outliers over the first packets
    subset = raw_all[:check_upto]
    valid_m = np.isfinite(subset.real)
    phase = np.unwrap(np.angle(np.where(valid_m, subset, 1 + 0j)), axis=1)
    phase[~valid_m] = np.nan
    row_valid = valid_m.sum(axis=1)
    row_avg = np.where(valid_m, np.abs(phase), 0.0).sum(axis=1) / row_valid.clip(1)
    bad_m = valid_m & (np.abs(phase) > threshold * row_avg[:, None])
    bad_m[row_valid == 0] = False
    bad_count = bad_m.sum(axis=0)
    del subset, valid_m, phase, bad_m
    min_votes = int(np.ceil(vote_fraction * check_upto))
    confirmed_bad = np.where(bad_count >= min_votes)[0]

    # B. replace confirmed-bad columns by interpolation between good neighbours
    result = raw_all.copy()
    del raw_all
    result[:, confirmed_bad] = np.nan
    if len(confirmed_bad) > 0:
        valid_idx = np.where(~np.isin(np.arange(n_pair), confirmed_bad))[0]
        if len(valid_idx) > 0:
            for bad_j in confirmed_bad:
                left = valid_idx[valid_idx < bad_j]
                right = valid_idx[valid_idx > bad_j]
                if len(left) == 0:
                    result[:, bad_j] = result[:, right[0]]
                elif len(right) == 0:
                    result[:, bad_j] = result[:, left[-1]]
                else:
                    lx, rx = int(left[-1]), int(right[0])
                    alpha = float(bad_j - lx) / (rx - lx)
                    result[:, bad_j] = (1.0 - alpha) * result[:, lx] + alpha * result[:, rx]
    still_nan = ~np.isfinite(result.real).all(axis=1)
    for p in np.where(still_nan)[0]:
        z = result[p]
        valid = np.isfinite(z.real)
        if valid.sum() == 0:
            result[p] = 0
        elif not valid.all():
            result[p] = np.interp(x, x[valid], z.real[valid]) + 1j * np.interp(x, x[valid], z.imag[valid])

    # C. clip magnitude outliers per packet
    mags = np.abs(result)
    med = np.median(mags, axis=1, keepdims=True)
    cap = np.where(med != 0, clip_mult * med, np.inf)
    bad_clip = mags > cap
    if bad_clip.any():
        result = result * np.where(bad_clip, cap / np.maximum(mags, 1e-300), 1.0)
    del mags, med, cap, bad_clip

    # D. Savitzky-Golay smoothing across tones
    w = sg_window if sg_window % 2 == 1 else sg_window - 1
    w = min(w, n_pair if n_pair % 2 == 1 else n_pair - 1)
    if w > sg_poly and w >= 3:
        result = (_savgol(result.real, w, sg_poly, axis=1).astype(float)
                  + 1j * _savgol(result.imag, w, sg_poly, axis=1).astype(float))
    return result, confirmed_bad, bad_count, min_votes


def temporal_clean(dr, t_clip_mult):
    """Flag packets with |value| > t_clip_mult * column median, then interpolate over them."""
    n_packets, _ = dr.shape
    t = np.arange(n_packets, dtype=float)
    out = dr.copy()
    mags = np.abs(out)
    col_med = np.median(mags, axis=0, keepdims=True)
    col_med = np.where(col_med == 0, 1.0, col_med)
    bad_mask = mags > t_clip_mult * col_med
    del mags
    out[bad_mask] = np.nan
    for j in np.where(bad_mask.any(axis=0))[0]:
        col = out[:, j]
        valid = np.isfinite(col.real)
        if valid.sum() == 0:
            out[:, j] = 0
        elif not valid.all():
            out[:, j] = np.interp(t, t[valid], col.real[valid]) + 1j * np.interp(t, t[valid], col.imag[valid])
    return out, int(bad_mask.sum())


def resample_uniform(t, X, fs):
    """NEW: interpolate (N, K) complex series from irregular packet times onto a uniform grid."""
    tg = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.empty((len(tg), X.shape[1]), dtype=np.complex64)
    for k in range(X.shape[1]):
        out[:, k] = np.interp(tg, t, X[:, k].real) + 1j * np.interp(tg, t, X[:, k].imag)
    return tg, out


def compute_doppler_raw(csi_complex, num_symbols, sliding, n_fft):
    """Raw Doppler power, shape (n_windows, n_fft)."""
    n_packets = csi_complex.shape[0]
    hann_win = np.expand_dims(hann(num_symbols), axis=-1)
    csi_clean = np.nan_to_num(csi_complex)
    starts = list(range(0, n_packets - num_symbols, sliding))
    profiles = np.empty((len(starts), n_fft))
    for k, i in enumerate(starts):
        win = csi_clean[i:i + num_symbols, :]
        win = win - win.mean(axis=0, keepdims=True)          # remove the static part
        spec = fftshift(fft(win * hann_win, n=n_fft, axis=0), axes=0)
        profiles[k] = (np.abs(spec * np.conj(spec))).sum(axis=1)
    return profiles, np.array(starts)


def save_png(path, title, t_win, freqs, doppler, db, motion, fmax):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.4]})
    ext = [t_win[0], t_win[-1], freqs[0], freqs[-1]]
    im0 = ax[0].imshow(np.log10(doppler).T, aspect="auto", origin="lower", extent=ext, cmap="jet",
                       vmin=NOISE_LEVEL, vmax=0)
    ax[0].set_ylim(-fmax, fmax)
    ax[0].set_ylabel("Doppler (Hz)")
    ax[0].set_title("Doppler spectrogram, each window normalised (GitHub style, log10)", fontsize=10, loc="left")
    fig.colorbar(im0, cax=ax[0].inset_axes([1.012, 0, 0.012, 1]))
    im1 = ax[1].imshow(db.T, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=-35, vmax=0)
    ax[1].set_ylim(-fmax, fmax)
    ax[1].set_ylabel("Doppler (Hz)")
    ax[1].set_title("Doppler spectrogram, absolute scale (dB relative to the recording's peak)", fontsize=10, loc="left")
    fig.colorbar(im1, cax=ax[1].inset_axes([1.012, 0, 0.012, 1]))
    ax[2].plot(t_win, motion, color="tab:red", lw=1.1)
    ax[2].fill_between(t_win, 0, motion, color="tab:red", alpha=0.25)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("Motion")
    ax[2].set_xlabel("Time (s) from the first packet")
    ax[2].grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.07, hspace=0.28)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process_one(path, out_base, args):
    name = path.stem
    d = np.load(path, allow_pickle=False)
    data = d["data"]
    t = d["t"].astype(float)
    activity = str(d["activity"]) if "activity" in d.files else "Unknown"
    fc = float(d["center_freq_mhz"]) * 1e6 if "center_freq_mhz" in d.files and float(d["center_freq_mhz"]) > 0 else args.fc
    tm = {}
    t_all = time.time()

    if args.remove_idx:
        keep = [i for i in range(data.shape[1]) if i not in set(args.remove_idx)]
        data = data[:, keep]
    if data.shape[1] % 4:
        data = data[:, : (data.shape[1] // 4) * 4]             # two ratio stages need a multiple of 4
    print(f"  data {data.shape}, {len(t)} packets over {t[-1] - t[0]:.1f} s, fc={fc / 1e9:.3f} GHz")

    s = time.time()
    single, _, _, _ = ratio_stage(data, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    del data
    gc.collect()
    tm["stage1_single_ratio"] = time.time() - s

    s = time.time()
    double, _, _, _ = ratio_stage(single, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    del single
    gc.collect()
    tm["stage2_double_ratio"] = time.time() - s

    s = time.time()
    double, n_flagged = temporal_clean(double, T_CLIP_MULT)
    tm["stage3_temporal"] = time.time() - s

    (out_base / "double_ratio").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_base / "double_ratio" / f"{name}.npz", double_ratio=double, t=t)

    s = time.time()
    tg, X = resample_uniform(t, double, args.fs)
    del double
    gc.collect()
    num_symbols = int(round(args.win * args.fs)) + 1
    sliding = args.sliding
    if len(tg) <= num_symbols + 2:
        raise RuntimeError("recording too short for the Doppler window")
    raw_power, starts = compute_doppler_raw(X, num_symbols, sliding, args.n_fft)
    del X
    gc.collect()

    noise_floor = mt.pow(10, NOISE_LEVEL)
    win_max = raw_power.max(axis=1, keepdims=True)
    win_max = np.where(win_max == 0, 1.0, win_max)
    doppler = np.clip(raw_power / win_max, noise_floor, 1.0)
    if SMOOTH_SIGMA_T > 0 or SMOOTH_SIGMA_V > 0:
        doppler = gaussian_filter(doppler, sigma=(SMOOTH_SIGMA_T, SMOOTH_SIGMA_V))
        doppler = np.clip(doppler, noise_floor, 1.0)
    ref = np.percentile(raw_power, 99.5) + 1e-30
    db = np.clip(10 * np.log10(raw_power / ref + 1e-9), -35, 0)
    freqs = fftshift(np.fft.fftfreq(args.n_fft, 1.0 / args.fs))
    t_win = tg[0] + (starts + num_symbols / 2) / args.fs
    hi = np.abs(freqs) > 1.5
    energy = raw_power[:, hi].sum(axis=1)
    energy = energy / (np.percentile(energy, 99.5) + 1e-30)
    k = max(1, int(round(0.5 / max(sliding / args.fs, 1e-9))))
    motion = np.clip(np.convolve(energy, np.ones(k) / k, mode="same"), 0, 1)
    tm["stage4_doppler"] = time.time() - s

    Tc = 1.0 / args.fs
    delta_v = round(V_LIGHT / (Tc * fc * num_symbols), 4)
    (out_base / "doppler").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_base / "doppler" / f"{name}.npz", doppler=doppler, doppler_db=db, raw_power=raw_power,
                        motion=motion, t_windows=t_win, freqs_hz=freqs, delta_v=delta_v, Tc=Tc, fs=args.fs,
                        num_symbols=num_symbols, sliding=sliding, n_fft=args.n_fft, fc=fc,
                        source_name=name, activity=activity)
    if not args.no_png:
        (out_base / "png").mkdir(parents=True, exist_ok=True)
        save_png(out_base / "png" / f"{name}.png", f"{activity}  |  {name}", t_win, freqs, doppler, db, motion, args.fmax)

    tm["total"] = time.time() - t_all
    print("  " + ", ".join(f"{k} {v:.1f}s" for k, v in tm.items()) + f"  | temporal outliers fixed: {n_flagged}")
    print(f"  Doppler windows {doppler.shape}, window {num_symbols} samples = {args.win:.2f} s at {args.fs:.0f} Hz, "
          f"delta_v {delta_v} m/s")
    return tm


def main():
    ap = argparse.ArgumentParser(description="Preprocess CSI .npz files (from csi_to_npz.py) into Doppler spectrograms")
    ap.add_argument("--input", required=True, help="folder with .npz files from csi_to_npz.py")
    ap.add_argument("--output", default="output", help="output folder (default ./output)")
    ap.add_argument("--fs", type=float, default=100.0, help="uniform resampling rate in Hz (default 100)")
    ap.add_argument("--win", type=float, default=0.5, help="Doppler window in seconds (default 0.5)")
    ap.add_argument("--sliding", type=int, default=1, help="window step in samples (default 1)")
    ap.add_argument("--n-fft", type=int, default=100, help="FFT size / Doppler bins (default 100)")
    ap.add_argument("--fmax", type=float, default=30.0, help="max |Doppler| shown in the PNG, Hz (default 30)")
    ap.add_argument("--fc", type=float, default=5.775e9, help="carrier frequency if the .npz has none (Hz)")
    ap.add_argument("--remove-idx", type=int, nargs="*", default=[], help="extra column indices to delete")
    ap.add_argument("--no-png", action="store_true", help="do not write PNG previews")
    ap.add_argument("--test", action="store_true", help="process only the first file")
    args = ap.parse_args()

    inp = Path(args.input).expanduser()
    out = Path(args.output).expanduser()
    files = sorted(inp.glob("*.npz"))
    if not files:
        raise SystemExit(f"Error: no .npz files in {inp} (run csi_to_npz.py first)")
    if args.test:
        files = files[:1]
    print(f"{len(files)} file(s) -> {out}")
    ok = bad = 0
    for k, f in enumerate(files, 1):
        print(f"\n[{k}/{len(files)}] {f.name}")
        try:
            process_one(f, out, args)
            ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  Error: {e}")
            bad += 1
    print(f"\nDone: {ok} processed, {bad} errors.  Pictures: {out / 'png'}")


if __name__ == "__main__":
    main()
