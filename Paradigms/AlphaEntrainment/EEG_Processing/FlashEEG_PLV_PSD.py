#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashEEG_PLV_PSD.py
===================
Measures Phase-Locking Value (PLV) and Power Spectral Density (PSD) of
  (a) the flashing stimulus signal reconstructed from the control.py stimulus log, and
  (b) the EEG recorded during that time,

for every epoch (frequency block) defined in the stimulus log.

The analysis is epoch-aware: the stimulus log produced by control.py contains
multiple consecutive blocks, each at a different frequency (8-13 Hz).  This
script handles all of them, producing:

  Per-epoch outputs
  -----------------
  * PSD of the flash stimulus (should show a sharp peak at the epoch frequency)
  * PSD of the EEG during the same window (alpha / flicker response)
  * PLV between flash stimulus and EEG for that epoch
  * Sliding-window PLV time series within each epoch

  Global summary
  --------------
  * Grand-average PLV vs. epoch frequency (entrainment curve)
  * Summary CSV

Modelled on PhaseLocking.py in the same repository.  Reuses identical helper
functions (narrowband_filter, hilbert_phase, circular_plv, sliding_plv) so the
two scripts stay consistent.

Usage
-----
Edit the CONFIG section below, then run:
    python FlashEEG_PLV_PSD.py

Requirements: mne, numpy, scipy, pandas, matplotlib
"""

import os
import warnings

# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# pyrefly: ignore [missing-import]
import mne
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from scipy.signal import butter, filtfilt, hilbert, welch

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ============================================================================
# CONFIG - edit these paths and parameters for your session
# ============================================================================

EDF_PATH = (
    r"C:\Users\saiik\Downloads\miriam expt edf\ARYA_SUB10\arya control"
    r"\SUB10~ ARYA_95a157e5-6c2b-48f6-bd54-848757e083e0.edf"
)
STIMULUS_CSV_PATH = (
    r"C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research"
    r"\Paradigms\AlphaEntrainment\data\stimulus_log.csv"
)

# EEG channel configuration
REFERENCE_CHANNEL    = "Cz"          # bipolar re-reference (set "" or None to skip)
REREFERENCE_CHANNELS = ["O1", "O2"]  # channels to re-derive as (ch - ref)
EEG_CHANNELS         = ["O1", "O2"]  # channels to analyse (None = all non-trigger)

# Trigger channel in the EDF
DC7_CHANNEL_NAME  = "DC7"
DC7_THRESHOLD     = 0.005   # absolute deviation from baseline; None = adaptive
MIN_TRIGGER_ISI_S = 1.0     # refractory period (s) between successive pulses

# Channels to exclude when EEG_CHANNELS is None
EXCLUDE_CHANNEL_PREFIXES = ("DC", "EDF Annotations", "STATUS", "TRIG", "EVENT")

# Clock alignment tolerance
TRIGGER_MATCH_WARN_TOL_S = 0.05   # warn if max alignment residual > this (s)
MERGE_ASOF_TOL_S         = 0.05   # max gap when mapping EEG time -> CSV frame

# Filtering
NARROWBAND_HALFWIDTH_HZ = 1.0  # +/-1 Hz around the epoch flicker frequency
FILTER_ORDER            = 4

# Sliding-window PLV within each epoch
SLIDING_PLV_WIN_S  = 2.0   # window length (s)
SLIDING_PLV_STEP_S = 0.5   # hop size (s)

# PSD
WELCH_NPERSEG_S = 4.0      # Welch segment length (s); set to None for auto

# Boundary exclusion around epoch transitions (filter ringing)
BOUNDARY_EXCLUDE_S = 0.5

# Output
OUTPUT_DIR = "flash_eeg_plv_psd_results"

# ============================================================================
# 1. STIMULUS LOG HELPERS
# ============================================================================

def load_stimulus_log(csv_path):
    """
    Load the control.py stimulus log and return:
      df            - raw DataFrame (sorted by global_time_s)
      epoch_table   - one row per epoch block, with columns:
                        block_num, frequency_hz, t_start_csv, t_end_csv,
                        n_frames, trigger_times_csv
      all_trigger_times_csv - all trigger-flagged global_time_s values
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("global_time_s").reset_index(drop=True)
    df["state_bin"] = (df["stimulus_state"] == "ON").astype(float)

    # Build per-epoch summary from the block_num column written by control.py
    epoch_rows = []
    for blk_num, grp in df.groupby("block_num"):
        freq   = float(grp["frequency_hz"].iloc[0])
        t0     = float(grp["global_time_s"].iloc[0])
        t1     = float(grp["global_time_s"].iloc[-1])
        n_f    = len(grp)
        trigs  = grp.loc[grp["trigger_sent"] == 1, "global_time_s"].values
        epoch_rows.append(dict(block_num=blk_num, frequency_hz=freq,
                               t_start_csv=t0, t_end_csv=t1,
                               n_frames=n_f, trigger_times_csv=trigs))
    epoch_table = pd.DataFrame(epoch_rows).sort_values("block_num").reset_index(drop=True)

    all_trig = df.loc[df["trigger_sent"] == 1, "global_time_s"].values
    all_trig = np.sort(all_trig)

    print(f"[stimulus] {len(df)} frames across {len(epoch_table)} epochs")
    for _, row in epoch_table.iterrows():
        print(f"  block {int(row.block_num):>2}  {row.frequency_hz:.2f} Hz  "
              f"{row.t_start_csv:.2f}s -> {row.t_end_csv:.2f}s  "
              f"({row.n_frames} frames, {len(row.trigger_times_csv)} triggers)")
    print(f"[stimulus] {len(all_trig)} total triggers, "
          f"span {all_trig[0]:.3f}s -> {all_trig[-1]:.3f}s")

    return df, epoch_table, all_trig


def reconstruct_flash_wave(csv_df, query_times_csv_clock, tol_s=MERGE_ASOF_TOL_S):
    """
    Step-interpolate the ON/OFF square wave at arbitrary query times
    (expressed in the CSV global_time_s clock).
    Returns NaN where no logged frame covers the query time.
    """
    left  = pd.DataFrame({"t": query_times_csv_clock})
    right = csv_df[["global_time_s", "state_bin"]].rename(
        columns={"global_time_s": "t"})
    merged = pd.merge_asof(
        left.sort_values("t"), right.sort_values("t"),
        on="t", direction="backward", tolerance=tol_s,
    )
    merged = merged.set_index(left.sort_values("t").index).sort_index()
    return merged["state_bin"].values


# ============================================================================
# 2. EDF / RE-REFERENCING / TRIGGER CHANNEL
# ============================================================================

def load_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(f"[edf] loaded {edf_path}")
    print(f"[edf] sfreq={raw.info['sfreq']:.3f} Hz, "
          f"duration={raw.times[-1]:.1f}s, {len(raw.ch_names)} channels")
    return raw


def rereference_to_channel(raw, channels, ref_channel):
    """Re-derive `channels` in-place as bipolar (channel - ref_channel)."""
    raw = raw.copy()
    if not ref_channel:
        print("[INFO] Re-referencing disabled. Using hardware reference.")
        return raw
    present = [ch for ch in channels if ch in raw.ch_names]
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        print(f"[WARN] Re-reference channels not found, skipping: {missing}")
    if ref_channel not in raw.ch_names:
        raise ValueError(f"Reference channel '{ref_channel}' not found in EDF.")
    if present:
        tmp = [f"{ch}__tmp" for ch in present]
        raw = mne.set_bipolar_reference(
            raw, anode=present, cathode=[ref_channel] * len(present),
            ch_name=tmp, drop_refs=True, verbose=False,
        )
        raw.rename_channels(dict(zip(tmp, present)))
        print(f"[INFO] Re-referenced to {ref_channel}: {present}")
    return raw


def detect_dc7_triggers(raw,
                         dc7_name=DC7_CHANNEL_NAME,
                         threshold=DC7_THRESHOLD,
                         min_isi_s=MIN_TRIGGER_ISI_S):
    """Detect TTL pulse onsets via rising-edge thresholding with a refractory period."""
    if dc7_name not in raw.ch_names:
        matches = [c for c in raw.ch_names if dc7_name.lower() in c.lower()]
        raise ValueError(
            f"Trigger channel '{dc7_name}' not found. "
            f"Closest matches: {matches}. "
            f"Available: {raw.ch_names}"
        )
    sig   = raw.get_data(picks=[dc7_name])[0]
    sfreq = raw.info["sfreq"]

    median = np.median(sig)
    dev    = np.abs(sig - median)
    if threshold is None:
        mad       = np.median(np.abs(dev - np.median(dev)))
        sigma     = 1.4826 * mad
        threshold = max(10 * sigma, 0.005)

    above  = dev > threshold
    rising = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    kept, last_t = [], -np.inf
    for idx in rising:
        t = idx / sfreq
        if t - last_t >= min_isi_s:
            kept.append(t)
            last_t = t
    trigger_times = np.array(kept)

    print(f"[edf] DC7 baseline={median:.5g}, threshold={threshold:.5g}; "
          f"{len(trigger_times)} pulses detected")
    if len(trigger_times) > 1:
        isis = np.diff(trigger_times)
        print(f"[edf] trigger ISIs: min={isis.min():.3f}s  "
              f"max={isis.max():.3f}s  median={np.median(isis):.3f}s")
    return trigger_times, sig, sfreq


# ============================================================================
# 3. CLOCK ALIGNMENT  (CSV clock <-> EDF clock)
# ============================================================================

def align_clocks(csv_trigger_times, eeg_trigger_times,
                  warn_tol_s=TRIGGER_MATCH_WARN_TOL_S):
    """
    Fit  t_edf = slope * t_csv + intercept  (accounts for clock drift).
    Returns slope, intercept, matched_eeg_times, residuals.
    """
    if len(eeg_trigger_times) < 3:
        raise ValueError(
            f"Only {len(eeg_trigger_times)} EDF triggers detected. "
            "Check DC7_CHANNEL_NAME / DC7_THRESHOLD."
        )

    best_inliers, best_count = None, 0
    for csv_t0 in csv_trigger_times[:5]:
        for eeg_t0 in eeg_trigger_times[:10]:
            offset = eeg_t0 - csv_t0
            cm, em = [], []
            for ct in csv_trigger_times:
                diffs  = np.abs(eeg_trigger_times - (ct + offset))
                mi     = np.argmin(diffs)
                if diffs[mi] < 0.2:
                    cm.append(ct)
                    em.append(eeg_trigger_times[mi])
            if len(cm) > best_count:
                best_count   = len(cm)
                best_inliers = (np.array(cm), np.array(em))

    if best_inliers is None or len(best_inliers[0]) < 3:
        raise ValueError("Could not align CSV triggers to EDF triggers.")

    matched_csv, matched_eeg = best_inliers
    slope, intercept = np.polyfit(matched_csv, matched_eeg, 1)
    fitted    = slope * matched_csv + intercept
    residuals = matched_eeg - fitted
    max_resid_ms = np.max(np.abs(residuals)) * 1000

    print(f"[align] matched {len(matched_csv)}/{len(csv_trigger_times)} CSV triggers "
          f"({len(eeg_trigger_times)} EDF triggers)")
    print(f"[align] t_edf = {slope:.8f} * t_csv + {intercept:.4f}  "
          f"(clock drift {(slope-1)*1e6:.1f} ppm)")
    print(f"[align] max residual: {max_resid_ms:.2f} ms")
    if max_resid_ms / 1000 > warn_tol_s:
        print(f"[align][WARNING] residual > {warn_tol_s*1000:.0f} ms - check alignment!")

    extra = len(eeg_trigger_times) - len(matched_eeg)
    if extra > 0:
        print(f"[align] {extra} extra EDF pulse(s) not matched "
              "(likely post-flash fixation / eyes-closed / end triggers)")

    return slope, intercept, matched_eeg, residuals


# ============================================================================
# 4. SIGNAL PROCESSING PRIMITIVES
# ============================================================================

def narrowband_filter(x, sfreq, f0,
                       half_bw=NARROWBAND_HALFWIDTH_HZ,
                       order=FILTER_ORDER):
    nyq  = sfreq / 2.0
    low  = max(f0 - half_bw, 0.1) / nyq
    high = min(f0 + half_bw, nyq * 0.99) / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def hilbert_phase(x):
    return np.angle(hilbert(x))


def circular_plv(phase_a, phase_b, mask=None):
    diff = phase_a - phase_b
    if mask is not None:
        diff = diff[mask]
    return float(np.abs(np.mean(np.exp(1j * diff))))


def sliding_plv(phase_a, phase_b, sfreq, valid_mask,
                 win_s=SLIDING_PLV_WIN_S, step_s=SLIDING_PLV_STEP_S):
    """
    Compute PLV in a sliding window.
    Returns (center_times_s, plv_values).
    """
    win  = int(win_s  * sfreq)
    step = int(step_s * sfreq)
    n    = len(phase_a)
    centers_t, plv_vals = [], []
    for start in range(0, n - win, step):
        sl = slice(start, start + win)
        if valid_mask[sl].mean() < 0.9:
            continue
        plv_vals.append(circular_plv(phase_a[sl], phase_b[sl]))
        centers_t.append((start + win / 2) / sfreq)
    return np.array(centers_t), np.array(plv_vals)


def compute_psd(signal, sfreq, nperseg_s=WELCH_NPERSEG_S):
    """Welch PSD. Returns (freqs, power_density)."""
    nperseg = (int(nperseg_s * sfreq) if nperseg_s is not None
               else min(256, len(signal)))
    nperseg = min(nperseg, len(signal))
    f, pxx  = welch(signal, fs=sfreq, nperseg=nperseg)
    return f, pxx


def snr_at_peak(f, pxx, target_hz, bw_noise=2.0, guard=0.5):
    """
    Spectral SNR = peak power / mean power in surrounding noise band.
    `guard` Hz around the target is excluded from the noise estimate.
    """
    idx  = np.argmin(np.abs(f - target_hz))
    band = (f > target_hz - bw_noise) & (f < target_hz + bw_noise) \
           & (np.abs(f - target_hz) > guard)
    if band.sum() == 0:
        return np.nan
    return float(pxx[idx] / np.mean(pxx[band]))


# ============================================================================
# 5. PER-EPOCH ANALYSIS
# ============================================================================

def analyze_epoch(epoch_info, ch_name, eeg, eeg_times,
                   csv_df, sfreq, slope, intercept, out_dir):
    """
    Full PLV + PSD analysis for one (epoch, channel) pair.

    Parameters
    ----------
    epoch_info : row from epoch_table  (block_num, frequency_hz, t_start_csv,
                 t_end_csv, trigger_times_csv)
    ch_name    : EEG channel label
    eeg        : full EEG time series (uV or V) - entire recording
    eeg_times  : corresponding time axis in EDF seconds
    csv_df     : full stimulus log DataFrame
    sfreq      : EEG sampling rate (Hz)
    slope, intercept : clock mapping  t_edf = slope * t_csv + intercept
    out_dir    : directory for figure files
    """
    blk_num  = int(epoch_info["block_num"])
    freq     = float(epoch_info["frequency_hz"])
    t0_csv   = float(epoch_info["t_start_csv"])
    t1_csv   = float(epoch_info["t_end_csv"])

    # Convert epoch boundaries to EDF time
    t0_edf = slope * t0_csv + intercept
    t1_edf = slope * t1_csv + intercept

    # Mask samples inside this epoch
    epoch_mask = (eeg_times >= t0_edf) & (eeg_times <= t1_edf)
    if epoch_mask.sum() < sfreq * 2:
        print(f"  [epoch {blk_num} | {ch_name}] < 2 s of EEG - skipping")
        return None

    eeg_epoch  = eeg[epoch_mask]
    t_epoch    = eeg_times[epoch_mask]

    # Reconstruct flash wave at EEG sample times
    csv_query  = (t_epoch - intercept) / slope
    flash_wave = reconstruct_flash_wave(csv_df, csv_query)
    valid_mask = ~np.isnan(flash_wave)

    # Exclude samples near epoch transition boundaries (filter ringing)
    boundary_t_edf = np.array([t0_edf, t1_edf])
    for bt in boundary_t_edf:
        valid_mask &= np.abs(t_epoch - bt) > BOUNDARY_EXCLUDE_S

    if valid_mask.sum() < sfreq * 2:
        print(f"  [epoch {blk_num} | {ch_name}] < 2 s valid after boundary exclusion - skipping")
        return None

    flash_filled = np.where(np.isnan(flash_wave), 0.5, flash_wave)

    # -- Narrowband filtering & Hilbert phase ---------------------------------
    eeg_nb   = narrowband_filter(eeg_epoch, sfreq, freq)
    flash_nb = narrowband_filter(flash_filled, sfreq, freq)

    eeg_phase   = hilbert_phase(eeg_nb)
    flash_phase = hilbert_phase(flash_nb)

    # -- Overall PLV ----------------------------------------------------------
    overall_plv = circular_plv(flash_phase, eeg_phase, mask=valid_mask)

    mean_lag_rad = np.angle(
        np.mean(np.exp(1j * (flash_phase[valid_mask] - eeg_phase[valid_mask])))
    )
    mean_lag_ms = (mean_lag_rad / (2 * np.pi * freq)) * 1000

    # -- Sliding-window PLV ---------------------------------------------------
    slid_t, slid_plv = sliding_plv(flash_phase, eeg_phase, sfreq, valid_mask)

    # -- PSD of flash stimulus (valid samples only) ---------------------------
    f_flash, pxx_flash = compute_psd(flash_filled[valid_mask], sfreq)
    snr_flash_f0  = snr_at_peak(f_flash, pxx_flash, freq)
    snr_flash_2f0 = snr_at_peak(f_flash, pxx_flash, 2 * freq)

    # -- PSD of EEG (valid samples only) -------------------------------------
    f_eeg, pxx_eeg = compute_psd(eeg_epoch[valid_mask], sfreq)
    snr_eeg_f0     = snr_at_peak(f_eeg, pxx_eeg, freq)
    snr_eeg_2f0    = snr_at_peak(f_eeg, pxx_eeg, 2 * freq)

    dur_s = float(valid_mask.sum()) / sfreq

    # -- Figure ---------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Block {blk_num}  |  {freq:.2f} Hz  |  {ch_name}  |  "
        f"PLV = {overall_plv:.3f}  lag = {mean_lag_ms:+.1f} ms  "
        f"({dur_s:.1f} s valid)",
        fontsize=12, fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    xlim = min(30, 4 * freq)

    # [0,0] Flash PSD
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(f_flash, pxx_flash, color="#5b8cff", linewidth=1.5, label="flash stimulus")
    ax.axvline(freq,     color="red",    linestyle="--", linewidth=1.2,
               label=f"f0 = {freq:.2f} Hz")
    ax.axvline(2 * freq, color="orange", linestyle=":",  linewidth=1.0,
               label=f"2f0 = {2*freq:.2f} Hz")
    ax.set_xlim(0, xlim)
    ax.set_title(f"Flash PSD  (SNR@f0={snr_flash_f0:.2f}, SNR@2f0={snr_flash_2f0:.2f})")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Power / Hz")
    ax.legend(fontsize=7)

    # [0,1] EEG PSD
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(f_eeg, pxx_eeg, color="#ff6b6b", linewidth=1.5, label="EEG")
    ax.axvline(freq,     color="red",    linestyle="--", linewidth=1.2,
               label=f"f0 = {freq:.2f} Hz")
    ax.axvline(2 * freq, color="orange", linestyle=":",  linewidth=1.0,
               label=f"2f0 = {2*freq:.2f} Hz")
    ax.set_xlim(0, xlim)
    ax.set_title(f"EEG PSD  (SNR@f0={snr_eeg_f0:.2f}, SNR@2f0={snr_eeg_2f0:.2f})")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Power / Hz")
    ax.legend(fontsize=7)

    # [0,2] Overlaid PSD (normalised to max for visual comparison)
    ax = fig.add_subplot(gs[0, 2])
    norm_f = pxx_flash / pxx_flash.max()
    norm_e = pxx_eeg   / pxx_eeg.max()
    ax.semilogy(f_flash, norm_f, color="#5b8cff", linewidth=1.5,
                label="flash (norm)", alpha=0.85)
    ax.semilogy(f_eeg,   norm_e, color="#ff6b6b", linewidth=1.5,
                label="EEG (norm)",   alpha=0.85)
    ax.axvline(freq, color="red", linestyle="--", linewidth=1.2,
               label=f"f0 = {freq:.2f} Hz")
    ax.set_xlim(0, xlim)
    ax.set_title("Overlaid PSD (normalised)")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Relative power")
    ax.legend(fontsize=7)

    # [1,0] Phase difference histogram
    ax = fig.add_subplot(gs[1, 0])
    phase_diff = (flash_phase[valid_mask] - eeg_phase[valid_mask]) % (2 * np.pi)
    ax.hist(phase_diff, bins=36, color="#a29bfe", edgecolor="white", linewidth=0.4)
    ax.axvline(np.pi, color="k", linestyle="--", linewidth=1)
    ax.set_title(f"Phase-difference histogram  (PLV = {overall_plv:.3f})")
    ax.set_xlabel("Phase diff (rad)")
    ax.set_ylabel("Count")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    # [1,1] Sliding-window PLV
    ax = fig.add_subplot(gs[1, 1])
    if len(slid_t) > 0:
        ax.plot(slid_t, slid_plv, color="#00b894", linewidth=1.8,
                marker="o", markersize=3, label="PLV")
        ax.axhline(overall_plv, color="k", linestyle="--", linewidth=1,
                   label=f"overall = {overall_plv:.3f}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Not enough data\nfor sliding PLV",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"Sliding PLV  (win={SLIDING_PLV_WIN_S}s, step={SLIDING_PLV_STEP_S}s)")
    ax.set_xlabel("Time within epoch (s)")
    ax.set_ylabel("PLV")

    # [1,2] Polar plot - mean resultant vector
    ax = fig.add_subplot(gs[1, 2], projection="polar")
    angles = (flash_phase[valid_mask] - eeg_phase[valid_mask])
    # subsample to avoid overloading the scatter
    subsample = max(1, len(angles) // 5000)
    ax.scatter(angles[::subsample] % (2 * np.pi),
               np.ones(len(angles[::subsample])),
               alpha=0.05, s=2, color="#6c5ce7")
    mean_vec = np.mean(np.exp(1j * angles))
    ax.annotate("", xy=(np.angle(mean_vec), np.abs(mean_vec)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax.set_title(f"Mean resultant vector\n|R| = PLV = {overall_plv:.3f}", pad=15)
    ax.set_ylim(0, 1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(out_dir,
                             f"block{blk_num:02d}_{freq:.2f}Hz_{ch_name}.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    print(f"  [epoch {blk_num} | {ch_name}] {freq:.2f} Hz  "
          f"PLV={overall_plv:.3f}  lag={mean_lag_ms:+.1f} ms  "
          f"SNR_EEG@f0={snr_eeg_f0:.2f}  -> {fig_path}")

    return dict(
        block_num=blk_num,
        channel=ch_name,
        frequency_hz=freq,
        duration_valid_s=round(dur_s, 2),
        overall_plv=round(overall_plv, 4),
        mean_lag_ms=round(mean_lag_ms, 2),
        mean_sliding_plv=round(float(np.mean(slid_plv)), 4) if len(slid_plv) > 0 else np.nan,
        snr_flash_f0=round(snr_flash_f0, 3),
        snr_flash_2f0=round(snr_flash_2f0, 3),
        snr_eeg_f0=round(snr_eeg_f0, 3),
        snr_eeg_2f0=round(snr_eeg_2f0, 3),
        figure=fig_path,
    )


# ============================================================================
# 6. SUMMARY PLOTS
# ============================================================================

def plot_summary(results_df, out_dir):
    """
    Grand-average PLV and EEG SNR as a function of flicker frequency.
    One line per EEG channel, overlaid.
    """
    channels = results_df["channel"].unique()
    colors   = plt.cm.tab10(np.linspace(0, 1, len(channels)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Summary: PLV & PSD SNR vs Flicker Frequency",
                 fontsize=13, fontweight="bold")

    # Panel 0: PLV vs frequency
    ax = axes[0]
    for ch, col in zip(channels, colors):
        sub = results_df[results_df["channel"] == ch].sort_values("frequency_hz")
        ax.plot(sub["frequency_hz"], sub["overall_plv"],
                marker="o", label=ch, color=col)
    ax.set_xlabel("Flicker frequency (Hz)")
    ax.set_ylabel("Overall PLV")
    ax.set_title("PLV vs Flicker Frequency")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 1: EEG SNR @ f0 vs frequency
    ax = axes[1]
    for ch, col in zip(channels, colors):
        sub = results_df[results_df["channel"] == ch].sort_values("frequency_hz")
        ax.plot(sub["frequency_hz"], sub["snr_eeg_f0"],
                marker="s", label=ch, color=col)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="SNR=1 (flat)")
    ax.set_xlabel("Flicker frequency (Hz)")
    ax.set_ylabel("Spectral SNR at f0")
    ax.set_title("EEG Spectral SNR @ f0")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Flash PSD SNR @ f0 (sanity check - should be high)
    ax = axes[2]
    for ch, col in zip(channels, colors):
        sub = results_df[results_df["channel"] == ch].sort_values("frequency_hz")
        ax.plot(sub["frequency_hz"], sub["snr_flash_f0"],
                marker="^", label=ch, color=col)
    ax.set_xlabel("Flicker frequency (Hz)")
    ax.set_ylabel("Spectral SNR at f0")
    ax.set_title("Flash Stimulus SNR @ f0 (sanity check)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "summary_plv_psd.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[summary] figure -> {path}")


# ============================================================================
# 7. CHANNEL SELECTION
# ============================================================================

def pick_eeg_channels(raw):
    if EEG_CHANNELS is not None:
        missing = [c for c in EEG_CHANNELS if c not in raw.ch_names]
        if missing:
            print(f"[WARN] Requested EEG channels not found in EDF: {missing}")
        return [c for c in EEG_CHANNELS if c in raw.ch_names]
    return [c for c in raw.ch_names
            if not c.upper().startswith(
                tuple(p.upper() for p in EXCLUDE_CHANNEL_PREFIXES))]


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Load stimulus log ----------------------------------------------------
    csv_df, epoch_table, all_csv_trigger_times = load_stimulus_log(STIMULUS_CSV_PATH)

    # -- Load EDF -------------------------------------------------------------
    raw = load_edf(EDF_PATH)

    # -- Detect triggers on as-recorded raw BEFORE re-referencing -------------
    eeg_trigger_times, dc7_sig, sfreq = detect_dc7_triggers(raw)

    # -- Re-reference ---------------------------------------------------------
    raw = rereference_to_channel(raw, REREFERENCE_CHANNELS, REFERENCE_CHANNEL)

    # -- Clock alignment ------------------------------------------------------
    slope, intercept, matched_eeg_trig, residuals = align_clocks(
        all_csv_trigger_times, eeg_trigger_times)

    # QC plot: DC7 channel with trigger markers
    fig, ax = plt.subplots(figsize=(12, 3))
    span = slice(0, min(int(40 * sfreq), len(dc7_sig)))
    ax.plot(np.arange(span.stop) / sfreq, dc7_sig[span], linewidth=0.8)
    for t in matched_eeg_trig[matched_eeg_trig < 40]:
        ax.axvline(t, color="red", linestyle="--", linewidth=1)
    ax.set_title(
        "DC7 trigger channel (first 40 s) - red lines = matched epoch-onset pulses")
    ax.set_xlabel("s")
    fig.tight_layout()
    qc_path = os.path.join(OUTPUT_DIR, "dc7_qc.png")
    fig.savefig(qc_path, dpi=130)
    plt.close(fig)
    print(f"[QC] DC7 plot -> {qc_path}")

    # -- Select EEG channels --------------------------------------------------
    channels = pick_eeg_channels(raw)
    print(f"\n[main] Analysing {len(channels)} channel(s): {channels}")
    print(f"[main] {len(epoch_table)} epochs to process\n")

    # -- Per-epoch, per-channel analysis --------------------------------------
    n         = len(raw.times)
    eeg_times = np.arange(n) / sfreq

    all_results = []
    for ch_name in channels:
        eeg_full = raw.get_data(picks=[ch_name])[0]
        print(f"\n{'='*60}")
        print(f"Channel: {ch_name}")
        print(f"{'='*60}")
        for _, epoch_row in epoch_table.iterrows():
            result = analyze_epoch(
                epoch_info=epoch_row,
                ch_name=ch_name,
                eeg=eeg_full,
                eeg_times=eeg_times,
                csv_df=csv_df,
                sfreq=sfreq,
                slope=slope,
                intercept=intercept,
                out_dir=OUTPUT_DIR,
            )
            if result is not None:
                all_results.append(result)

    if not all_results:
        print("\n[main] No results produced - check paths and trigger alignment.")
        return

    # -- Summary --------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    results_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("[main] === SUMMARY (sorted by frequency, then channel) ===")
    print(results_df[["block_num", "channel", "frequency_hz",
                       "overall_plv", "mean_lag_ms",
                       "snr_flash_f0", "snr_eeg_f0",
                       "duration_valid_s"]]
          .sort_values(["frequency_hz", "channel"])
          .to_string(index=False))
    print(f"\n[main] Full summary -> {summary_path}")

    plot_summary(results_df, OUTPUT_DIR)
    print(f"[main] All outputs written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
