#!/usr/bin/env python3
"""
phase_locking_analysis.py
==========================

Measures phase locking between a visual flicker stimulus (logged frame-by-frame
in a PsychoPy `stimulus_log.csv`, as produced by guidedBreathing_Flash.py) and
an EEG recording (.edf) that contains a hardware TTL trigger on one channel
(default: 'DC7'), sent once per breath cycle at the exact video-refresh flip
that the flicker starts.

WHY THIS APPROACH
------------------
1. The flicker is NOT one continuous sinusoid across the whole recording.
   Each call to flash_phase() in guidedBreathing_Flash.py restarts frame_n at 0,
   so the ON/OFF square wave's phase resets at every inhale->exhale and
   exhale->inhale boundary (54 resets across 27 cycles). Any analysis that
   assumes a single continuous carrier phase from t=0 will be wrong. Instead,
   this script reconstructs the *exact* stimulus waveform frame-by-frame from
   the CSV (which already encodes those resets correctly) and treats that as
   ground truth.

2. The CSV's time base is the stimulus PC's clock (global_time_s, zeroed at
   the very first flicker frame). The EDF's time base is the amplifier's
   internal clock. The two clocks are neither perfectly synchronized in
   offset nor (usually) in rate. This script recovers both an offset AND a
   rate correction (i.e. clock drift) by fitting a straight line between the
   27 CSV-logged cycle-onset triggers and the matching pulses detected on the
   DC7 channel, rather than just anchoring on the first trigger.

3. "Phase locking between the two" is quantified three complementary ways:
     a) Continuous PLV  - Hilbert phase of a reconstructed reference stimulus
        waveform vs Hilbert phase of the (narrowband-filtered) EEG, over the
        whole entrainment period. This is the most literal reading of
        "phase locking between stimulus and EEG".
     b) Trial-locked ITPC - classic SSVEP inter-trial phase coherence,
        epoching the EEG to each of the 27 cycle-onset triggers.
     c) Spectral SNR - Welch PSD peak at the flicker frequency (and 2nd
        harmonic) during flicker-on segments.

USAGE
-----
Edit the CONFIG block below (EDF_PATH at minimum), then:
    python3 phase_locking_analysis.py

Requires: numpy, scipy, pandas, mne, matplotlib
    pip install mne  (mne pulls in edfio/numpy/scipy already)
"""

import os
import warnings

# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
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
# CONFIG - edit these for your session
# ============================================================================
EDF_PATH = r"C:\Users\saiik\Downloads\MIRIAM EXPT\sub00 sai\post flash sai\sub00~ Sai_76ef5294-fcd4-4840-9222-ecb7887a3274.edf"                 # <-- point this at your .edf file
STIMULUS_CSV_PATH = r"C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\data\stimulus_log.csv"     # PsychoPy stimulus log

REFERENCE_CHANNEL = "Cz"                   # Reference channel to remove noise
REREFERENCE_CHANNELS = ["O1", "O2"]        # Re-derived as bipolar (O1 - Cz, O2 - Cz)
EEG_CHANNELS = ["O1", "O2"]                # Only analyze O1 and O2 channels for phase locking

DC7_CHANNEL_NAME = "DC7"                   # Trigger channel name in the EDF
DC7_THRESHOLD = 0.005                      # Threshold on deviation from baseline to isolate ~30 TTL pulses

# Channels to auto-exclude if EEG_CHANNELS is set to None
EXCLUDE_CHANNEL_PREFIXES = ("DC", "EDF Annotations", "STATUS", "TRIG", "EVENT")

NARROWBAND_HALFWIDTH_HZ = 1.0     # Bandpass half-width around the flicker freq
FILTER_ORDER = 4
TRIGGER_MATCH_WARN_TOL_S = 0.05   # Warn if best-fit trigger alignment residual exceeds this
MIN_TRIGGER_ISI_S = 1.0           # Refractory period for trigger detection (cycles are ~10s apart)
MERGE_ASOF_TOL_S = 0.05           # Max gap allowed when mapping an EEG sample to a CSV frame

EPOCH_TMIN, EPOCH_TMAX = -1.0, 10.0   # Window around each cycle-onset trigger for ITPC (s)
SLIDING_PLV_WIN_S, SLIDING_PLV_STEP_S = 2.0, 0.5

OUTPUT_DIR = "phase_locking_results"

# ============================================================================
# 1. STIMULUS LOG
# ============================================================================

def load_stimulus_log(csv_path):
    """Load the PsychoPy stimulus log and derive:
      - the flicker frequency (fit from the phase data itself, not assumed)
      - the CSV-clock (global_time_s) times of the 27 cycle-onset triggers
      - a fast lookup table for reconstructing the exact ON/OFF square wave
        at any arbitrary query time (used to build the reference signal in
        EEG time later).
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("global_time_s").reset_index(drop=True)
    df["state_bin"] = (df["stimulus_state"] == "ON").astype(float)

    # Fit flicker frequency per contiguous (cycle, breath_phase) block -
    # each block's frame counter restarts at 0, so phase must be unwrapped
    # and fit separately per block, then averaged.
    freqs = []
    for (_cyc, _bp), blk in df.groupby(["cycle", "breath_phase"]):
        if len(blk) < 10:
            continue
        t = blk["time_s"].values
        ph = np.unwrap(np.deg2rad(blk["phase_deg"].values))
        A = np.vstack([t, np.ones_like(t)]).T
        sol, *_ = np.linalg.lstsq(A, ph, rcond=None)
        freqs.append(sol[0] / (2 * np.pi))
    flicker_freq = float(np.median(freqs))

    trigger_times = df.loc[df["trigger_sent"] == 1, "global_time_s"].values
    trigger_times = np.sort(trigger_times)

    print(f"[stimulus] {len(df)} logged frames, flicker freq = {flicker_freq:.4f} Hz "
          f"(median of {len(freqs)} block fits, std={np.std(freqs):.4f} Hz)")
    print(f"[stimulus] {len(trigger_times)} cycle-onset triggers logged, "
          f"span {trigger_times[0]:.3f}s -> {trigger_times[-1]:.3f}s")

    return df, flicker_freq, trigger_times


def reference_square_wave(csv_df, query_times_csv_clock, tol_s=MERGE_ASOF_TOL_S):
    """Reconstruct the exact ON/OFF square wave at arbitrary query times
    (expressed in the CSV's clock) by holding each frame's state constant
    until the next logged frame ("previous-value" / step interpolation,
    which is what the monitor physically did between flips).

    Returns an array the same length as query_times_csv_clock, with NaN
    where the query time falls outside the logged flicker windows (e.g.
    during the inter-block gap, or during fixation/eyes-closed rest, which
    were not frame-logged).
    """
    left = pd.DataFrame({"t": query_times_csv_clock})
    right = csv_df[["global_time_s", "state_bin"]].rename(columns={"global_time_s": "t"})
    merged = pd.merge_asof(
        left.sort_values("t"), right.sort_values("t"), on="t",
        direction="backward", tolerance=tol_s,
    )
    # merge_asof requires sorted input; restore original order
    merged = merged.set_index(left.sort_values("t").index).sort_index()
    return merged["state_bin"].values


# ============================================================================
# 2. EDF / RE-REFERENCING / TRIGGER CHANNEL
# ============================================================================

def load_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(f"[edf] loaded {edf_path}")
    print(f"[edf] sfreq={raw.info['sfreq']:.3f} Hz, duration={raw.times[-1]:.1f}s, "
          f"{len(raw.ch_names)} channels")
    return raw


def rereference_to_channel(raw, channels, ref_channel):
    """Re-derive `channels` in-place as bipolar (channel - ref_channel).
    The original channel and the reference channel are replaced by the
    new derivation. Everything outside `channels` and `ref_channel` is untouched."""
    raw = raw.copy()
    if not ref_channel:
        print("[INFO] Re-referencing disabled (REFERENCE_CHANNEL is empty/None). "
              "Using as-recorded hardware reference.")
        return raw

    present = [ch for ch in channels if ch in raw.ch_names]
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        print(f"[WARN] Re-reference channels not found, skipping: {missing}")
    if ref_channel not in raw.ch_names:
        raise ValueError(f"Reference channel '{ref_channel}' not found in recording.")
    if present:
        tmp_names = [f"{ch}__bipolar_tmp" for ch in present]
        raw = mne.set_bipolar_reference(
            raw, anode=present, cathode=[ref_channel] * len(present),
            ch_name=tmp_names, drop_refs=True, verbose=False,
        )
        raw.rename_channels(dict(zip(tmp_names, present)))
        print(f"[INFO] Re-referenced to {ref_channel}: {present}")
    return raw


def detect_dc7_triggers(raw, dc7_name=DC7_CHANNEL_NAME, threshold=DC7_THRESHOLD, min_isi_s=MIN_TRIGGER_ISI_S):
    """Detect TTL pulse onsets on the trigger channel via deviation from baseline
    thresholding + rising-edge detection with a refractory period."""
    if dc7_name not in raw.ch_names:
        matches = [c for c in raw.ch_names if dc7_name.lower() in c.lower()]
        raise ValueError(
            f"Channel '{dc7_name}' not found in EDF. Available channels: {raw.ch_names}. "
            f"Closest matches: {matches}"
        )
    sig = raw.get_data(picks=[dc7_name])[0]
    sfreq = raw.info["sfreq"]

    median = np.median(sig)
    dev = np.abs(sig - median)

    if threshold is None:
        # Adaptive threshold: 10 * robust sigma (MAD)
        mad = np.median(np.abs(dev - np.median(dev)))
        sigma = 1.4826 * mad
        threshold = max(10 * sigma, 0.005)

    above = dev > threshold
    rising = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    # Enforce refractory period so one physical pulse isn't double-counted
    kept = []
    last_t = -np.inf
    for idx in rising:
        t = idx / sfreq
        if t - last_t >= min_isi_s:
            kept.append(t)
            last_t = t
    trigger_times = np.array(kept)

    print(f"[edf] DC7 baseline={median:.5g}, threshold={threshold:.5g}; "
          f"{len(trigger_times)} trigger pulses detected")
    if len(trigger_times) > 1:
        isis = np.diff(trigger_times)
        print(f"[edf] trigger ISIs: min={isis.min():.3f}s max={isis.max():.3f}s "
              f"median={np.median(isis):.3f}s")
    return trigger_times, sig, sfreq


# ============================================================================
# 3. CLOCK ALIGNMENT (CSV clock <-> EDF clock)
# ============================================================================

def match_and_fit_time_mapping(csv_trigger_times, eeg_trigger_times,
                                warn_tol_s=TRIGGER_MATCH_WARN_TOL_S):
    """Match CSV cycle triggers with detected EEG triggers on DC7 and fit a straight line
        t_eeg = slope * t_csv + intercept
    accounting for clock drift (slope != 1) and offset (intercept).
    Robustly handles extra post-cycle triggers and any dropped/missing pulses."""
    if len(eeg_trigger_times) < 3:
        raise ValueError(
            f"Only {len(eeg_trigger_times)} triggers detected on DC7. "
            f"Check DC7_CHANNEL_NAME / DC7_THRESHOLD."
        )

    # Search for best initial alignment using RANSAC / maximum-inlier approach
    best_inliers = None
    best_count = 0

    for csv_t0 in csv_trigger_times[:5]:
        for eeg_t0 in eeg_trigger_times[:10]:
            cand_intercept = eeg_t0 - 1.0 * csv_t0
            cand_csv_matched = []
            cand_eeg_matched = []
            for ct in csv_trigger_times:
                predicted_et = 1.0 * ct + cand_intercept
                diffs = np.abs(eeg_trigger_times - predicted_et)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] < 0.2:  # within 200 ms
                    cand_csv_matched.append(ct)
                    cand_eeg_matched.append(eeg_trigger_times[min_idx])
            if len(cand_csv_matched) > best_count:
                best_count = len(cand_csv_matched)
                best_inliers = (np.array(cand_csv_matched), np.array(cand_eeg_matched))

    if best_inliers is None or len(best_inliers[0]) < 3:
        raise ValueError(
            f"Could not find consistent trigger alignment between {len(csv_trigger_times)} "
            f"CSV triggers and {len(eeg_trigger_times)} EDF triggers."
        )

    matched_csv_times, matched_eeg_times = best_inliers
    slope, intercept = np.polyfit(matched_csv_times, matched_eeg_times, 1)
    fitted = slope * matched_csv_times + intercept
    residuals = matched_eeg_times - fitted
    max_resid_ms = np.max(np.abs(residuals)) * 1000

    print(f"[align] matched {len(matched_csv_times)} of {len(csv_trigger_times)} CSV triggers "
          f"to EDF triggers (total {len(eeg_trigger_times)} detected on DC7)")
    print(f"[align] fit: t_eeg = {slope:.8f} * t_csv + {intercept:.4f}  "
          f"(clock drift {(slope-1)*1e6:.1f} ppm)")
    print(f"[align] max residual after fit: {max_resid_ms:.2f} ms")
    if max_resid_ms / 1000 > warn_tol_s:
        print(f"[align][WARNING] residual exceeds {warn_tol_s*1000:.0f} ms - "
              f"double check DC7 detection / that no triggers were misidentified.")

    if len(eeg_trigger_times) > len(matched_eeg_times):
        n_extra = len(eeg_trigger_times) - len(matched_eeg_times)
        print(f"[align] note: {n_extra} extra DC7 pulse(s) detected beyond the matched "
              f"cycle-onset triggers (e.g., post-breathing fixation / eyes-closed / end triggers).")

    return slope, intercept, matched_eeg_times, residuals


# ============================================================================
# 4. SIGNAL PROCESSING
# ============================================================================

def narrowband_filter(x, sfreq, f0, half_bw=NARROWBAND_HALFWIDTH_HZ, order=FILTER_ORDER):
    nyq = sfreq / 2
    low, high = max(f0 - half_bw, 0.1) / nyq, min(f0 + half_bw, nyq * 0.99) / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def hilbert_phase(x):
    return np.angle(hilbert(x))


def circular_plv(phase_a, phase_b, mask=None):
    diff = phase_a - phase_b
    if mask is not None:
        diff = diff[mask]
    return np.abs(np.mean(np.exp(1j * diff)))


def sliding_plv(phase_a, phase_b, sfreq, valid_mask, win_s=SLIDING_PLV_WIN_S,
                 step_s=SLIDING_PLV_STEP_S):
    win, step = int(win_s * sfreq), int(step_s * sfreq)
    n = len(phase_a)
    centers_t, plv_vals = [], []
    for start in range(0, n - win, step):
        sl = slice(start, start + win)
        if valid_mask[sl].mean() < 0.9:   # require the window to be (almost) fully valid
            continue
        plv_vals.append(circular_plv(phase_a[sl], phase_b[sl]))
        centers_t.append((start + win / 2) / sfreq)
    return np.array(centers_t), np.array(plv_vals)


# ============================================================================
# 5. MAIN ANALYSIS PER CHANNEL
# ============================================================================

def analyze_channel(ch_name, raw, csv_df, flicker_freq, eeg_trigger_times_matched,
                     slope, intercept, out_dir):
    sfreq = raw.info["sfreq"]
    eeg = raw.get_data(picks=[ch_name])[0]
    n = len(eeg)
    eeg_times = np.arange(n) / sfreq  # seconds from EDF recording start

    # Map EEG sample times -> CSV clock, then look up the reconstructed
    # square-wave stimulus at those instants.
    csv_query_times = (eeg_times - intercept) / slope
    ref_wave = reference_square_wave(csv_df, csv_query_times)
    valid_mask = ~np.isnan(ref_wave)
    if valid_mask.sum() < sfreq * 5:
        print(f"[{ch_name}] fewer than 5s of overlap with logged stimulus frames - skipping")
        return None
    ref_wave_filled = np.nan_to_num(ref_wave, nan=0.5)

    eeg_nb = narrowband_filter(eeg, sfreq, flicker_freq)
    ref_nb = narrowband_filter(ref_wave_filled, sfreq, flicker_freq)

    eeg_phase = hilbert_phase(eeg_nb)
    ref_phase = hilbert_phase(ref_nb)

    overall_plv = circular_plv(ref_phase, eeg_phase, mask=valid_mask)
    # Positive mean_lag_ms = EEG follows (lags behind) the stimulus by that many ms,
    # i.e. a physiological response latency. Negative would mean EEG leads the
    # stimulus, which is not physiologically meaningful and signals a mis-alignment.
    mean_lag_rad = np.angle(np.mean(np.exp(1j * (ref_phase[valid_mask] - eeg_phase[valid_mask]))))
    mean_lag_ms = (mean_lag_rad / (2 * np.pi * flicker_freq)) * 1000

    t_slide, plv_slide = sliding_plv(ref_phase, eeg_phase, sfreq, valid_mask)

    # Per-cycle PLV using the matched trigger times as cycle boundaries
    cycle_plvs = []
    for i in range(len(eeg_trigger_times_matched) - 1):
        t0, t1 = eeg_trigger_times_matched[i], eeg_trigger_times_matched[i + 1]
        sl_mask = (eeg_times >= t0) & (eeg_times < t1) & valid_mask
        if sl_mask.sum() > sfreq:
            cycle_plvs.append(circular_plv(ref_phase, eeg_phase, mask=sl_mask))
        else:
            cycle_plvs.append(np.nan)
    cycle_plvs = np.array(cycle_plvs)

    # ITPC locked to trigger onset (classic SSVEP check) - uses continuous
    # narrowband EEG phase, epoched around each cycle-onset trigger.
    pre, post = int(EPOCH_TMIN * sfreq), int(EPOCH_TMAX * sfreq)
    epochs = []
    for trig_t in eeg_trigger_times_matched:
        c = int(round(trig_t * sfreq))
        if c + pre >= 0 and c + post < n:
            epochs.append(eeg_phase[c + pre:c + post])
    itpc_t = np.arange(pre, post) / sfreq
    itpc = None
    if epochs:
        epochs = np.array(epochs)
        itpc = np.abs(np.mean(np.exp(1j * epochs), axis=0))

    # Spectral SNR: during-flicker segments vs baseline post-cycles
    f_on, pxx_on = welch(eeg[valid_mask], fs=sfreq, nperseg=min(4 * int(sfreq), valid_mask.sum()))
    baseline_start = eeg_trigger_times_matched[-1] + 15.0  # skip a few seconds after last cycle
    base_mask = eeg_times > baseline_start
    f_base = pxx_base = None
    if base_mask.sum() > 4 * sfreq:
        f_base, pxx_base = welch(eeg[base_mask], fs=sfreq, nperseg=min(4 * int(sfreq), base_mask.sum()))

    def snr_at(f, pxx, target):
        idx = np.argmin(np.abs(f - target))
        band = (f > target - 2) & (f < target + 2) & (np.abs(f - target) > 0.5)
        return pxx[idx] / np.mean(pxx[band])

    snr_on_f0 = snr_at(f_on, pxx_on, flicker_freq)
    snr_on_2f0 = snr_at(f_on, pxx_on, 2 * flicker_freq)

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ref_info = f" (re-ref to {REFERENCE_CHANNEL})" if REFERENCE_CHANNEL else ""
    fig.suptitle(f"Channel {ch_name}{ref_info}  |  flicker {flicker_freq:.3f} Hz  |  "
                 f"overall PLV={overall_plv:.3f}  lag={mean_lag_ms:+.1f} ms")

    ax = axes[0, 0]
    ax.plot(t_slide, plv_slide)
    ax.set_title("Sliding PLV (stimulus vs EEG)")
    ax.set_xlabel("time since recording start (s)")
    ax.set_ylabel("PLV")
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.bar(np.arange(1, len(cycle_plvs) + 1), cycle_plvs)
    ax.set_title("PLV per breath cycle")
    ax.set_xlabel("cycle #")
    ax.set_ylabel("PLV")
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    if itpc is not None:
        ax.plot(itpc_t, itpc)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Trigger-locked ITPC")
    ax.set_xlabel("time from cycle onset (s)")
    ax.set_ylabel("ITPC")
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    ax.semilogy(f_on, pxx_on, label="during flicker")
    if pxx_base is not None:
        ax.semilogy(f_base, pxx_base, label="baseline (post-cycles)")
    ax.axvline(flicker_freq, color="r", linestyle="--", linewidth=1, label="f0")
    ax.axvline(2 * flicker_freq, color="r", linestyle=":", linewidth=1, label="2*f0")
    ax.set_xlim(0, min(40, 4 * flicker_freq))
    ax.set_title(f"PSD  (SNR@f0={snr_on_f0:.2f}, SNR@2f0={snr_on_2f0:.2f})")
    ax.set_xlabel("Hz")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{ch_name}_phase_locking.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    return {
        "channel": ch_name,
        "overall_plv": overall_plv,
        "mean_lag_ms": mean_lag_ms,
        "mean_cycle_plv": np.nanmean(cycle_plvs),
        "peak_itpc": np.max(itpc) if itpc is not None else np.nan,
        "snr_f0": snr_on_f0,
        "snr_2f0": snr_on_2f0,
        "figure": fig_path,
    }


# ============================================================================
# 6. ORCHESTRATION
# ============================================================================

def pick_eeg_channels(raw):
    if EEG_CHANNELS is not None:
        return [c for c in EEG_CHANNELS if c in raw.ch_names]
    return [c for c in raw.ch_names
            if not c.upper().startswith(tuple(p.upper() for p in EXCLUDE_CHANNEL_PREFIXES))]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_df, flicker_freq, csv_trigger_times = load_stimulus_log(STIMULUS_CSV_PATH)
    raw = load_edf(EDF_PATH)

    # 1. Detect DC7 triggers on as-recorded raw before re-referencing
    eeg_trigger_times, dc7_sig, sfreq = detect_dc7_triggers(raw)

    # 2. Re-reference O1 and O2 against Cz (O1 - Cz, O2 - Cz)
    raw = rereference_to_channel(raw, REREFERENCE_CHANNELS, REFERENCE_CHANNEL)

    slope, intercept, matched_eeg_trig, residuals = match_and_fit_time_mapping(
        csv_trigger_times, eeg_trigger_times)

    # QC plot of the DC7 channel around the first few triggers
    fig, ax = plt.subplots(figsize=(10, 3))
    span = slice(0, int(35 * sfreq))
    ax.plot(np.arange(span.stop) / sfreq, dc7_sig[span])
    for t in matched_eeg_trig[:4]:
        ax.axvline(t, color="r", linestyle="--", linewidth=1)
    ax.set_title("DC7 trigger channel (first 35s) with detected cycle-onset pulses")
    ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dc7_qc.png"), dpi=130)
    plt.close(fig)

    channels = pick_eeg_channels(raw)
    print(f"[main] analyzing {len(channels)} candidate EEG channel(s): {channels}")

    results = []
    for ch in channels:
        r = analyze_channel(ch, raw, csv_df, flicker_freq, matched_eeg_trig,
                             slope, intercept, OUTPUT_DIR)
        if r:
            results.append(r)

    if not results:
        print("[main] no channels produced results.")
        return

    summary = pd.DataFrame(results).sort_values("overall_plv", ascending=False)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n[main] === SUMMARY (sorted by overall PLV) ===")
    print(summary.to_string(index=False))
    print(f"\n[main] full results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()