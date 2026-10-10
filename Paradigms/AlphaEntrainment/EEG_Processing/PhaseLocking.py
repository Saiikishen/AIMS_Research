#!/usr/bin/env python3
"""Stimulus locking and power spectra from synchronized EEG and stimulus logs.

Preprocess with a 50 Hz notch and 3-30 Hz band-pass. Estimate PLV/ITPC from
a separate, flicker-centered band; estimate PSD from the broadband output.
Filter settings are fixed in advance, never selected to maximize PLV or SNR.
"""
import os

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
from scipy.signal import butter, iirnotch, tf2sos, sosfiltfilt, hilbert, welch

mne.set_log_level("ERROR")

# ============================================================================
# CONFIG - edit these for your session
# ============================================================================
EDF_PATH = r"C:\Users\saiik\Downloads\miriam expt edf\dataset #4\MRIDUL_SUB27\FLASHING\SUB27~ MRIDUL_e886c48a-5c51-4654-8466-22148860ba27.edf"
STIMULUS_CSV_PATH = r"C:\Users\saiik\Downloads\miriam expt edf\dataset #4\MRIDUL_SUB27\data\stimulus_log.csv"

REFERENCE_CHANNEL = "Cz"                   # Reference channel to remove noise
REREFERENCE_CHANNELS = ["O1", "O2"]        # Re-derived as bipolar (O1 - Cz, O2 - Cz)
EEG_CHANNELS = ["O1", "O2"]                # Only analyze O1 and O2 channels for phase locking

DC7_CHANNEL_NAME = "DC7"                   # Trigger channel name in the EDF
DC7_THRESHOLD = 0.005                      # Threshold on deviation from baseline to isolate ~30 TTL pulses

# Channels to auto-exclude if EEG_CHANNELS is set to None
EXCLUDE_CHANNEL_PREFIXES = ("DC", "EDF Annotations", "STATUS", "TRIG", "EVENT")

NOTCH_FREQ_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ = 3.0, 30.0
FILTER_ORDER = 4
# Phase is meaningful for the selected oscillation: isolate the fundamental
# for both EEG and stimulus. This extra band-pass NEVER feeds the PSD branch.
PHASE_HALF_BANDWIDTH_HZ = 0.5    # f0 +/- 1 Hz, fixed across subjects
TRIGGER_MATCH_WARN_TOL_S = 0.05   # Warn if best-fit trigger alignment residual exceeds this
TRIGGER_MATCH_TOL_S = 0.2        # Maximum CSV/EDF trigger-pair error
MIN_TRIGGER_ISI_S = 1.0           # Refractory period for trigger detection (cycles are ~10s apart)
MERGE_ASOF_TOL_S = 0.05           # Max gap allowed when mapping an EEG sample to a CSV frame

EPOCH_TMIN, EPOCH_TMAX = -1.0, 10.0   # Window around each cycle-onset trigger for ITPC (s)
SLIDING_PLV_WIN_S, SLIDING_PLV_STEP_S = 2.0, 0.5
BOUNDARY_EXCLUDE_S = 0.5           # seconds to exclude around each phase-reset boundary (filter ringing)

# None: locate offset using the frame-log end and post-flash marker sequence.
# Set a one-based CSV trigger number only for a log with an explicit offset row.
FLASH_STOP_TRIGGER_NUMBER = None
# Offset -> fixation end -> eyes-closed end, from guidedBreathing_Flash.py.
# Set to None only for protocols without this marker sequence.
POST_FLASH_MARKER_INTERVALS_S = (30.0, 60.0)
AFTER_FLASH_DELAY_S = 15.0        # Delay after flash offset, not the last cycle onset

# Frequency filtering only: no amplitude, flatline or BAD-annotation rejection.
# Nonfinite samples remain gaps so they cannot propagate through the transforms.
PHASE_EDGE_EXCLUDE_S = 1.0       # Discard filter/Hilbert edges of each finite run
PSD_WINDOW_S = 4.0               # Same Welch window/grid for both conditions
MIN_CYCLE_PHASE_S = 1.0          # Require more than this much phase per cycle
PLV_CYCLE_COVERAGE_TARGET = 0.90  # Report shortfalls; never tune filters to meet it

OUTPUT_DIR = "phase_locking_results"

# ============================================================================
# 1. STIMULUS LOG
# ============================================================================

def load_stimulus_log(csv_path):
    """Load the PsychoPy stimulus log and derive:
      - the flicker frequency (fit from the phase data itself, not assumed)
      - the CSV-clock (global_time_s) times of the cycle-onset triggers
      - a fast lookup table for reconstructing the logged ON/OFF square wave
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
    """Reconstruct the logged ON/OFF square wave at arbitrary query times
    (expressed in the CSV's clock) by holding each frame's state constant
    until the next logged frame ("previous-value" / step interpolation,
    approximating the displayed state between flips). This is a log-based
    reference, not a direct photodiode measurement of the monitor output.

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

def find_protocol_flash_offset(eeg_trigger_times):
    """Identify offset from an unambiguous sequence of consecutive rest markers."""
    if POST_FLASH_MARKER_INTERVALS_S is None:
        return None
    intervals = np.asarray(POST_FLASH_MARKER_INTERVALS_S, dtype=float)
    if (intervals.ndim != 1 or len(intervals) < 2 or not np.isfinite(intervals).all()
            or np.any(intervals <= 2 * TRIGGER_MATCH_TOL_S)):
        raise ValueError("POST_FLASH_MARKER_INTERVALS_S needs at least two positive intervals.")
    times = np.asarray(eeg_trigger_times, dtype=float)
    gaps = np.diff(times)
    # Do not skip intervening pulses: regular cycle onsets also span 30/60 s.
    candidates = [times[i] for i in range(len(times) - len(intervals))
                  if np.all(np.abs(gaps[i:i + len(intervals)] - intervals)
                            < TRIGGER_MATCH_TOL_S)]
    if len(candidates) > 1:
        raise ValueError("Multiple post-flash marker sequences found; check the session/protocol "
                         "or specify an explicit CSV offset marker.")
    if not candidates:
        return None
    stop = float(candidates[0])
    print(f"[protocol] offset anchor at {stop:.3f}s EDF time from post-flash "
          f"marker intervals {intervals.tolist()}s")
    return stop


def match_and_fit_time_mapping(csv_trigger_times, eeg_trigger_times,
                                warn_tol_s=TRIGGER_MATCH_WARN_TOL_S,
                                csv_stop_time=None, eeg_stop_time=None):
    """Match CSV cycle triggers with detected EEG triggers on DC7 and fit a straight line
        t_eeg = slope * t_csv + intercept
    accounting for clock drift (slope != 1) and offset (intercept).
    Refit and rematch until the one-to-one pairing stabilizes. An independently
    identified offset, when supplied, prevents slips by a whole breathing cycle.
    Return matched CSV indices so missing pulses retain their cycle identity.
    """
    csv_trigger_times = np.asarray(csv_trigger_times, dtype=float)
    eeg_trigger_times = np.asarray(eeg_trigger_times, dtype=float)
    for name, times in (("CSV", csv_trigger_times), ("EDF", eeg_trigger_times)):
        if (times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError(f"{name} needs at least 3 finite, strictly increasing triggers.")
    if (csv_stop_time is None) != (eeg_stop_time is None):
        raise ValueError("Supply both CSV and EDF offset times, or neither.")
    anchored = eeg_stop_time is not None
    if anchored and (not np.isfinite([csv_stop_time, eeg_stop_time]).all()
                     or csv_stop_time <= csv_trigger_times[-1]):
        raise ValueError("The offset anchor must be finite and follow the last CSV onset.")

    def match(slope, intercept):
        csv_indices, eeg_indices = [], []
        last_eeg_index = -1
        for i, predicted in enumerate(slope * csv_trigger_times + intercept):
            diffs = np.abs(eeg_trigger_times - predicted)
            diffs[:last_eeg_index + 1] = np.inf  # ordered, unique pulse matches
            if anchored:
                diffs[eeg_trigger_times >= eeg_stop_time] = np.inf
            j = int(np.argmin(diffs))
            if diffs[j] < TRIGGER_MATCH_TOL_S:
                csv_indices.append(i)
                eeg_indices.append(j)
                last_eeg_index = j
        return np.array(csv_indices, dtype=int), np.array(eeg_indices, dtype=int)

    seed_slopes = [1.0]
    interval_ratio = np.median(np.diff(eeg_trigger_times)) / np.median(np.diff(csv_trigger_times))
    if 0.95 <= interval_ratio <= 1.05:
        seed_slopes.append(interval_ratio)
    best = None
    best_count = 0
    for seed_slope in seed_slopes:
        for csv_t0 in csv_trigger_times[:5]:
            for eeg_t0 in eeg_trigger_times[:10]:
                slope = seed_slope
                intercept = eeg_t0 - slope * csv_t0
                ci, ei = match(slope, intercept)
                for _ in range(20):
                    if len(ci) < 3:
                        break
                    slope, intercept = np.polyfit(csv_trigger_times[ci], eeg_trigger_times[ei], 1)
                    if not 0.95 <= slope <= 1.05:
                        break
                    next_ci, next_ei = match(slope, intercept)
                    if np.array_equal(ci, next_ci) and np.array_equal(ei, next_ei):
                        if anchored and abs(slope * csv_stop_time + intercept - eeg_stop_time) >= TRIGGER_MATCH_TOL_S:
                            break
                        # Equal-coverage periodic matches retain the earliest seed.
                        if len(ci) > best_count:
                            best_count = len(ci)
                            best = slope, intercept, ci, ei
                        break
                    ci, ei = next_ci, next_ei

    if best is None or best_count < max(3, int(np.ceil(len(csv_trigger_times) / 2))):
        raise ValueError(
            f"Could not align enough triggers: best match was {best_count} of "
            f"{len(csv_trigger_times)} CSV triggers ({len(eeg_trigger_times)} EDF pulses). "
            "Check that CSV and EDF belong to the same session and inspect DC7 detection."
        )

    slope, intercept, csv_indices, eeg_indices = best
    matched_csv_times = csv_trigger_times[csv_indices]
    matched_eeg_times = eeg_trigger_times[eeg_indices]
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

    missing = np.setdiff1d(np.arange(len(csv_trigger_times)), csv_indices) + 1
    if len(missing):
        print(f"[align][WARNING] CSV trigger(s) without a detected EDF match: {missing.tolist()}")
    n_unmatched = len(eeg_trigger_times) - len(matched_eeg_times)
    if n_unmatched:
        print(f"[align] {n_unmatched} DC7 pulse(s) unmatched to CSV cycle onsets; "
              "these may include flash offset and later protocol markers.")

    return slope, intercept, csv_indices, matched_eeg_times, residuals


# ============================================================================
# 4. SIGNAL PROCESSING
# ============================================================================

def contiguous_runs(mask):
    """Yield [start, stop) sample intervals without deleting gaps in time."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def flashing_period(csv_df, csv_triggers, eeg_triggers, matched_eeg_triggers, slope, intercept,
                    protocol_stop_eeg=None):
    """Locate a measured offset pulse using the frame-log end or an explicit marker."""
    if FLASH_STOP_TRIGGER_NUMBER is None:
        stop_csv = float(csv_df["global_time_s"].max())
        candidates = eeg_triggers[~np.isin(eeg_triggers, matched_eeg_triggers)]
        candidates = candidates[candidates > slope * csv_triggers[-1] + intercept]
        if protocol_stop_eeg is not None:
            candidates = candidates[candidates == protocol_stop_eeg]
        flash_df = csv_df.copy()
        marker = ("post-flash marker sequence" if protocol_stop_eeg is not None
                  else "EDF pulse near the end of the frame log")
    else:
        if not 2 <= FLASH_STOP_TRIGGER_NUMBER <= len(csv_triggers):
            raise ValueError("FLASH_STOP_TRIGGER_NUMBER must identify a logged offset trigger.")
        stop_csv = csv_triggers[FLASH_STOP_TRIGGER_NUMBER - 1]
        candidates = matched_eeg_triggers
        flash_df = csv_df.loc[csv_df["global_time_s"] < stop_csv].copy()
        marker = f"explicit CSV offset trigger {FLASH_STOP_TRIGGER_NUMBER}"
    predicted_stop = slope * stop_csv + intercept
    if not len(candidates):
        raise ValueError(f"No eligible flash-offset pulse near {predicted_stop:.3f}s EDF time. "
                         "Check the frame log, protocol and DC7 detection.")
    stop_eeg = candidates[np.argmin(np.abs(candidates - predicted_stop))]
    if abs(stop_eeg - predicted_stop) >= TRIGGER_MATCH_TOL_S:
        raise ValueError(
            f"Flash-offset pulse not found within {TRIGGER_MATCH_TOL_S:.3f}s of "
            f"{predicted_stop:.3f}s EDF time (nearest eligible pulse: {stop_eeg:.3f}s). "
            "Check the frame log, protocol and trigger alignment."
        )
    print(f"[protocol] flashing stops at {stop_eeg:.3f}s EDF time ({marker})")
    return flash_df, stop_eeg


def cycle_timing(csv_triggers, matched_csv_indices, matched_eeg_triggers,
                 slope, intercept, flash_stop_eeg):
    """Keep every logged cycle, flagging onsets inferred from the clock fit."""
    onsets = slope * np.asarray(csv_triggers) + intercept
    detected = np.zeros(len(onsets), dtype=bool)
    onsets[matched_csv_indices] = matched_eeg_triggers
    detected[matched_csv_indices] = True
    keep = onsets < flash_stop_eeg
    onsets, detected = onsets[keep], detected[keep]
    if np.any(np.diff(onsets) <= 0):
        raise ValueError("Fitted and measured cycle boundaries are not strictly increasing.")
    if not detected.all():
        print(f"[protocol] {np.sum(~detected)} cycle onset(s) estimated from the clock fit; "
              "flagged in cycle CSV output and excluded from trigger-locked ITPC")
    if len(onsets) and onsets[0] < 0:
        print(f"[protocol][WARNING] first logged onset maps to {onsets[0]:.3f}s, "
              "before EDF start; the initial cycle is only partially recorded")
    return onsets, detected


def frequency_filter(x, sfreq):
    """Apply a zero-phase 50 Hz notch, then a 3-30 Hz Butterworth band-pass.

    No amplitude-dependent sample selection. This preprocessing feeds PSD
    directly and precedes the separate flicker-centered phase extraction.
    """
    if not 0 < BANDPASS_LOW_HZ < BANDPASS_HIGH_HZ < sfreq / 2:
        raise ValueError("Band-pass frequencies must lie below the Nyquist frequency.")
    if not 0 < NOTCH_FREQ_HZ < sfreq / 2 or NOTCH_Q <= 0:
        raise ValueError("Notch frequency must lie below Nyquist and Q must be positive.")
    b, a = iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, fs=sfreq)
    notched = sosfiltfilt(tf2sos(b, a), x)
    bandpass = butter(FILTER_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                      btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(bandpass, notched)


def hilbert_phase(x):
    return np.angle(hilbert(x))


def phase_filter_sos(sfreq, flicker_freq):
    """Fixed-bandwidth, zero-phase-compatible filter for the fundamental."""
    low = flicker_freq - PHASE_HALF_BANDWIDTH_HZ
    high = flicker_freq + PHASE_HALF_BANDWIDTH_HZ
    if (not np.isfinite([sfreq, low, high]).all()
            or PHASE_HALF_BANDWIDTH_HZ <= 0
            or not BANDPASS_LOW_HZ < low < high < BANDPASS_HIGH_HZ < sfreq / 2):
        raise ValueError("The flicker-centered phase band must lie strictly inside "
                         "the preprocessing band and below Nyquist. Check f0 and "
                         "PHASE_HALF_BANDWIDTH_HZ.")
    return butter(FILTER_ORDER, [low, high], btype="bandpass", fs=sfreq, output="sos")


def filter_and_extract_phase(eeg, reference, sfreq, flicker_freq):
    """Return broadband EEG for PSD and fundamental-only EEG/reference phases.

    Nonfinite input samples keep their time positions as gaps. Discard only
    transform edges; large amplitudes, flatlines and annotations do not reject
    any samples. Filter complete finite EEG runs before selecting conditions.
    Reference gaps interrupt phase extraction without affecting EEG for PSD.
    Both phase signals receive identical filters before the Hilbert transform.
    """
    eeg = np.asarray(eeg, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if eeg.ndim != 1 or reference.shape != eeg.shape:
        raise ValueError("EEG and reference must be one-dimensional arrays of equal length.")
    phase_sos = phase_filter_sos(sfreq, flicker_freq)
    filtered_eeg = np.full(len(eeg), np.nan)
    eeg_phase = np.full(len(eeg), np.nan)
    ref_phase = np.full(len(eeg), np.nan)
    edge = max(1, int(round(PHASE_EDGE_EXCLUDE_S * sfreq)))
    min_samples = max(2 * edge + int(round(sfreq)), 3 * (2 * FILTER_ORDER + 1) + 1)
    for start, stop in contiguous_runs(np.isfinite(eeg)):
        if stop - start < min_samples:
            continue
        eeg_run = frequency_filter(eeg[start:stop], sfreq)
        filtered_eeg[start + edge:stop - edge] = eeg_run[edge:-edge]
        for ref_start, ref_stop in contiguous_runs(np.isfinite(reference[start:stop])):
            if ref_stop - ref_start < min_samples:
                continue
            reference_run = frequency_filter(reference[start + ref_start:start + ref_stop], sfreq)
            ep = hilbert_phase(sosfiltfilt(phase_sos, eeg_run[ref_start:ref_stop]))
            rp = hilbert_phase(sosfiltfilt(phase_sos, reference_run))
            dest = slice(start + ref_start + edge, start + ref_stop - edge)
            eeg_phase[dest] = ep[edge:-edge]
            ref_phase[dest] = rp[edge:-edge]
    return filtered_eeg, eeg_phase, ref_phase


def contiguous_welch(eeg, sfreq, valid_mask, window_s=PSD_WINDOW_S):
    """Average Welch PSDs without any window crossing an excluded interval.

    All runs use the same window length/grid and 50% overlap. Weight each
    run's mean PSD by its number of full Welch windows, not by run count.
    Runs shorter than one full window contribute nothing.
    """
    win = int(round(window_s * sfreq))
    if win < 2:
        raise ValueError("PSD window must contain at least two samples.")
    overlap = win // 2
    step = win - overlap
    total_windows = 0
    weighted_psd = None
    frequencies = None
    for start, stop in contiguous_runs(valid_mask & np.isfinite(eeg)):
        if stop - start < win:
            continue
        frequencies, psd = welch(eeg[start:stop], fs=sfreq, window="hann",
                                 nperseg=win, noverlap=overlap, nfft=win,
                                 detrend="constant", average="mean")
        count = 1 + (stop - start - win) // step
        if weighted_psd is None:
            weighted_psd = psd * count
        else:
            weighted_psd += psd * count
        total_windows += count
    if total_windows == 0:
        return None, None, 0
    return frequencies, weighted_psd / total_windows, total_windows


def circular_plv(phase_a, phase_b, mask=None):
    phase_a, phase_b = np.asarray(phase_a), np.asarray(phase_b)
    if phase_a.shape != phase_b.shape:
        raise ValueError("Phase arrays must have the same shape.")
    valid = np.isfinite(phase_a) & np.isfinite(phase_b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        return np.nan
    diff = phase_a[valid] - phase_b[valid]
    return np.abs(np.mean(np.exp(1j * diff)))


def sliding_plv(phase_a, phase_b, sfreq, valid_mask, win_s=SLIDING_PLV_WIN_S,
                 step_s=SLIDING_PLV_STEP_S):
    phase_a, phase_b = np.asarray(phase_a), np.asarray(phase_b)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(phase_a) & np.isfinite(phase_b)
    win, step = int(round(win_s * sfreq)), int(round(step_s * sfreq))
    if win < 1 or step < 1:
        raise ValueError("Sliding PLV window and step must contain at least one sample.")
    n = len(phase_a)
    centers_t, plv_vals = [], []
    for start in range(0, n - win + 1, step):
        sl = slice(start, start + win)
        if valid_mask[sl].mean() < 0.9:   # require the window to be (almost) fully valid
            continue
        plv_vals.append(circular_plv(phase_a[sl], phase_b[sl], mask=valid_mask[sl]))
        centers_t.append((start + win / 2) / sfreq)
    return np.array(centers_t), np.array(plv_vals)


# ============================================================================
# 5. MAIN ANALYSIS PER CHANNEL
# ============================================================================

def analyze_channel(ch_name, raw, csv_df, flicker_freq, cycle_onsets_eeg,
                     cycle_onsets_detected, slope, intercept, out_dir, flash_stop_eeg):
    sfreq = raw.info["sfreq"]
    eeg = raw.get_data(picks=[ch_name])[0]
    n = len(eeg)
    eeg_times = np.arange(n) / sfreq  # seconds from EDF recording start

    # Map EEG sample times -> CSV clock, then look up the reconstructed
    # square-wave stimulus at those instants.
    csv_query_times = (eeg_times - intercept) / slope
    ref_wave = reference_square_wave(csv_df, csv_query_times).copy()
    # Do not extend the last logged frame across the measured flash-offset pulse.
    ref_wave[eeg_times >= flash_stop_eeg] = np.nan
    stimulus_mask = np.isfinite(ref_wave)

    # Exclude samples near phase-reset boundaries to avoid filter-ringing artifacts.
    # Each flash_phase() call resets frame_n to 0, creating a step discontinuity in
    # the reference square wave that causes IIR filter ringing for several cycles.
    boundary_times_csv = []
    for _cyc, grp in csv_df.groupby("cycle"):
        for bp in grp["breath_phase"].unique():
            sub = grp[grp["breath_phase"] == bp]
            boundary_times_csv.append(sub["global_time_s"].iloc[0])
    boundary_times_eeg = slope * np.array(boundary_times_csv) + intercept
    boundary_times_eeg = np.r_[boundary_times_eeg, flash_stop_eeg]
    boundary_mask = np.ones(n, dtype=bool)
    for bt in boundary_times_eeg:
        boundary_mask &= np.abs(eeg_times - bt) > BOUNDARY_EXCLUDE_S
    ref_wave_filled = np.nan_to_num(ref_wave, nan=0.5)

    # PSD uses notch + 3-30 Hz EEG. PLV/ITPC additionally isolate f0 +/- 1 Hz
    # in BOTH EEG and reference. No amplitude-dependent sample inclusion.
    filtered_eeg, eeg_phase, ref_phase = filter_and_extract_phase(
        eeg, ref_wave_filled, sfreq, flicker_freq)
    psd_on_mask = stimulus_mask & boundary_mask & np.isfinite(filtered_eeg)
    valid_mask = psd_on_mask & np.isfinite(eeg_phase) & np.isfinite(ref_phase)
    if valid_mask.sum() < sfreq * 5:
        print(f"[{ch_name}] fewer than 5s of usable flashing phase - skipping")
        return None

    overall_plv = circular_plv(ref_phase, eeg_phase, mask=valid_mask)
    # Wrapped phase-equivalent lag, not an unambiguous physiological latency.
    mean_lag_rad = np.angle(np.mean(np.exp(1j * (ref_phase[valid_mask] - eeg_phase[valid_mask]))))
    mean_lag_ms = (mean_lag_rad / (2 * np.pi * flicker_freq)) * 1000

    # Missing pulses must not merge cycles or shift subsequent cycle numbers.
    flash_onsets = cycle_onsets_eeg[cycle_onsets_detected]
    cycle_boundaries = np.r_[cycle_onsets_eeg, flash_stop_eeg]
    boundary_detected = np.r_[cycle_onsets_detected, True]  # measured offset TTL
    cycle_plvs = []
    cycle_rows = []
    for cycle_number, (t0, t1) in enumerate(zip(cycle_boundaries[:-1], cycle_boundaries[1:]), 1):
        cycle_mask = (eeg_times >= t0) & (eeg_times < t1)
        sl_mask = cycle_mask & valid_mask
        usable_s = sl_mask.sum() / sfreq
        plv = (circular_plv(ref_phase, eeg_phase, mask=sl_mask)
               if usable_s > MIN_CYCLE_PHASE_S else np.nan)
        cycle_plvs.append(plv)
        cycle_rows.append({
            "cycle": cycle_number, "start_s": t0, "stop_s": t1,
            "start_trigger_detected": bool(boundary_detected[cycle_number - 1]),
            "stop_trigger_detected": bool(boundary_detected[cycle_number]),
            "duration_s": t1 - t0, "usable_phase_s": usable_s,
            "usable_phase_fraction": sl_mask.sum() / cycle_mask.sum() if cycle_mask.any() else 0.0,
            "plv": plv, "plv_available": bool(np.isfinite(plv)),
        })
    cycle_plvs = np.array(cycle_plvs)
    cycle_path = os.path.join(out_dir, f"{ch_name}_cycle_plv.csv")
    pd.DataFrame(cycle_rows).to_csv(cycle_path, index=False)


    pre, post = int(EPOCH_TMIN * sfreq), int(EPOCH_TMAX * sfreq)
    epochs = []
    for trig_t in flash_onsets:
        c = int(round(trig_t * sfreq))
        if c + pre >= 0 and c + post <= n and (c + post) / sfreq <= flash_stop_eeg:
            epoch = eeg_phase[c + pre:c + post]
            # Only missing samples or transform edges invalidate a trial.
            if np.isfinite(epoch).all():
                epochs.append(epoch)
    itpc_epoch_count = len(epochs)
    itpc_t = np.arange(pre, post) / sfreq
    itpc = None
    if itpc_epoch_count >= 2:
        epochs = np.array(epochs)
        itpc = np.abs(np.mean(np.exp(1j * epochs), axis=0))

    # A final cycle onset precedes offset by a full cycle in standard logs.
    after_flash_start = flash_stop_eeg + AFTER_FLASH_DELAY_S
    after_flash_mask = (eeg_times > after_flash_start) & np.isfinite(filtered_eeg)
    f_on, pxx_on, on_windows = contiguous_welch(filtered_eeg, sfreq, psd_on_mask)
    f_after, pxx_after, after_windows = contiguous_welch(filtered_eeg, sfreq, after_flash_mask)
    plv_cycle_count = np.isfinite(cycle_plvs).sum()
    cycle_coverage = plv_cycle_count / len(cycle_plvs) if len(cycle_plvs) else 0.0
    if cycle_coverage < PLV_CYCLE_COVERAGE_TARGET:
        print(f"[{ch_name}][WARN] PLV available for {cycle_coverage:.1%} of flashing cycles "
              f"(target {PLV_CYCLE_COVERAGE_TARGET:.0%}); inspect {cycle_path}.")
    print(f"[{ch_name}] PSD: {on_windows} During Flash / {after_windows} After Flash "
          f"{PSD_WINDOW_S:g}s windows; PLV: {plv_cycle_count}/{len(cycle_plvs)} cycles; "
          f"ITPC: {itpc_epoch_count}/{len(flash_onsets)} complete epochs; "
          f"phase: {valid_mask.sum() / sfreq:.2f}s")

    def snr_at(f, pxx, target):
        # Do not interpret filter roll-off as reduced background noise. The
        # full +/- 2 Hz neighborhood must be inside the preprocessing band.
        if (f is None or target > f[-1] or target - 2 <= BANDPASS_LOW_HZ
                or target + 2 >= BANDPASS_HIGH_HZ):
            return np.nan
        idx = np.argmin(np.abs(f - target))
        band = (f > target - 2) & (f < target + 2) & (np.abs(f - target) > 0.5)
        noise = np.mean(pxx[band]) if band.any() else np.nan
        return pxx[idx] / noise if noise > 0 else np.nan

    snr_on_f0 = snr_at(f_on, pxx_on, flicker_freq)
    snr_on_2f0 = snr_at(f_on, pxx_on, 2 * flicker_freq)
    phase_low = flicker_freq - PHASE_HALF_BANDWIDTH_HZ
    phase_high = flicker_freq + PHASE_HALF_BANDWIDTH_HZ

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ref_info = f" (re-ref to {REFERENCE_CHANNEL})" if REFERENCE_CHANNEL else ""
    fig.suptitle(f"Channel {ch_name}{ref_info}  |  flicker {flicker_freq:.3f} Hz  |  "
                 f"overall PLV={overall_plv:.3f}  wrapped lag={mean_lag_ms:+.1f} ms")

    ax = axes[0]
    ax.bar(np.arange(1, len(cycle_plvs) + 1), cycle_plvs)
    ax.set_title(f"PLV per breath cycle ({plv_cycle_count}/{len(cycle_plvs)})")
    ax.set_xlabel("cycle #")
    ax.set_ylabel("PLV")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, len(cycle_plvs) + 0.5)

    ax = axes[1]
    if itpc is not None:
        ax.plot(itpc_t, itpc)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, "Fewer than 2 usable epochs", transform=ax.transAxes,
                ha="center")
    ax.set_title(f"Trigger-locked ITPC (n={itpc_epoch_count})")
    ax.set_xlabel("time from cycle onset (s)")
    ax.set_ylabel("ITPC")
    ax.set_ylim(0, 1)
    ax.set_xlim(EPOCH_TMIN, EPOCH_TMAX)

    ax = axes[2]
    if pxx_on is not None:
        display_band = (f_on >= BANDPASS_LOW_HZ) & (f_on <= BANDPASS_HIGH_HZ)
        ax.semilogy(f_on[display_band], pxx_on[display_band], label="During Flash")
    else:
        ax.text(0.03, 0.95, "During Flash: no full usable PSD windows",
                transform=ax.transAxes, va="top", fontsize=8)
    if pxx_after is not None:
        display_band = (f_after >= BANDPASS_LOW_HZ) & (f_after <= BANDPASS_HIGH_HZ)
        ax.semilogy(f_after[display_band], pxx_after[display_band], label="After Flash")
    else:
        ax.text(0.03, 0.87, "After Flash: no full usable PSD windows",
                transform=ax.transAxes, va="top", fontsize=8)
    ax.axvline(flicker_freq, color="r", linestyle="--", linewidth=1, label="f0")
    ax.axvline(2 * flicker_freq, color="r", linestyle=":", linewidth=1, label="2*f0")
    ax.set_xlim(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
    ax.set_title(f"PSD  (SNR ratios: f0={snr_on_f0:.2f}, 2f0={snr_on_2f0:.2f})")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Power (V²/Hz)")
    ax.legend(fontsize=8)

    fig.text(0.5, 0.015,
             f"Preprocessing / PSD: {NOTCH_FREQ_HZ:g} Hz notch + "
             f"{BANDPASS_LOW_HZ:g}-{BANDPASS_HIGH_HZ:g} Hz  |  "
             f"PLV / ITPC: {phase_low:.2f}-{phase_high:.2f} Hz\n"
             f"Usable phase: {valid_mask.sum() / sfreq:.1f}s  |  "
             f"{PSD_WINDOW_S:g}s PSD windows: During Flash {on_windows}, After Flash {after_windows}",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig_path = os.path.join(out_dir, f"{ch_name}_phase_locking.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    return {
        "channel": ch_name,
        "overall_plv": overall_plv,
        "mean_lag_ms": mean_lag_ms,
        "mean_cycle_plv": np.nanmean(cycle_plvs) if np.isfinite(cycle_plvs).any() else np.nan,
        "peak_itpc": np.max(itpc) if itpc is not None else np.nan,
        "snr_f0": snr_on_f0,
        "snr_2f0": snr_on_2f0,
        "notch_hz": NOTCH_FREQ_HZ,
        "bandpass_low_hz": BANDPASS_LOW_HZ,
        "bandpass_high_hz": BANDPASS_HIGH_HZ,
        "phase_low_hz": phase_low,
        "phase_high_hz": phase_high,
        "nonfinite_input_s": (~np.isfinite(eeg)).sum() / sfreq,
        "usable_phase_s": valid_mask.sum() / sfreq,
        "plv_cycles": plv_cycle_count,
        "flashing_cycles": len(cycle_plvs),
        "estimated_cycle_onsets": int(np.sum(~cycle_onsets_detected)),
        "plv_cycle_coverage": cycle_coverage,
        "itpc_epochs": itpc_epoch_count,
        "psd_during_flash_windows": on_windows,
        "psd_after_flash_windows": after_windows,
        "flash_stop_s": flash_stop_eeg,
        "after_flash_start_s": after_flash_start,
        "cycle_csv": cycle_path,
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
    print(f"[script] {os.path.abspath(__file__)} | alignment: protocol-anchored")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_df, flicker_freq, csv_trigger_times = load_stimulus_log(STIMULUS_CSV_PATH)
    raw = load_edf(EDF_PATH)

    # Detect DC7 triggers on the original, unmodified timeline.
    eeg_trigger_times, dc7_sig, sfreq = detect_dc7_triggers(raw)

    # 2. Re-reference O1 and O2 against Cz (O1 - Cz, O2 - Cz)
    raw = rereference_to_channel(raw, REREFERENCE_CHANNELS, REFERENCE_CHANNEL)

    protocol_stop_eeg = (find_protocol_flash_offset(eeg_trigger_times)
                         if FLASH_STOP_TRIGGER_NUMBER is None else None)
    csv_stop_time = (float(csv_df["global_time_s"].max())
                     if protocol_stop_eeg is not None else None)
    slope, intercept, matched_csv_indices, matched_eeg_trig, residuals = match_and_fit_time_mapping(
        csv_trigger_times, eeg_trigger_times,
        csv_stop_time=csv_stop_time, eeg_stop_time=protocol_stop_eeg)
    flash_df, flash_stop_eeg = flashing_period(
        csv_df, csv_trigger_times, eeg_trigger_times, matched_eeg_trig, slope, intercept,
        protocol_stop_eeg=protocol_stop_eeg)
    cycle_onsets_eeg, cycle_onsets_detected = cycle_timing(
        csv_trigger_times, matched_csv_indices, matched_eeg_trig, slope, intercept, flash_stop_eeg)
    alignment_max_ms = np.max(np.abs(residuals)) * 1000
    alignment_rms_ms = np.sqrt(np.mean(residuals ** 2)) * 1000
    alignment_phase_deg = alignment_max_ms / 1000 * flicker_freq * 360
    print(f"[align] maximum trigger-fit residual is equivalent to {alignment_phase_deg:.1f} "
          f"degrees at {flicker_freq:.3f} Hz; timing uncertainty limits interpretation "
          "of absolute phase and wrapped lag")

    # QC plot of the DC7 channel around the first few triggers
    fig, ax = plt.subplots(figsize=(10, 3))
    span = slice(0, min(len(dc7_sig), int(35 * sfreq)))
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
    print(f"[filters] {NOTCH_FREQ_HZ:g} Hz notch (Q={NOTCH_Q:g}), then "
          f"{BANDPASS_LOW_HZ:g}-{BANDPASS_HIGH_HZ:g} Hz band-pass; "
          "no amplitude-based rejection")
    print(f"[phase] additional {flicker_freq - PHASE_HALF_BANDWIDTH_HZ:.2f}-"
          f"{flicker_freq + PHASE_HALF_BANDWIDTH_HZ:.2f} Hz band-pass for EEG and "
          "reference before Hilbert; PSD uses the 3-30 Hz preprocessing output")

    results = []
    for ch in channels:
        r = analyze_channel(ch, raw, flash_df, flicker_freq, cycle_onsets_eeg,
                             cycle_onsets_detected, slope, intercept, OUTPUT_DIR, flash_stop_eeg)
        if r:
            r.update({"alignment_max_residual_ms": alignment_max_ms,
                      "alignment_rms_residual_ms": alignment_rms_ms,
                      "alignment_max_phase_equivalent_deg": alignment_phase_deg,
                      "flash_stop_source": ("post_flash_marker_sequence"
                                            if protocol_stop_eeg is not None else "offset_pulse"),
                      "first_cycle_start_s": cycle_onsets_eeg[0],
                      "recording_starts_during_cycle": bool(cycle_onsets_eeg[0] < 0)})
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
