#!/usr/bin/env python3
"""Calculate six HRV measures for four manually selected EDF windows.

Edit the four EDF paths and their TMIN/TMAX values below, then run this file.
Times are seconds from the beginning of each recording; each window must be
at least 45 seconds. The only saved output is summary.csv with four rows.

Requirements: numpy, pandas, scipy, mne, neurokit2.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

# EDIT YOUR FOUR FILES AND TIME WINDOWS HERE.
BEFORE_EXP_EDF = r"C:\Users\saiik\Downloads\analysis to do\SHEREYA_SUB26\SUB26~ SHREYA_85f31e46-e9ec-4e68-bf1d-f1d4ecf841b6.edf"
BEFORE_EXP_TMIN = 75
BEFORE_EXP_TMAX = 135

AFTER_EXP_EDF = r"C:\Users\saiik\Downloads\analysis to do\SHEREYA_SUB26\FLASHING\SUB26~ SHREYA_7f477fba-d6a0-48ec-b15f-e312289f993c.edf"
AFTER_EXP_TMIN = 306
AFTER_EXP_TMAX = 360

BEFORE_CONTROL_EDF = r"C:\Users\saiik\Downloads\analysis to do\SHEREYA_SUB26\control base\SUB26~ SHREYA_0c7881cd-5321-4918-82d4-bc47e9b9c6c6.edf"
BEFORE_CONTROL_TMIN = 75
BEFORE_CONTROL_TMAX = 135

AFTER_CONTROL_EDF = r"C:\Users\saiik\Downloads\analysis to do\SHEREYA_SUB26\control\SUB26~ SHREYA_7d964fcf-6ac0-4f43-9ca3-0a7214e99443.edf"
AFTER_CONTROL_TMIN = 313
AFTER_CONTROL_TMAX = 368

# Auto-detect a complete pair in each EDF, in this preference order.
# Matching ignores case and surrounding spaces: ekg/ekg' and ecgl/ecgr work too.
ECG_CHANNELS = [("EKG", "EKG'"), ("ECGL", "ECGR")]
ECG_CHANNEL = None  # None selects automatically; set a name to override.
ECG_REFERENCE_CHANNEL = None  # With an explicit ECG_CHANNEL, None means no subtraction.
ECG_POLARITY = "auto"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "hrv_results" / "four_windows_metrics"

FOUR_WINDOW_ORDER = ("before_exp", "after_exp", "before_control", "after_control")
WINDOW_LABELS = {"before_exp": "Before EXP", "after_exp": "After EXP",
                 "before_control": "Before CONTROL", "after_control": "After CONTROL"}
# Only these six measurements appear in the terminal and saved CSV.
METRIC_COLUMNS = {
    "ln_rmssd": "lnRMSSD",
    "mean_hr_bpm": "Mean heart rate (bpm)",
    "rmssd_ms": "RMSSD (ms)",
    "sdnn_ms": "SDNN (ms)",
    "hf_power_ms2": "HF power (ms^2)",
    "lf_hf_ratio": "LF/HF",
}


@dataclass(frozen=True)
class Config:
    """Internal calculation settings; no configuration file is required."""
    ecg_low_hz: float = 0.5
    ecg_high_hz: float = 40.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    filter_order: int = 4
    detector: str = "neurokit"
    edge_s: float = 1.0
    min_run_s: float = 5.0
    rr_min_ms: float = 300.0
    rr_max_ms: float = 2000.0
    local_window_beats: int = 11
    local_deviation_fraction: float = 0.30
    local_deviation_floor_ms: float = 200.0
    min_template_correlation: float = 0.80
    artifact_warning_fraction: float = 0.05
    artifact_failure_fraction: float = 0.20
    min_valid_fraction: float = 0.90
    min_nn_intervals: int = 30
    irregular_jump_fraction: float = 0.20
    irregular_window_beats: int = 30
    irregular_burden_fraction: float = 0.30
    tachogram_hz: float = 4.0
    spectrum_window_s: float = 40.0
    spectral_max_gap_s: float = 3.0
    spectral_max_gap_intervals: int = 2


def finite_runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def resolve_ecg_channels(channel_names, channel=None, reference=None):
    """Return actual EDF names; use complete configured pairs, never mixed leads."""
    def normalized(name):
        return str(name).strip().casefold().replace("’", "'").replace("′", "'")

    def find(name):
        matches = [actual for actual in channel_names if normalized(actual) == normalized(name)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous ECG channel {name!r}: {matches}")
        return matches[0] if matches else None

    if channel is not None:
        actual = find(channel)
        ref = find(reference) if reference is not None else None
        if actual is None or (reference is not None and ref is None):
            raise ValueError(f"Requested ECG channel/reference missing. Available: {channel_names}")
        if actual == ref:
            raise ValueError("ECG channel and reference must be different.")
        return actual, ref
    if reference is not None:
        raise ValueError("Set ECG_CHANNEL too when specifying ECG_REFERENCE_CHANNEL.")
    for left, right in ECG_CHANNELS:
        actual, ref = find(left), find(right)
        if actual is not None and ref is not None and actual != ref:
            return actual, ref
    raise ValueError(f"No complete ECG pair found. Expected one of {ECG_CHANNELS}. "
                     f"Available: {channel_names}")


def load_recording(session):
    """Load the ECG pair from the full recording before selecting the window."""
    path = Path(session["input"])
    reader = mne.io.read_raw_bdf if path.suffix.lower() == ".bdf" else mne.io.read_raw_edf
    with warnings.catch_warnings(record=True):
        raw = reader(str(path), preload=False, verbose="ERROR")
        try:
            channel, reference = resolve_ecg_channels(
                raw.ch_names, session["ecg_channel"], session["ecg_reference_channel"])
            wanted = [channel] + ([reference] if reference else [])
            raw.pick(wanted)
            raw.load_data(verbose="ERROR")
            ecg = raw.get_data(picks=[channel])[0]
            if reference:
                ecg = ecg - raw.get_data(picks=[reference])[0]
            sfreq = float(raw.info["sfreq"])
        finally:
            raw.close()
    return {"times": np.arange(len(ecg)) / sfreq, "sfreq": sfreq, "ecg": ecg}


def template_correlations(signal, peaks, sfreq):
    """Morphology similarity, not a calibrated probability of correct detection."""
    pre, post = int(round(.15 * sfreq)), int(round(.25 * sfreq))
    scores = np.full(len(peaks), np.nan)
    eligible = np.flatnonzero((peaks >= pre) & (peaks + post < len(signal)))
    if len(eligible) < 5:
        return scores
    waves = np.array([signal[peaks[k] - pre:peaks[k] + post] for k in eligible])
    waves -= waves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(waves, axis=1, keepdims=True)
    normalized = np.divide(waves, norms, out=np.zeros_like(waves), where=norms > 0)
    template = np.median(normalized, axis=0)
    norm = np.linalg.norm(template)
    if norm > 0:
        scores[eligible] = np.clip(normalized @ (template / norm), -1, 1)
    return scores


def detect_beats(recording, cfg, polarity="auto"):
    sfreq, ecg, times = recording["sfreq"], recording["ecg"], recording["times"]
    if sfreq < 100 or cfg.ecg_high_hz >= sfreq / 2:
        raise ValueError("ECG requires at least 100 Hz and filter cutoffs below Nyquist; 250+ Hz is preferred.")
    if polarity not in {"auto", "positive", "negative"}:
        raise ValueError("ecg_polarity must be auto, positive or negative.")
    band = butter(cfg.filter_order, [cfg.ecg_low_hz, cfg.ecg_high_hz], fs=sfreq,
                  btype="bandpass", output="sos")
    filtered = np.full(len(ecg), np.nan)
    rows, logs = [], []
    for run_id, (start, stop) in enumerate(finite_runs(np.isfinite(ecg))):
        if stop - start < max(int(cfg.min_run_s * sfreq), 3 * (2 * len(band) + 1) + 1):
            logs.append({"run": run_id, "status": "too_short", "start_s": times[start]})
            continue
        x = ecg[start:stop].copy()
        if 0 < cfg.notch_hz < sfreq / 2:
            b, a = iirnotch(cfg.notch_hz, cfg.notch_q, fs=sfreq)
            x = sosfiltfilt(tf2sos(b, a), x)
        x = sosfiltfilt(band, x)
        center = np.median(x)
        sign = -1 if polarity == "negative" else 1
        if polarity == "auto":
            sign = -1 if center - np.percentile(x, .5) > np.percentile(x, 99.5) - center else 1
        filtered[start:stop] = sign * x
        if np.ptp(x) == 0:
            logs.append({"run": run_id, "status": "no_signal", "polarity": sign})
            continue
        with warnings.catch_warnings(record=True) as captured:
            _, info = nk.ecg_peaks(sign * x, sampling_rate=sfreq,
                                   method=cfg.detector, correct_artifacts=False)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        edge = int(round(cfg.edge_s * sfreq))
        peaks = peaks[(peaks >= edge) & (peaks < len(x) - edge)]
        scores = template_correlations(sign * x, peaks, sfreq)
        for p, score in zip(peaks, scores):
            rows.append({"sample_index": int(start + p), "time_s": float(times[start + p]),
                         "run_id": run_id, "template_correlation": score, "polarity": sign})
        logs.append({"run": run_id, "status": "processed", "polarity": sign,
                     "peaks": len(peaks), "warnings": [str(w.message) for w in captured]})
    columns = ["sample_index", "time_s", "run_id", "template_correlation", "polarity"]
    beats = pd.DataFrame(rows, columns=columns)
    beats.insert(0, "beat_id", np.arange(len(beats)))
    return beats, filtered, logs


def classify_intervals(beats, cfg):
    columns = ["interval_id", "left_beat_id", "right_beat_id", "start_s", "end_s", "rr_ms",
               "run_id", "local_median_ms", "accepted", "reasons"]
    if len(beats) < 2:
        return pd.DataFrame(columns=columns)
    rr = np.diff(beats.time_s.to_numpy()) * 1000
    left_run = beats.run_id.to_numpy()[:-1]
    same_run = left_run == beats.run_id.to_numpy()[1:]
    physiological = (rr >= cfg.rr_min_ms) & (rr <= cfg.rr_max_ms) & same_run
    medians = np.full(len(rr), np.nan)
    half = cfg.local_window_beats // 2
    for i in range(len(rr)):
        a, b = max(0, i - half), min(len(rr), i + half + 1)
        candidates = rr[a:b][physiological[a:b] & (left_run[a:b] == left_run[i])]
        if len(candidates) >= 3:
            medians[i] = np.median(candidates)
    morphology = beats.template_correlation.to_numpy()
    rows = []
    for i, value in enumerate(rr):
        reasons = []
        if not same_run[i]:
            reasons.append("signal_gap")
        if value < cfg.rr_min_ms or value > cfg.rr_max_ms:
            reasons.append("rr_out_of_range")
        if (np.isfinite(medians[i]) and abs(value - medians[i]) >
                max(cfg.local_deviation_floor_ms, cfg.local_deviation_fraction * medians[i])):
            reasons.append("local_rr_deviation")
        endpoint_scores = morphology[i:i + 2]
        if not np.isfinite(endpoint_scores).all():
            reasons.append("morphology_unavailable")
        elif np.min(endpoint_scores) < cfg.min_template_correlation:
            reasons.append("low_template_correlation")
        rows.append({"interval_id": i, "left_beat_id": i, "right_beat_id": i + 1,
                     "start_s": beats.time_s.iloc[i], "end_s": beats.time_s.iloc[i + 1],
                     "rr_ms": value, "run_id": left_run[i], "local_median_ms": medians[i],
                     "accepted": not reasons, "reasons": ";".join(reasons)})
    return pd.DataFrame(rows, columns=columns)


def time_domain_metrics(rr_ms, accepted=None, interval_ids=None):
    """RMSSD uses original adjacent accepted intervals; deletion never bridges a gap."""
    rr = np.asarray(rr_ms, dtype=float)
    valid = np.isfinite(rr) & (rr > 0)
    if accepted is not None:
        valid &= np.asarray(accepted, dtype=bool)
    ids = np.arange(len(rr)) if interval_ids is None else np.asarray(interval_ids)
    adjacent = valid[:-1] & valid[1:] & (np.diff(ids) == 1)
    diffs = np.diff(rr)[adjacent]
    nn = rr[valid]
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else np.nan
    # Equal sample spacings can differ by roundoff after subtracting timestamps.
    numerical_floor_ms = max(1e-9, 32 * np.finfo(float).eps * np.mean(nn)) if len(nn) else 1e-9
    if rmssd <= numerical_floor_ms:
        rmssd = 0.0
    sdnn = float(np.std(nn, ddof=1)) if len(nn) > 1 else np.nan
    if sdnn <= numerical_floor_ms:
        sdnn = 0.0
    return {"mean_nn_ms": float(np.mean(nn)) if len(nn) else np.nan,
            "mean_hr_bpm": float(60000 / np.mean(nn)) if len(nn) else np.nan,
            "rmssd_ms": rmssd, "ln_rmssd": float(np.log(rmssd)) if rmssd > 0 else np.nan,
            "sdnn_ms": sdnn,
            "accepted_intervals": len(nn), "rmssd_pairs": int(adjacent.sum()),
            "valid_seconds": float(nn.sum() / 1000)}


def irregular_pattern(intervals, cfg):
    """Conservative review flag for sustained abrupt RR changes; not AF diagnosis."""
    size = cfg.irregular_window_beats
    if len(intervals) < size:
        return False
    rr = intervals.rr_ms.to_numpy(dtype=float)
    for start in range(len(rr) - size + 1):
        part = intervals.iloc[start:start + size]
        if part.run_id.nunique() != 1 or not np.all(np.diff(part.interval_id) == 1):
            continue
        values = rr[start:start + size]
        scale = np.median(values)
        if scale > 0 and np.mean(abs(np.diff(values)) > cfg.irregular_jump_fraction * scale) >= cfg.irregular_burden_fraction:
            return True
    return False


def integrate_band(f, power, low, high):
    if f is None or low < f[0] or high > f[-1] or low >= high:
        return np.nan
    inside = (f > low) & (f < high)
    grid = np.r_[low, f[inside], high]
    values = np.r_[np.interp(low, f, power), power[inside], np.interp(high, f, power)]
    return float(np.trapezoid(values, grid))


def spectral_intervals(intervals, cfg):
    """Interpolate isolated excluded intervals for spectra only, with an audit flag.

    No extrapolation, no signal-gap reconstruction, at most two intervals and
    three seconds by default. Time-domain NN metrics always use the originals.
    """
    data = intervals.copy()
    data["spectral_interpolated"] = False
    if len(data) == 0 or cfg.spectral_max_gap_s == 0:
        return data
    valid = data.accepted.to_numpy(dtype=bool)
    rr = data.rr_ms.to_numpy(dtype=float).copy()
    anchors = (data.start_s.to_numpy(dtype=float) + data.end_s.to_numpy(dtype=float)) / 2
    for a, b in finite_runs(~valid):
        if a == 0 or b == len(data) or b - a > cfg.spectral_max_gap_intervals:
            continue
        section = data.iloc[a - 1:b + 1]
        if (section.run_id.nunique() != 1 or not np.all(np.diff(section.interval_id) == 1)
                or data.end_s.iloc[b - 1] - data.start_s.iloc[a] > cfg.spectral_max_gap_s
                or section.reasons.str.contains("signal_gap", regex=False).any()):
            continue
        rr[a:b] = np.interp(anchors[a:b], anchors[[a - 1, b]], rr[[a - 1, b]])
        data.iloc[a:b, data.columns.get_loc("spectral_interpolated")] = True
    data["rr_ms"] = rr
    data["accepted"] = valid | data.spectral_interpolated.to_numpy(dtype=bool)
    return data


def hrv_spectrum(intervals, cfg):
    """Compute HF and LF/HF; LF and the PSD remain intermediate calculations."""
    intervals = spectral_intervals(intervals, cfg)
    fs = cfg.tachogram_hz
    win = int(round(cfg.spectrum_window_s * fs))
    overlap = win // 2
    accumulated, count, f = None, 0, None
    accepted = intervals.accepted.to_numpy(dtype=bool)
    starts = []
    for a, b in finite_runs(accepted):
        part = intervals.iloc[a:b]
        # Never join intervals across missing IDs or separate ECG sections.
        broken = ((np.diff(part.interval_id.to_numpy()) != 1)
                  | (np.diff(part.run_id.to_numpy()) != 0))
        splits = np.r_[a, a + 1 + np.flatnonzero(broken), b]
        starts.extend(zip(splits[:-1], splits[1:]))
    for a, b in starts:
        part = intervals.iloc[a:b]
        if len(part) < 3:
            continue
        anchors = (part.start_s.to_numpy() + part.end_s.to_numpy()) / 2
        grid = np.arange(np.ceil(anchors[0] * fs), np.floor(anchors[-1] * fs) + 1) / fs
        if len(grid) < win:
            continue
        nn = np.interp(grid, anchors, part.rr_ms.to_numpy())
        f, power = welch(nn, fs=fs, window="hann", nperseg=win,
                         noverlap=overlap, detrend="constant")
        nwin = 1 + (len(nn) - win) // (win - overlap)
        accumulated = power * nwin if accumulated is None else accumulated + power * nwin
        count += nwin
    if count == 0:
        return {"hf_power_ms2": np.nan, "lf_hf_ratio": np.nan}
    power = accumulated / count
    lf = integrate_band(f, power, .04, .15)
    hf = integrate_band(f, power, .15, .40)
    return {"hf_power_ms2": hf, "lf_hf_ratio": lf / hf if hf > 0 else np.nan}


def manual_inputs():
    """Read editable variables at call time, so importing never runs an analysis."""
    return {
        "before_exp": {"path": BEFORE_EXP_EDF, "tmin": BEFORE_EXP_TMIN, "tmax": BEFORE_EXP_TMAX},
        "after_exp": {"path": AFTER_EXP_EDF, "tmin": AFTER_EXP_TMIN, "tmax": AFTER_EXP_TMAX},
        "before_control": {"path": BEFORE_CONTROL_EDF, "tmin": BEFORE_CONTROL_TMIN, "tmax": BEFORE_CONTROL_TMAX},
        "after_control": {"path": AFTER_CONTROL_EDF, "tmin": AFTER_CONTROL_TMIN, "tmax": AFTER_CONTROL_TMAX},
    }


def prepare_four_windows(inputs):
    """Check all four paths, times, and channels before writing any results."""
    if set(inputs) != set(FOUR_WINDOW_ORDER):
        raise ValueError("Supply Before EXP, After EXP, Before CONTROL, and After CONTROL.")
    sessions, errors, paths = [], [], []
    for key in FOUR_WINDOW_ORDER:
        entry = inputs[key]
        label = WINDOW_LABELS[key]
        filename = str(entry.get("path", "")).strip()
        if not filename:
            errors.append(f"{label}: enter the EDF path at the top of this file.")
            continue
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        path = path.resolve()
        if path.suffix.lower() not in {".edf", ".bdf"} or not path.is_file():
            errors.append(f"{label}: EDF/BDF file not found: {path}")
            continue
        try:
            start, end = float(entry["tmin"]), float(entry["tmax"])
        except (TypeError, ValueError, KeyError):
            errors.append(f"{label}: TMIN and TMAX must be numbers in seconds.")
            continue
        if not np.isfinite([start, end]).all() or start < 0 or end - start < 45 - 1e-6:
            errors.append(f"{label}: TMIN must be at least 0 and TMAX - TMIN must be at least 45 seconds.")
            continue
        channel = entry.get("ecg_channel", ECG_CHANNEL)
        reference = entry.get("ecg_reference_channel", ECG_REFERENCE_CHANNEL)
        polarity = entry.get("ecg_polarity", ECG_POLARITY)
        if polarity not in {"auto", "positive", "negative"}:
            errors.append(f"{label}: ECG_POLARITY must be auto, positive, or negative.")
            continue
        reader = mne.io.read_raw_bdf if path.suffix.lower() == ".bdf" else mne.io.read_raw_edf
        try:
            with warnings.catch_warnings(record=True):
                raw = reader(str(path), preload=False, verbose="ERROR")
                try:
                    duration = raw.n_times / raw.info["sfreq"]
                    if end > duration + 1e-6:
                        raise ValueError(f"TMAX {end:g} exceeds the recording length of {duration:g} seconds.")
                    if raw.info["sfreq"] < 100:
                        raise ValueError("The ECG recording must have at least 100 samples per second.")
                    channel, reference = resolve_ecg_channels(raw.ch_names, channel, reference)
                finally:
                    raw.close()
        except (ValueError, OSError) as error:
            errors.append(f"{label}: {error}")
            continue
        paths.append(path)
        sessions.append({"input": str(path), "session_id": key, "start_s": start, "end_s": end,
                         "ecg_channel": channel, "ecg_reference_channel": reference,
                         "ecg_polarity": polarity})
    if len(paths) != len(set(paths)):
        errors.append("Use four different EDF files; a path is repeated in the input variables.")
    if errors:
        raise ValueError("Fix the input variables:\n  " + "\n  ".join(errors))
    return sessions


def analyze_window(recording, session, cfg):
    """Keep quality checks internal and return only the six requested measures."""
    label = WINDOW_LABELS[session["session_id"]]
    beats, _, _ = detect_beats(recording, cfg, session["ecg_polarity"])
    intervals = classify_intervals(beats, cfg)
    start, end = session["start_s"], session["end_s"]
    selected = intervals.loc[(intervals.start_s >= start) & (intervals.end_s < end)].copy()
    accepted = selected.accepted.to_numpy(dtype=bool)
    metrics = time_domain_metrics(selected.rr_ms.to_numpy(dtype=float), accepted,
                                  selected.interval_id.to_numpy())
    excluded_fraction = float((~accepted).mean()) if len(selected) else 1.0
    if (metrics["accepted_intervals"] < cfg.min_nn_intervals
            or metrics["rmssd_pairs"] < cfg.min_nn_intervals - 1
            or metrics["valid_seconds"] / (end - start) < cfg.min_valid_fraction
            or excluded_fraction > cfg.artifact_failure_fraction):
        raise ValueError(f"{label}: not enough reliable heartbeat data. Choose a cleaner or longer window.")
    if excluded_fraction > cfg.artifact_warning_fraction or irregular_pattern(selected, cfg):
        print(f"{label}: some heartbeats need review; check this recording before using the results.",
              file=sys.stderr)
    metrics.update(hrv_spectrum(selected, cfg))
    missing = [name for key, name in METRIC_COLUMNS.items() if not np.isfinite(metrics[key])]
    if missing:
        print(f"{WINDOW_LABELS[session['session_id']]}: could not calculate {', '.join(missing)}; shown as N/A.",
              file=sys.stderr)
    return {key: metrics[key] for key in METRIC_COLUMNS}


def run_four_windows(inputs=None, output=None):
    sessions = prepare_four_windows(manual_inputs() if inputs is None else inputs)
    cfg = Config()
    rows = []
    for session in sessions:
        recording = load_recording(session)
        metrics = analyze_window(recording, session, cfg)
        rows.append({"Window": WINDOW_LABELS[session["session_id"]],
                     **{label: metrics[key] for key, label in METRIC_COLUMNS.items()}})
    table = pd.DataFrame(rows, columns=["Window", *METRIC_COLUMNS.values()])
    # A separate folder keeps these results apart from earlier detailed exports.
    output = Path(OUTPUT_DIR if output is None else output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "summary.csv"
    temporary = output / "summary.csv.tmp"
    try:
        table.to_csv(temporary, index=False, na_rep="N/A")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}", na_rep="N/A"))
    print(f"\nSaved: {destination}")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calculate the six HRV measures for your four EDF windows.")
    parser.add_argument("--output", type=Path, default=None, help="Folder for the single summary.csv file")
    parser.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        run_four_windows(output=args.output)
    except (ValueError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
