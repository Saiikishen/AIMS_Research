#!/usr/bin/env python3


from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import warnings


PARTICIPANT_ID = "jacob"

BEFORE_EXP_EDF = r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\SUB15~ JACOB_10be020a-ff91-4adc-810c-1c1157b51507.edf" 
BEFORE_EXP_TMIN = 75
BEFORE_EXP_TMAX = 135

AFTER_EXP_EDF = r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\flashing.edf"
AFTER_EXP_TMIN = 306
AFTER_EXP_TMAX = 360

BEFORE_CONTROL_EDF = r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\CONTROL BASE\SUB15~ JACOB_256cd2b6-b9c1-4b38-b12a-9b2509cd444d.edf"
BEFORE_CONTROL_TMIN = 75
BEFORE_CONTROL_TMAX = 135

AFTER_CONTROL_EDF = r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\CONTROL\SUB15~ JACOB_0e04f0e8-fb17-4f50-8a72-31c9e4d27116.edf"
AFTER_CONTROL_TMIN = 337
AFTER_CONTROL_TMAX = 390

ECG_CHANNEL = "ECGL"
ECG_REFERENCE_CHANNEL = "ECGR"  
RESP_CHANNEL = None  
ECG_POLARITY = "auto"  
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "hrv_results" / "four_windows"


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, iirnotch, sosfiltfilt, tf2sos, welch

PROCESSING_VERSION = "1.2.0"
FOUR_WINDOW_ORDER = ("before_exp", "after_exp", "before_control", "after_control")
WINDOW_LABELS = {"before_exp": "Before EXP", "after_exp": "After EXP",
                 "before_control": "Before CONTROL", "after_control": "After CONTROL"}
COMPARISON_METRICS = {
    "rmssd_ms": ("RMSSD", "ms"), "ln_rmssd": ("lnRMSSD", "ln(ms)"),
    "mean_hr_bpm": ("Mean heart rate", "bpm"), "sdnn_ms": ("SDNN", "ms"),
    "lf_power_ms2": ("LF power", "ms²"), "hf_power_ms2": ("HF power", "ms²"),
    "lf_hf_ratio": ("LF/HF", "ratio"),
}
SHORT_WINDOW_NOTE = (
    "Manually selected intervals of at least 45 seconds. LF 0.04–0.15 Hz; HF 0.15–0.40 Hz. "
    "PSD uses identical 40-second Hann windows with 50% overlap (0.025 Hz bin spacing); "
    "longer usable intervals contribute additional spectral windows when available. "
    "LF and LF/HF remain exploratory at this spectral resolution. Unequal durations are flagged. "
    "Higher HRV alone does not establish attention or relaxation; LF/HF is descriptive."
)
SEGMENTS = {"pre_rest", "pre_paced", "during", "post_rest", "delayed_post"}
BREATHING_MODES = {"spontaneous", "paced", "unknown"}


@dataclass(frozen=True)
class Config:
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
    primary_duration_s: float = 300.0
    duration_tolerance_s: float = 5.0
    min_nn_intervals: int = 30
    irregular_jump_fraction: float = 0.20
    irregular_window_beats: int = 30
    irregular_burden_fraction: float = 0.30
    resp_low_hz: float = 0.05
    resp_high_hz: float = 0.50
    resp_min_s: float = 60.0
    resp_peak_fraction: float = 0.40
    resp_tolerance_bpm: float = 0.5
    expected_paced_bpm: float = 6.0
    tachogram_hz: float = 4.0
    spectrum_window_s: float = 120.0
    spectral_max_gap_s: float = 3.0
    spectral_max_gap_intervals: int = 2
    resp_half_band_hz: float = 0.03
    min_calibration_recordings: int = 5

    def validate(self):
        for setting in fields(self):
            value = getattr(self, setting.name)
            if isinstance(setting.default, int):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{setting.name} must be an integer.")
            elif isinstance(setting.default, float):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"{setting.name} must be numeric.")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"{setting.name} must be a nonempty string.")
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError("All numeric processing settings must be finite.")
        if not 0 < self.ecg_low_hz < self.ecg_high_hz:
            raise ValueError("Invalid ECG filter band.")
        if self.notch_hz < 0 or self.notch_q <= 0 or self.filter_order < 1:
            raise ValueError("Invalid notch or filter-order setting.")
        if not 0 < self.rr_min_ms < self.rr_max_ms:
            raise ValueError("Invalid RR interval range.")
        if not 0 <= self.artifact_warning_fraction < self.artifact_failure_fraction <= 1:
            raise ValueError("Artifact warning/failure fractions must be ordered in [0,1].")
        if not 0 < self.min_valid_fraction <= 1 or not -1 <= self.min_template_correlation <= 1:
            raise ValueError("Invalid valid-data fraction or template correlation.")
        if self.edge_s < 0 or self.min_run_s <= 2 * self.edge_s:
            raise ValueError("Minimum run length must exceed both filter edges.")
        if self.local_window_beats < 3 or self.local_window_beats % 2 != 1:
            raise ValueError("local_window_beats must be odd and at least 3.")
        if (self.local_deviation_fraction <= 0 or self.local_deviation_floor_ms < 0
                or self.min_nn_intervals < 3 or self.irregular_window_beats < 3
                or not 0 < self.irregular_jump_fraction < 1
                or not 0 < self.irregular_burden_fraction <= 1):
            raise ValueError("Invalid RR quality settings.")
        if not 0 < self.resp_low_hz < self.resp_high_hz < self.tachogram_hz / 2:
            raise ValueError("Invalid respiration band or tachogram sampling rate.")
        if (self.primary_duration_s <= 0 or self.duration_tolerance_s < 0
                or self.spectrum_window_s < 30 or self.resp_min_s <= 0
                or self.spectral_max_gap_s < 0 or self.spectral_max_gap_intervals < 0
                or self.resp_tolerance_bpm < 0 or self.resp_half_band_hz <= 0
                or self.expected_paced_bpm <= 0 or not 0 < self.resp_peak_fraction <= 1
                or self.min_calibration_recordings < 5):
            raise ValueError("Invalid duration, respiration, or calibration settings.")


def finite_runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def clean_json(value):
    """Produce strict JSON: missing numerical estimates become null, never NaN."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value):
    payload = json.dumps(clean_json(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_times(times, claimed_sfreq=None):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all():
        raise ValueError("At least three finite timestamps are required.")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Timestamps must be strictly increasing; duplicates are not allowed.")
    sfreq = float(claimed_sfreq) if claimed_sfreq is not None else 1 / np.median(dt)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Invalid sampling rate.")
    steps = np.rint(dt * sfreq).astype(int)
    grid_positions = (times - times[0]) * sfreq
    if (np.any(steps < 1) or np.max(abs(dt * sfreq - steps)) > 0.02
            or np.max(abs(grid_positions - np.rint(grid_positions))) > 0.02):
        raise ValueError("ECG timestamps must lie on a regular sampling grid. "
                         "Represent missing samples as gaps; do not silently resample irregular ECG.")
    return sfreq


def load_recording(session, root):
    path = Path(session["input"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    flags = []
    if path.suffix.lower() in {".edf", ".bdf"}:
        reader = mne.io.read_raw_bdf if path.suffix.lower() == ".bdf" else mne.io.read_raw_edf
        channel = session.get("ecg_channel")
        if not channel:
            raise ValueError("Set ecg_channel explicitly; use the inspect command to list EDF channels.")
        with warnings.catch_warnings(record=True) as captured:
            raw = reader(str(path), preload=False, verbose="ERROR")
            wanted = [channel] + [session[k] for k in ("ecg_reference_channel", "resp_channel") if session.get(k)]
            if any(ch not in raw.ch_names for ch in wanted):
                raise ValueError(f"Requested channel missing. Available: {raw.ch_names}")
            raw.pick(list(dict.fromkeys(wanted)))
            raw.load_data()
        flags.extend(str(w.message) for w in captured)
        ecg = raw.get_data(picks=[channel])[0]
        reference = session.get("ecg_reference_channel")
        if reference:
            if reference == channel:
                raise ValueError("ECG channel and ECG reference channel must be different.")
            ecg = ecg - raw.get_data(picks=[reference])[0]
        resp = raw.get_data(picks=[session["resp_channel"]])[0] if session.get("resp_channel") else None
        sfreq = float(raw.info["sfreq"])
        times = np.arange(len(ecg)) / sfreq
        annotations = [{"start_s": float(a["onset"]), "duration_s": float(a["duration"]),
                        "description": str(a["description"])} for a in raw.annotations]
        input_units = "V (MNE-scaled ECG)"
    elif path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if not {"timestamp_s", "ecg_value"} <= set(table):
            raise ValueError("CSV requires timestamp_s and ecg_value columns. PPG is not ECG.")
        t = pd.to_numeric(table.timestamp_s, errors="raise").to_numpy(dtype=float)
        sfreq = validate_times(t, session.get("sampling_rate_hz"))
        indices = np.rint((t - t[0]) * sfreq).astype(int)
        if indices[-1] > max(100 * len(t), 20_000_000):
            raise ValueError("Timestamp gap is too large to represent safely as one recording.")
        times = t[0] + np.arange(indices[-1] + 1) / sfreq
        ecg = np.full(len(times), np.nan)
        ecg[indices] = pd.to_numeric(table.ecg_value, errors="raise").to_numpy(dtype=float)
        resp = None
        if "resp_value" in table and table.resp_value.notna().any():
            resp = np.full(len(times), np.nan)
            resp[indices] = pd.to_numeric(table.resp_value, errors="raise").to_numpy(dtype=float)
        for key in ("participant_id", "session_id", "condition"):
            if key in table:
                values = table[key].dropna().astype(str).unique()
                if len(values) != 1 or values[0] != str(session[key]):
                    raise ValueError(f"CSV {key} does not agree with the session manifest.")
        if "segment" in table:
            for segment in session.get("segments", []):
                inside = (t >= segment["start_s"]) & (t < segment["end_s"])
                labels = table.loc[inside, "segment"].dropna().astype(str).unique()
                if len(labels) and (len(labels) != 1 or labels[0] != segment["name"]):
                    raise ValueError(f"CSV segment labels disagree with the supplied {segment['name']} boundaries.")
        annotations = []
        if "event_code" in table:
            annotations = [{"start_s": float(t[i]), "duration_s": 0.0, "description": str(v)}
                           for i, v in enumerate(table.event_code) if pd.notna(v)]
        input_units = session.get("ecg_units", "arbitrary input units")
    else:
        raise ValueError("Supported raw ECG formats are EDF, BDF and timestamped CSV.")
    validate_times(times, sfreq)
    return {"path": path, "times": times, "sfreq": sfreq, "ecg": ecg, "resp": resp,
            "annotations": annotations, "units": input_units, "load_flags": flags}


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


def respiration_metrics(times, resp, start_s, end_s, sfreq, cfg, cue_times=None):
    result = {"resp_rate_bpm": np.nan, "resp_frequency_hz": np.nan,
              "resp_source": "unavailable", "resp_status": "unavailable",
              "resp_valid_seconds": 0.0, "resp_peak_fraction": np.nan,
              "cue_rate_bpm": np.nan}
    if cue_times is not None:
        cues = np.asarray(cue_times, dtype=float)
        if not np.isfinite(cues).all() or np.any(np.diff(cues) <= 0):
            raise ValueError("Breathing cue timestamps must be finite and strictly increasing.")
        cues = cues[(cues >= start_s) & (cues < end_s)]
        if len(cues) >= 3:
            result["cue_rate_bpm"] = float(60 / np.median(np.diff(cues)))
            result["resp_source"] = "cue_only"
    if resp is None:
        return result
    mask = (times >= start_s) & (times < end_s) & np.isfinite(resp)
    band = butter(2, [cfg.resp_low_hz, cfg.resp_high_hz], btype="bandpass", fs=sfreq, output="sos")
    win = int(round(cfg.resp_min_s * sfreq))
    overlap = win // 2
    accumulated, count, seconds = None, 0, 0.0
    for a, b in finite_runs(mask):
        if b - a < win or np.ptp(resp[a:b]) == 0:
            continue
        x = sosfiltfilt(band, resp[a:b])
        f, p = welch(x, fs=sfreq, nperseg=win, noverlap=overlap, window="hann", detrend="constant")
        nwin = 1 + (b - a - win) // (win - overlap)
        accumulated = p * nwin if accumulated is None else accumulated + p * nwin
        count += nwin
        seconds += (b - a) / sfreq
    result.update(resp_source="measured", resp_valid_seconds=seconds, resp_status="insufficient_data")
    if not count:
        return result
    power = accumulated / count
    idxs = np.flatnonzero((f >= cfg.resp_low_hz) & (f <= cfg.resp_high_hz))
    peak = idxs[np.argmax(power[idxs])]
    frequency = f[peak]
    # Quadratic refinement of the spectral peak; no use of the pacing target.
    if 0 < peak < len(f) - 1 and np.all(power[peak - 1:peak + 2] > 0):
        a, b, c = np.log(power[peak - 1:peak + 2])
        denominator = a - 2 * b + c
        if denominator != 0:
            frequency += np.clip(.5 * (a - c) / denominator, -.5, .5) * (f[1] - f[0])
    total = integrate_band(f, power, cfg.resp_low_hz, cfg.resp_high_hz)
    around = integrate_band(f, power, max(cfg.resp_low_hz, frequency - cfg.resp_half_band_hz),
                            min(cfg.resp_high_hz, frequency + cfg.resp_half_band_hz))
    fraction = around / total if total > 0 else np.nan
    status = "valid" if fraction >= cfg.resp_peak_fraction and seconds / (end_s - start_s) >= cfg.min_valid_fraction else "review_required"
    if frequency <= cfg.resp_low_hz + .005 or frequency >= cfg.resp_high_hz - .005:
        status = "review_required"
    result.update(resp_rate_bpm=float(frequency * 60), resp_frequency_hz=float(frequency),
                  resp_peak_fraction=fraction, resp_status=status)
    return result


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


def hrv_spectrum(intervals, times, resp, resp_info, cfg):
    """Regularize the spectrum-only NN series on its original clock, then Welch."""
    intervals = spectral_intervals(intervals, cfg)
    fs = cfg.tachogram_hz
    win = int(round(cfg.spectrum_window_s * fs))
    overlap = win // 2
    accumulated, count, coh_sum, coh_count = None, 0, None, 0
    f = None
    accepted = intervals.accepted.to_numpy(dtype=bool)
    # Missing interval IDs also break runs (e.g. explicit segment/gap boundaries).
    starts = []
    for a, b in finite_runs(accepted):
        ids = intervals.interval_id.to_numpy()[a:b]
        splits = np.r_[a, a + 1 + np.flatnonzero(np.diff(ids) != 1), b]
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
        f, p = welch(nn, fs=fs, window="hann", nperseg=win, noverlap=overlap, detrend="constant")
        nwin = 1 + (len(nn) - win) // (win - overlap)
        accumulated = p * nwin if accumulated is None else accumulated + p * nwin
        count += nwin
        if nwin >= 3 and resp is not None and resp_info["resp_status"] == "valid":
            indices = (times >= grid[0]) & (times <= grid[-1])
            # No interpolation across a missing respiratory section.
            if indices.sum() >= 2 and np.isfinite(resp[indices]).all():
                r = np.interp(grid, times, resp)
                if np.ptp(nn) > 0 and np.ptp(r) > 0:
                    # Zero power at individual frequencies makes coherence undefined.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _, cxy = coherence(nn, r, fs=fs, window="hann", nperseg=win,
                                           noverlap=overlap, detrend="constant")
                    coh_sum = cxy * nwin if coh_sum is None else coh_sum + cxy * nwin
                    coh_count += nwin
    result = {"lf_power_ms2": np.nan, "hf_power_ms2": np.nan, "lf_hf_ratio": np.nan,
              "resp_centered_power": np.nan, "resp_hr_coherence": np.nan,
              "spectral_interpolated_intervals": int(intervals.spectral_interpolated.sum()),
              "spectral_interpolated_fraction": float(intervals.spectral_interpolated.mean()) if len(intervals) else 0.0,
              "spectral_windows": count, "spectral_status": "insufficient_contiguous_data"}
    if count == 0:
        return result, None
    power = accumulated / count
    lf = integrate_band(f, power, .04, .15)
    hf = integrate_band(f, power, .15, .40)
    result.update(lf_power_ms2=lf, hf_power_ms2=hf, lf_hf_ratio=lf / hf if hf > 0 else np.nan,
                  spectral_status="available")
    rate = resp_info["resp_frequency_hz"]
    if resp_info["resp_status"] == "valid" and np.isfinite(rate):
        result["resp_centered_power"] = integrate_band(f, power, max(0, rate - cfg.resp_half_band_hz), rate + cfg.resp_half_band_hz)
        if coh_count:
            result["resp_hr_coherence"] = float(np.interp(rate, f, coh_sum / coh_count))
    spectrum = pd.DataFrame({"frequency_hz": f, "power_ms2_per_hz": power})
    return result, spectrum


def analyze_segment(recording, beats, intervals, session, segment, cfg, settings_id):
    start, end = float(segment["start_s"]), float(segment["end_s"])
    selected = intervals.loc[(intervals.start_s >= start) & (intervals.end_s < end)].copy()
    accepted = selected.accepted.to_numpy(dtype=bool)
    metrics = time_domain_metrics(selected.rr_ms.to_numpy(dtype=float), accepted,
                                  selected.interval_id.to_numpy())
    flags, failure, review = [], False, False
    duration = end - start
    excluded = int((~accepted).sum())
    excluded_fraction = excluded / len(selected) if len(selected) else 1.0
    valid_fraction = metrics["valid_seconds"] / duration
    sample_mask = (recording["times"] >= start) & (recording["times"] < end)
    missing_seconds = np.sum(sample_mask & ~np.isfinite(recording["ecg"])) / recording["sfreq"]
    if duration + 1 / recording["sfreq"] < cfg.primary_duration_s:
        flags.append("shorter_than_primary_segment"); failure = True
    if (metrics["accepted_intervals"] < cfg.min_nn_intervals
            or metrics["rmssd_pairs"] < cfg.min_nn_intervals - 1):
        flags.append("too_few_nn_intervals_or_adjacent_pairs"); failure = True
    if valid_fraction < cfg.min_valid_fraction:
        flags.append("insufficient_valid_duration"); failure = True
    if excluded_fraction > cfg.artifact_failure_fraction:
        flags.append("excessive_interval_exclusions"); failure = True
    elif excluded_fraction > cfg.artifact_warning_fraction:
        flags.append("interval_exclusions_above_warning_threshold"); review = True
    if not np.isfinite(metrics["ln_rmssd"]):
        flags.append("lnrmssd_undefined"); failure = True
    if missing_seconds:
        flags.append("missing_ecg_samples")
    irregular = irregular_pattern(selected, cfg) or bool(segment.get("suspected_irregular_rhythm", session.get("suspected_irregular_rhythm", False)))
    if irregular:
        flags.append("irregular_rr_pattern_requires_review"); review = True
    if recording["sfreq"] < 250:
        flags.append("sampling_rate_below_preferred_250hz")
    resp_info = respiration_metrics(recording["times"], recording["resp"], start, end,
                                     recording["sfreq"], cfg, session.get("breathing_cue_times_s"))
    if resp_info["resp_status"] != "valid":
        flags.append("measured_respiration_not_validated")
    breathing_mode = segment["breathing_mode"]
    if (breathing_mode == "paced" and resp_info["resp_status"] == "valid"
            and abs(resp_info["resp_rate_bpm"] - cfg.expected_paced_bpm) > cfg.resp_tolerance_bpm):
        flags.append("measured_breathing_differs_from_pacing_target")
    spectral, spectrum = hrv_spectrum(selected, recording["times"], recording["resp"], resp_info, cfg)
    if spectral["spectral_status"] != "available":
        flags.append("insufficient_contiguous_nn_for_spectrum")
    qc_status = "insufficient_data" if failure else "review_required" if review else "valid"
    context = {**session.get("matched_context", {}), **segment.get("matched_context", {})}
    result = {"participant_id": session["participant_id"], "session_id": session["session_id"],
              "condition": session["condition"], "segment": segment["name"],
              "start_s": start, "end_s": end, "duration_s": duration,
              "breathing_mode": breathing_mode, "matched_context": context,
              "matched_context_id": fingerprint(context) if context else None,
              "settings_id": settings_id, "processing_version": PROCESSING_VERSION,
              "detector": cfg.detector, "detector_version": nk.__version__,
              "detector_library": "NeuroKit2", "sensor_type": "ECG",
              "ecg_channel": session.get("ecg_channel", "ecg_value"),
              "ecg_reference_channel": session.get("ecg_reference_channel"),
              "resp_channel": session.get("resp_channel", "resp_value" if recording["resp"] is not None else None),
              "sampling_rate_hz": recording["sfreq"], "ecg_units": recording["units"],
              **metrics, **resp_info, **spectral,
              "valid_fraction": valid_fraction, "missing_ecg_seconds": missing_seconds,
              "total_beats": int(((beats.time_s >= start) & (beats.time_s < end)).sum()),
              "total_intervals": len(selected), "excluded_intervals": excluded,
              "excluded_fraction": excluded_fraction, "corrected_beats": 0,
              "corrected_fraction": 0.0, "correction_method": "exclude_no_time_domain_interpolation",
              "qc_status": qc_status, "qc_flags": flags, "irregular_rhythm_review": irregular,
              "vas": segment.get("vas"), "comparison_status": "not_requested",
              "effect_lnrmssd": None, "effect_hr": None, "effect_vas": None,
              "personalized_score": None, "score_suppression_reason": "calibration_not_provided",
              "calibration_eligible": bool(segment.get("calibration_eligible", False)),
              "calibration_date": segment.get("calibration_date", session.get("recording_date")),
              "input_sha256": recording["sha256"]}
    return result, selected, spectrum


def match_reasons(records, cfg, match_resp=True):
    reasons = []
    if any(r.get("qc_status") != "valid" for r in records):
        reasons.append("segment_quality_not_valid")
    if len({r.get("participant_id") for r in records}) != 1:
        reasons.append("participant_mismatch")
    if len({r.get("settings_id") for r in records}) != 1 or any(not r.get("settings_id") for r in records):
        reasons.append("processing_settings_mismatch")
    if len({r.get("sensor_type") for r in records}) != 1 or len({r.get("sampling_rate_hz") for r in records}) != 1:
        reasons.append("sensor_or_sampling_rate_mismatch")
    if len({(r.get("ecg_channel"), r.get("ecg_reference_channel")) for r in records}) != 1:
        reasons.append("ecg_lead_configuration_mismatch")
    modes = {r.get("breathing_mode") for r in records}
    if len(modes) != 1 or modes == {"unknown"}:
        reasons.append("breathing_mode_mismatch_or_unknown")
    contexts = {r.get("matched_context_id") for r in records}
    if len(contexts) != 1 or None in contexts or "" in contexts:
        reasons.append("matched_context_missing_or_different")
    if max(r["duration_s"] for r in records) - min(r["duration_s"] for r in records) > cfg.duration_tolerance_s:
        reasons.append("nominal_duration_mismatch")
    if max(r["valid_seconds"] for r in records) - min(r["valid_seconds"] for r in records) > cfg.duration_tolerance_s:
        reasons.append("effective_duration_mismatch")
    if any(r.get("irregular_rhythm_review") for r in records):
        reasons.append("irregular_rhythm_requires_review")
    if match_resp:
        rates = [r.get("resp_rate_bpm") for r in records]
        if any(r.get("resp_source") != "measured" or r.get("resp_status") != "valid" for r in records) or any(v is None or not np.isfinite(v) for v in rates):
            reasons.append("measured_respiration_unavailable_or_low_quality")
        elif max(rates) - min(rates) > cfg.resp_tolerance_bpm:
            reasons.append("respiratory_rate_mismatch")
    return reasons


def compare_records(records, comparison, cfg):
    """Difference in change, or a prespecified breathing-matched during contrast."""
    lookup = {(r["session_id"], r["segment"]): r for r in records}
    keys = comparison.get("records", {})
    kind = comparison.get("kind", "pre_post")
    roles = ("exp_pre", "exp_post", "ctrl_pre", "ctrl_post") if kind == "pre_post" else ("exp_during", "ctrl_during")
    result = {"comparison_id": comparison["id"], "kind": kind, "records": keys,
              "comparison_status": "insufficient_data", "reasons": [],
              "effect_lnrmssd": None, "effect_hr": None, "effect_vas": None}
    if kind not in {"pre_post", "during"}:
        raise ValueError("Comparison kind must be pre_post or during.")
    chosen = {}
    for role in roles:
        ref = keys.get(role)
        if not isinstance(ref, list) or len(ref) != 2 or tuple(ref) not in lookup:
            result["reasons"].append(f"missing_record:{role}")
        else:
            chosen[role] = lookup[tuple(ref)]
    if result["reasons"]:
        return result
    selected = [chosen[r] for r in roles]
    if len({tuple(keys[r]) for r in roles}) != len(roles):
        result["reasons"].append("duplicate_comparison_record")
    for role, r in chosen.items():
        expected = "experimental" if role.startswith("exp") else "control"
        if r["condition"] != expected:
            result["reasons"].append(f"condition_mismatch:{role}")
    result["reasons"] += match_reasons(selected, cfg)
    if kind == "pre_post":
        for prefix in ("exp", "ctrl"):
            pre, post = chosen[f"{prefix}_pre"], chosen[f"{prefix}_post"]
            if pre["segment"] not in {"pre_rest", "pre_paced"} or post["segment"] not in {"post_rest", "delayed_post"}:
                result["reasons"].append(f"invalid_pre_post_roles:{prefix}")
            if pre["session_id"] != post["session_id"] or pre["end_s"] > post["start_s"]:
                result["reasons"].append(f"pre_post_session_or_order_mismatch:{prefix}")
    elif any(r["segment"] != "during" for r in selected):
        result["reasons"].append("during_segment_required")
    if result["reasons"]:
        return result
    for metric, output in (("ln_rmssd", "effect_lnrmssd"), ("mean_hr_bpm", "effect_hr")):
        if not all(r.get(metric) is not None and np.isfinite(r[metric]) for r in selected):
            result["reasons"].append(f"missing_metric:{metric}")
            return result
        if kind == "pre_post":
            exp_delta = chosen["exp_post"][metric] - chosen["exp_pre"][metric]
            ctrl_delta = chosen["ctrl_post"][metric] - chosen["ctrl_pre"][metric]
            result[f"experimental_change_{metric}"] = exp_delta
            result[f"control_change_{metric}"] = ctrl_delta
            result[output] = exp_delta - ctrl_delta
        else:
            result[output] = chosen["exp_during"][metric] - chosen["ctrl_during"][metric]
    if kind == "pre_post" and all(r.get("vas") is not None for r in selected):
        result["effect_vas"] = (chosen["exp_post"]["vas"] - chosen["exp_pre"]["vas"]
                                - chosen["ctrl_post"]["vas"] + chosen["ctrl_pre"]["vas"])
    compatible = result["effect_lnrmssd"] > 0 and result["effect_hr"] < 0
    result["comparison_status"] = "relaxation_compatible" if compatible else "physiology_mixed"
    if compatible and result["effect_vas"] is not None and result["effect_vas"] > 0:
        result["comparison_status"] = "multimodal_convergence"
    return result


def personalized_score(record, baselines, cfg):
    """Only independent, explicitly designated, matched personal baselines qualify."""
    eligible = [r for r in baselines if r.get("calibration_eligible")
                and r.get("segment") in {"pre_rest", "pre_paced"}
                and r.get("participant_id") == record["participant_id"]
                and r.get("session_id") != record["session_id"]
                and r.get("input_sha256") != record.get("input_sha256")]
    # Repeated segments from the same recording cannot increase calibration n.
    unique = {}
    for r in eligible:
        key = r.get("input_sha256")
        if key and r.get("session_id") not in {x["session_id"] for x in unique.values()}:
            if not match_reasons([record, r], cfg):
                unique[key] = r
    valid = [r for r in unique.values() if r.get("ln_rmssd") is not None
             and r.get("mean_hr_bpm") is not None
             and np.isfinite([r["ln_rmssd"], r["mean_hr_bpm"]]).all()]
    provenance = {"eligible_baselines": len(valid), "required_baselines": cfg.min_calibration_recordings,
                  "baseline_sessions": sorted(r["session_id"] for r in valid)}
    if record.get("qc_status") != "valid" or record.get("irregular_rhythm_review"):
        return None, "signal_quality_or_rhythm_review", provenance
    if record.get("resp_source") != "measured" or record.get("resp_status") != "valid":
        return None, "valid_measured_respiration_required", provenance
    if len(valid) < cfg.min_calibration_recordings:
        return None, "fewer_than_five_independent_matched_baselines", provenance
    centers, scales = {}, {}
    for metric in ("ln_rmssd", "mean_hr_bpm"):
        values = np.array([r[metric] for r in valid])
        centers[metric] = float(np.median(values))
        scales[metric] = float(1.4826 * np.median(abs(values - centers[metric])))
        if scales[metric] <= 1e-12:
            return None, "personal_baseline_variability_is_zero", provenance
    z_rmssd = (record["ln_rmssd"] - centers["ln_rmssd"]) / scales["ln_rmssd"]
    z_hr = (record["mean_hr_bpm"] - centers["mean_hr_bpm"]) / scales["mean_hr_bpm"]
    raw_index = .5 * z_rmssd - .5 * z_hr
    score = float(np.clip(50 + 10 * raw_index, 0, 100))
    provenance.update(medians=centers, scaled_mads=scales, z_lnrmssd=z_rmssd,
                      z_hr=z_hr, raw_index=raw_index, formula="clip(50 + 10*(0.5*z_lnRMSSD - 0.5*z_HR), 0, 100)")
    return score, None, provenance


def validate_session(session, recording):
    for key in ("participant_id", "session_id", "condition"):
        if not isinstance(session.get(key), str) or not session[key].strip():
            raise ValueError(f"Each session requires a nonempty {key}.")
    if session["condition"] not in {"experimental", "control", "unspecified"}:
        raise ValueError("condition must be experimental or control (unspecified is preview-only).")
    if not session.get("segments"):
        raise ValueError("Specify segment start/end times; analysis does not infer protocol labels from filenames.")
    first, stop = recording["times"][0], recording["times"][-1] + 1 / recording["sfreq"]
    previous_end, names = first, set()
    for segment in session["segments"]:
        name = segment.get("name")
        if name not in SEGMENTS or name in names:
            raise ValueError(f"Segments must have unique names from {sorted(SEGMENTS)}.")
        names.add(name)
        start, end = float(segment["start_s"]), float(segment["end_s"])
        if (not np.isfinite([start, end]).all() or start < first or end > stop + 1e-6
                or start >= end or start < previous_end):
            raise ValueError(f"Invalid/overlapping/out-of-order segment {name}: [{start}, {end}); recording [{first}, {stop}).")
        previous_end = end
        if segment.get("breathing_mode") not in BREATHING_MODES:
            raise ValueError("Each segment requires breathing_mode: spontaneous, paced or unknown.")
        vas = segment.get("vas")
        if vas is not None and (not isinstance(vas, (int, float)) or not np.isfinite(vas) or not 0 <= vas <= 100):
            raise ValueError("Subjective VAS ratings must be in [0,100].")
        if segment.get("calibration_eligible") and name not in {"pre_rest", "pre_paced"}:
            raise ValueError("Only explicitly designated pre segments can be personal baselines.")


def plot_segment(recording, filtered, beats, selected, result, spectrum, output):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), layout="constrained")
    times = recording["times"]
    start, end = result["start_s"], result["end_s"]
    # Show an interior ECG strip. Full raw ECG remains in the input file.
    strip_start = min(start + 5, max(start, end - 10))
    strip_end = min(end, strip_start + 10)
    strip = (times >= strip_start) & (times < strip_end)
    axes[0].plot(times[strip], recording["ecg"][strip], color="0.75", lw=.7, label="Raw ECG")
    axes[0].plot(times[strip], filtered[strip], color="#116466", lw=1, label="Filtered ECG (polarity oriented)")
    local = beats.loc[(beats.time_s >= strip_start) & (beats.time_s < strip_end)]
    if len(local):
        indices = local.sample_index.to_numpy(dtype=int)
        axes[0].scatter(local.time_s, filtered[indices], s=22, color="#c85131", label="Detected R peaks", zorder=3)
    axes[0].set(title="ECG and detected beats - review strip", ylabel=recording["units"])
    axes[0].legend(loc="upper right", fontsize=8)
    if len(selected):
        good = selected.accepted.to_numpy(dtype=bool)
        axes[1].plot(selected.end_s, selected.rr_ms, color="0.75", lw=.6)
        axes[1].scatter(selected.end_s[good], selected.rr_ms[good], s=9, color="#116466", label="Accepted NN")
        axes[1].scatter(selected.end_s[~good], selected.rr_ms[~good], s=22, marker="x", color="#c85131", label="Excluded RR")
    axes[1].set(title="All RR intervals with quality decisions", ylabel="RR interval (ms)", xlim=(start, end))
    if len(selected):
        axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel("Recording time (s)")
        ax.grid(alpha=.15)
    if spectrum is not None:
        f, p = spectrum.frequency_hz, spectrum.power_ms2_per_hz
        displayed = (f >= .02) & (f <= .5)
        axes[2].plot(f[displayed], p[displayed], color="#116466")
        axes[2].axvspan(.04, .15, color="#edcf82", alpha=.3, label="LF 0.04-0.15 Hz")
        axes[2].axvspan(.15, .4, color="#b1ceda", alpha=.3, label="HF 0.15-0.40 Hz")
        axes[2].legend(fontsize=9)
    else:
        axes[2].text(.5, .5, "Insufficient contiguous NN data for spectrum", transform=axes[2].transAxes, ha="center")
    axes[2].set(title="NN interval spectrum - LF/HF is a descriptive ratio", xlabel="Frequency (Hz)", ylabel="Power (ms²/Hz)", xlim=(.02, .5))
    label = WINDOW_LABELS.get(result["session_id"], result["session_id"])
    fig.suptitle(f"{result['participant_id']} | {label} | ECG quality: {result['qc_status']}\n"
                 f"RMSSD {result['rmssd_ms']:.2f} ms | lnRMSSD {result['ln_rmssd']:.3f} | HR {result['mean_hr_bpm']:.1f} bpm\n"
                 f"LF {result['lf_power_ms2']:.2f} ms² | HF {result['hf_power_ms2']:.2f} ms² | LF/HF {result['lf_hf_ratio']:.2f}\n"
                 f"Window {start:.3f}–{end:.3f} s | Valid {result['valid_seconds']:.1f}/{result['duration_s']:.1f} s | "
                 f"Excluded intervals {result['excluded_fraction']:.1%}", fontsize=12)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_analysis(manifest, root, output, flat_state_figures=False):
    unknown = set(manifest.get("processing", {})) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown processing settings: {sorted(unknown)}")
    cfg = Config(**manifest.get("processing", {}))
    cfg.validate()
    settings_id = fingerprint({"config": asdict(cfg), "version": PROCESSING_VERSION,
                               "detector_version": nk.__version__})
    sessions = manifest.get("sessions", [])
    if not sessions or len({s.get("session_id") for s in sessions}) != len(sessions):
        raise ValueError("Provide at least one session, with distinct session_id values.")
    comparison_ids = [c.get("id") for c in manifest.get("comparisons", [])]
    if any(not x for x in comparison_ids) or len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("Comparisons require distinct, nonempty IDs.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results, audits = [], []
    for session_number, session in enumerate(sessions, 1):
        recording = load_recording(session, root)
        validate_session(session, recording)
        recording["sha256"] = file_hash(recording["path"])
        print(f"[ECG] {session['session_id']}: {len(recording['ecg'])/recording['sfreq']:.1f}s, {recording['sfreq']:g} Hz")
        beats, filtered, logs = detect_beats(recording, cfg, session.get("ecg_polarity", "auto"))
        intervals = classify_intervals(beats, cfg)
        intervals["segment"] = None
        for segment in session["segments"]:
            within = (intervals.start_s >= segment["start_s"]) & (intervals.end_s < segment["end_s"])
            intervals.loc[within, "segment"] = segment["name"]
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session["session_id"]).strip(".") or "session"
        session_dir = output / f"{session_number:02d}_{safe_id}"
        session_dir.mkdir(exist_ok=True)
        beats.to_csv(session_dir / "r_peaks.csv", index=False)
        intervals.to_csv(session_dir / "rr_intervals.csv", index=False)
        audits.append({"session": session, "input_path": str(recording["path"]),
                       "input_sha256": recording["sha256"], "annotations": recording["annotations"],
                       "load_flags": recording["load_flags"], "detector_runs": logs,
                       "ecg_units": recording["units"], "sampling_rate_hz": recording["sfreq"]})
        for segment in session["segments"]:
            result, selected, spectrum = analyze_segment(recording, beats, intervals, session, segment, cfg, settings_id)
            figure = (output / f"{safe_id}.png" if flat_state_figures
                      else session_dir / f"{segment['name']}_hrv.png")
            plot_segment(recording, filtered, beats, selected, result, spectrum, figure)
            result["figure"] = str(figure.relative_to(output))
            result["data_directory"] = str(session_dir.relative_to(output))
            selected.to_csv(session_dir / f"{segment['name']}_intervals.csv", index=False)
            spectral_intervals(selected, cfg).to_csv(session_dir / f"{segment['name']}_spectral_intervals.csv", index=False)
            if spectrum is not None:
                spectrum.to_csv(session_dir / f"{segment['name']}_spectrum.csv", index=False)
            else:
                # A rerun with insufficient data must not leave an old spectrum.
                (session_dir / f"{segment['name']}_spectrum.csv").unlink(missing_ok=True)
            results.append(result)
            print(f"  {segment['name']}: {result['qc_status']}; RMSSD={result['rmssd_ms']:.2f} ms, "
                  f"HR={result['mean_hr_bpm']:.2f} bpm, valid={result['valid_fraction']:.1%}, excluded={result['excluded_fraction']:.1%}")
    baselines = []
    for filename in manifest.get("calibration_results", []):
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        previous = json.loads(path.read_text(encoding="utf-8"))
        baselines.extend(previous.get("segments", []))
    for record in results:
        score, reason, provenance = personalized_score(record, baselines, cfg)
        record.update(personalized_score=score, score_suppression_reason=reason, score_provenance=provenance)
    comparisons = [compare_records(results, c, cfg) for c in manifest.get("comparisons", [])]
    for result in results:
        relevant = [c["comparison_id"] for c in comparisons
                    if [result["session_id"], result["segment"]] in c["records"].values()]
        result["comparison_ids"] = relevant
        # Effects live on comparison records; a segment may belong to multiple contrasts.
        result["comparison_status"] = "see_comparisons" if relevant else "not_requested"
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "mne", "neurokit2", "matplotlib")}
    report = clean_json({"processing_version": PROCESSING_VERSION, "settings_id": settings_id,
                         "processing": asdict(cfg), "software_versions": versions,
                         "segments": results, "comparisons": comparisons, "audit": audits,
                         "interpretation": "HRV components and breathing-matched physiological contrasts; no clinical diagnosis. LF/HF is descriptive, not autonomic balance.",
                         "spectrum_method": "Linear NN interpolation at the configured grid rate; isolated exclusions may be interpolated within configured time/count limits for spectra only. Long gaps and missing ECG runs are never bridged. Window-weighted Welch.",
                         "quality_method": "RR range and local median plus morphology correlation; irregular RR review flag is not an arrhythmia diagnosis."})
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat = [{k: json.dumps(clean_json(v), sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in results]
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons, columns=list(comparisons[0]) if comparisons else
                 ["comparison_id", "kind", "comparison_status", "effect_lnrmssd", "effect_hr", "effect_vas", "reasons"]).to_csv(output / "comparisons.csv", index=False)
    print(f"[HRV] Results: {output}")
    return report


def demo_manifest(output, cfg=None):
    """Generate reproducible known-beat ECG/respiration examples, never subject data."""
    output = Path(output).resolve()
    source = output / "synthetic_inputs"
    source.mkdir(parents=True, exist_ok=True)
    fs = 250
    duration = 630
    times = np.arange(duration * fs) / fs
    rng = np.random.default_rng(20260914)
    sessions = []
    for condition in ("experimental", "control"):
        signal = .015 * np.sin(2 * np.pi * .3 * times) + .003 * rng.normal(size=len(times))
        signal += .01 * np.sin(2 * np.pi * 50 * times)
        peak_times, beat = [], 1.0
        while beat < duration - 1:
            post = beat >= 320
            base = .90 if post and condition == "experimental" else .80
            amplitude = .070 if post and condition == "experimental" else .025
            peak_times.append(beat)
            center = int(round(beat * fs))
            a, b = max(0, center - int(.25 * fs)), min(len(times), center + int(.45 * fs))
            dt = times[a:b] - beat
            signal[a:b] += (1.0 * np.exp(-.5 * (dt / .012) ** 2)
                            - .18 * np.exp(-.5 * ((dt + .035) / .014) ** 2)
                            - .25 * np.exp(-.5 * ((dt - .03) / .016) ** 2)
                            + .12 * np.exp(-.5 * ((dt + .17) / .035) ** 2)
                            + .20 * np.exp(-.5 * ((dt - .25) / .06) ** 2))
            beat += base + amplitude * np.sin(2 * np.pi * .1 * beat)
        resp = np.sin(2 * np.pi * .1 * times) + .02 * rng.normal(size=len(times))
        filename = source / f"synthetic_{condition}.csv"
        pd.DataFrame({"timestamp_s": times, "ecg_value": signal, "resp_value": resp}).to_csv(filename, index=False)
        pd.DataFrame({"true_r_peak_s": peak_times}).to_csv(source / f"synthetic_{condition}_true_peaks.csv", index=False)
        sessions.append({"input": str(filename), "participant_id": "SYNTHETIC", "session_id": condition,
                         "condition": condition, "sampling_rate_hz": fs, "ecg_units": "synthetic units",
                         "matched_context": {"posture": "simulated", "eye_condition": "simulated", "sensor_placement": "simulated"},
                         "metadata": {"device_model": "Synthetic Gaussian PQRST generator", "clock_sync": "Shared generated sample clock"},
                         "segments": [{"name": "pre_rest", "start_s": 10, "end_s": 310,
                                       "breathing_mode": "spontaneous", "vas": 40},
                                      {"name": "post_rest", "start_s": 320, "end_s": 620,
                                       "breathing_mode": "spontaneous", "vas": 70 if condition == "experimental" else 43}]})
    manifest = {"processing": {}, "sessions": sessions,
                "comparisons": [{"id": "synthetic_pre_post_effect", "kind": "pre_post",
                                 "records": {"exp_pre": ["experimental", "pre_rest"],
                                             "exp_post": ["experimental", "post_rest"],
                                             "ctrl_pre": ["control", "pre_rest"],
                                             "ctrl_post": ["control", "post_rest"]}}]}
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def manual_inputs():
    """Read editable variables at call time, so importing never runs an analysis."""
    return {
        "before_exp": {"path": BEFORE_EXP_EDF, "tmin": BEFORE_EXP_TMIN, "tmax": BEFORE_EXP_TMAX},
        "after_exp": {"path": AFTER_EXP_EDF, "tmin": AFTER_EXP_TMIN, "tmax": AFTER_EXP_TMAX},
        "before_control": {"path": BEFORE_CONTROL_EDF, "tmin": BEFORE_CONTROL_TMIN, "tmax": BEFORE_CONTROL_TMAX},
        "after_control": {"path": AFTER_CONTROL_EDF, "tmin": AFTER_CONTROL_TMIN, "tmax": AFTER_CONTROL_TMAX},
    }


def prepare_four_windows(inputs):
    """Validate every input before analysis or output creation; no inferred times."""
    if set(inputs) != set(FOUR_WINDOW_ORDER):
        raise ValueError("Supply before_exp, after_exp, before_control and after_control.")
    if not isinstance(PARTICIPANT_ID, str) or not PARTICIPANT_ID.strip():
        raise ValueError("Set a nonempty PARTICIPANT_ID at the top of this file.")
    sessions, errors, paths = [], [], []
    for key in FOUR_WINDOW_ORDER:
        entry = inputs[key]
        label = WINDOW_LABELS[key]
        filename = str(entry.get("path", "")).strip()
        if not filename:
            errors.append(f"{label}: set {key.upper()}_EDF at the top of this file.")
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
        if not np.isfinite([start, end]).all() or start < 0 or end-start < 45-1e-6:
            errors.append(f"{label}: require TMIN >= 0 and TMAX - TMIN >= 45 seconds; got {start}–{end}.")
            continue
        ecg = entry.get("ecg_channel", ECG_CHANNEL)
        reference = entry.get("ecg_reference_channel", ECG_REFERENCE_CHANNEL)
        resp = entry.get("resp_channel", RESP_CHANNEL)
        reader = mne.io.read_raw_bdf if path.suffix.lower() == ".bdf" else mne.io.read_raw_edf
        try:
            raw = reader(str(path), preload=False, verbose="ERROR")
            try:
                duration = raw.n_times / raw.info["sfreq"]
                if end > duration + 1e-6:
                    raise ValueError(f"TMAX {end:g} exceeds recording duration {duration:g} s.")
                if raw.info["sfreq"] < 100:
                    raise ValueError("ECG sampling rate must be at least 100 Hz.")
                if not ecg or reference == ecg:
                    raise ValueError("Set ECG_CHANNEL; its reference must be different or None.")
                missing = [ch for ch in (ecg, reference, resp) if ch and ch not in raw.ch_names]
                if missing:
                    raise ValueError(f"Missing channels {missing}. Available: {raw.ch_names}")
            finally:
                raw.close()
        except (ValueError, OSError) as error:
            errors.append(f"{label}: {error}")
            continue
        paths.append(path)
        sessions.append({"input": str(path), "participant_id": PARTICIPANT_ID,
                         "session_id": key, "condition": "control" if "control" in key else "experimental",
                         "ecg_channel": ecg, "ecg_reference_channel": reference, "resp_channel": resp,
                         "ecg_polarity": entry.get("ecg_polarity", ECG_POLARITY),
                         "segments": [{"name": "pre_rest" if key.startswith("before") else "post_rest",
                                       "start_s": start, "end_s": end,
                                       "breathing_mode": entry.get("breathing_mode", "unknown")}],
                         "metadata": {"window_label": label, "analysis_mode": "four_manual_windows",
                                      "timing_source": "Manually entered by user; not inferred from triggers."}})
    if len(paths) != len(set(paths)):
        errors.append("Use four different EDF files; a path is repeated in the input variables.")
    if errors:
        raise ValueError("Fix the input variables:\n  " + "\n  ".join(errors))
    return {"processing": {"primary_duration_s": 45.0, "spectrum_window_s": 40.0,
                            "resp_min_s": 30.0}, "sessions": sessions,
            "comparisons": [], "calibration_results": []}


def four_window_comparisons(records):
    """Descriptive arithmetic, retaining all input quality flags; no relaxation label."""
    lookup = {r["session_id"]: r for r in records}
    contrasts = {
        "After EXP minus after CONTROL": {"after_exp": 1, "after_control": -1},
        "Before EXP minus before CONTROL": {"before_exp": 1, "before_control": -1},
        "EXP change (after minus before)": {"after_exp": 1, "before_exp": -1},
        "CONTROL change (after minus before)": {"after_control": 1, "before_control": -1},
        "Difference in changes (EXP minus CONTROL)": {
            "after_exp": 1, "before_exp": -1, "after_control": -1, "before_control": 1},
    }
    rows = []
    for name, weights in contrasts.items():
        chosen = [lookup[k] for k in weights]
        flags = [f"{r['session_id']}:{flag}" for r in chosen for flag in r["qc_flags"]]
        if len({(r["ecg_channel"], r["ecg_reference_channel"], r["sampling_rate_hz"]) for r in chosen}) != 1:
            flags.append("lead_or_sampling_mismatch")
        if max(r["duration_s"] for r in chosen)-min(r["duration_s"] for r in chosen) > 1e-6:
            flags.append("unequal_selected_durations")
        if max(r["valid_seconds"] for r in chosen)-min(r["valid_seconds"] for r in chosen) > 5:
            flags.append("effective_duration_mismatch_over_5s")
        if any(r["breathing_mode"] == "unknown" for r in chosen):
            flags.append("breathing_conditions_unconfirmed")
        elif len({r["breathing_mode"] for r in chosen}) != 1:
            flags.append("breathing_mode_mismatch")
        rates = [r["resp_rate_bpm"] for r in chosen]
        if all(r["resp_status"] == "valid" for r in chosen) and max(rates)-min(rates) > .5:
            flags.append("respiratory_rate_mismatch")
        if len({r["input_sha256"] for r in chosen}) != len(chosen):
            flags.append("duplicate_recording_content")
        review = any(r["qc_status"] != "valid" for r in chosen)
        for metric, (label, unit) in COMPARISON_METRICS.items():
            values = [lookup[k].get(metric) for k in weights]
            available = all(v is not None and np.isfinite(v) for v in values)
            difference = sum(weights[k]*lookup[k][metric] for k in weights) if available else None
            rows.append({"comparison": name, "metric": metric, "metric_label": label,
                         "unit": unit, "difference": difference,
                         "status": "unavailable" if not available else
                         "descriptive_review_required" if review else "descriptive_only",
                         "quality_flags": "; ".join(flags), "spectral_duration_limited": metric in
                         {"lf_power_ms2", "hf_power_ms2", "lf_hf_ratio"}})
    return rows


def plot_four_window_comparisons(records, output):
    """One comparison image: all four PSDs plus paired RMSSD, LF/HF, LF and HF."""
    lookup = {r["session_id"]: r for r in records}
    colors = {"before_exp": "#65a8a0", "after_exp": "#116466",
              "before_control": "#d9a466", "after_control": "#a95216"}
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig = plt.figure(figsize=(15, 14), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1])
        ax = fig.add_subplot(grid[0, :])
        ax.axvspan(.04, .15, color="#f3dc9a", alpha=.28)
        ax.axvspan(.15, .4, color="#bdd7e8", alpha=.28)
        missing = []
        for key in FOUR_WINDOW_ORDER:
            record = lookup[key]
            directory = record.get("data_directory", str(Path(record["figure"]).parent))
            path = output / directory / f"{record['segment']}_spectrum.csv"
            if record["spectral_status"] != "available" or not path.is_file():
                missing.append(WINDOW_LABELS[key])
                continue
            spectrum = pd.read_csv(path)
            mask = (spectrum.frequency_hz >= .025) & (spectrum.frequency_hz <= .4)
            ax.plot(spectrum.frequency_hz[mask], spectrum.power_ms2_per_hz[mask],
                    color=colors[key], lw=2, ls="--" if key.startswith("before") else "-",
                    label=f"{WINDOW_LABELS[key]} ({record['duration_s']:g} s)")
        if missing:
            ax.text(.98, .95, "PSD unavailable: " + ", ".join(missing), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        if ax.lines:
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=4, fontsize=10, frameon=False)
        ax.set(title="All four states · NN interval PSD", xlabel="Frequency (Hz)",
               ylabel="Power (ms²/Hz)", xlim=(.025, .4))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        for cell, metric in zip([grid[1, 0], grid[1, 1], grid[2, 0], grid[2, 1]],
                                ["rmssd_ms", "lf_hf_ratio", "lf_power_ms2", "hf_power_ms2"]):
            ax = fig.add_subplot(cell)
            label, unit = COMPARISON_METRICS[metric]
            for suffix, color, offset, name in [("exp", "#116466", -.035, "EXP"),
                                                 ("control", "#a95216", .035, "CONTROL")]:
                values = [lookup[f"{phase}_{suffix}"].get(metric) for phase in ("before", "after")]
                y = [v if v is not None and np.isfinite(v) else np.nan for v in values]
                x = np.array([0, 1]) + offset
                ax.plot(x, y, "o-", color=color, lw=2, ms=7, label=name)
                for position, value in zip(x, y):
                    if np.isfinite(value):
                        ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 9 if suffix == "exp" else -17),
                                    textcoords="offset points", ha="center", fontsize=10, color=color)
                    else:
                        ax.text(position, .05 if suffix == "exp" else .13, f"{name}: N/A",
                                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=color)
            ax.set(title=label, ylabel=unit, xticks=[0, 1], xticklabels=["Before", "After"], xlim=(-.3, 1.3))
            ax.margins(y=.3)
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=10, frameon=False)
        durations = " | ".join(f"{WINDOW_LABELS[k]}: {lookup[k]['duration_s']:g} s" for k in FOUR_WINDOW_ORDER)
        flagged = [WINDOW_LABELS[k] for k in FOUR_WINDOW_ORDER if lookup[k]["qc_status"] != "valid"]
        quality = "ECG review needed: " + ", ".join(flagged) if flagged else "ECG quality checks passed for all four states"
        fig.suptitle(f"{records[0]['participant_id']} · HRV state comparison\n{durations}\n"
                     f"{quality}\nLF 0.04–0.15 Hz (gold) | HF 0.15–0.40 Hz (blue); spectral estimates are exploratory",
                     fontsize=13)
        fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def remove_obsolete_graph_outputs(output):
    """Remove only obsolete files produced by older versions in this run folder."""
    root = output.resolve()
    names = ["report.html", "metrics_comparison.png", "metrics_comparison.pdf",
             "psd_comparison.png", "psd_comparison.pdf"]
    for index, key in enumerate(FOUR_WINDOW_ORDER, 1):
        segment = "pre_rest" if key.startswith("before") else "post_rest"
        names.append(f"{index:02d}_{key}/{segment}_hrv.png")
    for name in names:
        path = root / name
        if root not in path.resolve().parents:
            raise ValueError(f"Refusing cleanup outside output directory: {path}")
        if path.is_file():
            path.unlink()


def run_four_windows(inputs=None, output=None):
    manifest = prepare_four_windows(manual_inputs() if inputs is None else inputs)
    output = Path(OUTPUT_DIR if output is None else output).resolve()
    print("[HRV] " + SHORT_WINDOW_NOTE)
    report = run_analysis(manifest, Path(__file__).resolve().parent, output, flat_state_figures=True)
    report["analysis_mode"] = "four_manual_windows"
    report["interpretation"] = SHORT_WINDOW_NOTE
    report["comparisons"] = four_window_comparisons(report["segments"])
    for r in report["segments"]:
        r.update(window_label=WINDOW_LABELS[r["session_id"]], comparison_status="descriptive_only",
                 spectral_duration_limited=True, score_suppression_reason="not_requested_in_four_window_mode")
        r["qc_flags"].append("short_psd_window_exploratory_spectrum")
    # Comparisons use descriptive arithmetic, not the legacy relaxation decision rules.
    pd.DataFrame(report["comparisons"]).to_csv(output / "comparisons.csv", index=False)
    flat = [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in report["segments"]]
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    all_spectra = []
    for r in report["segments"]:
        if r["spectral_status"] == "available":
            table = pd.read_csv(output / r["data_directory"] / f"{r['segment']}_spectrum.csv")
            table.insert(0, "window", WINDOW_LABELS[r["session_id"]])
            all_spectra.append(table)
    combined = pd.concat(all_spectra, ignore_index=True) if all_spectra else pd.DataFrame(
        columns=["window", "frequency_hz", "power_ms2_per_hz"])
    combined.to_csv(output / "psd_values.csv", index=False)
    (output / "results.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot_four_window_comparisons(report["segments"], output)
    remove_obsolete_graph_outputs(output)
    columns = ["window_label", "duration_s", "rmssd_ms", "mean_hr_bpm", "lf_power_ms2", "hf_power_ms2", "lf_hf_ratio", "qc_status"]
    print("\n" + pd.DataFrame(report["segments"])[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[GRAPHS] {output}")
    for name in [*(f"{key}.png" for key in FOUR_WINDOW_ORDER), "comparison.png"]:
        print(f"  {name}")
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv == ["--no-open"]:
        try:
            run_four_windows()
        except (ValueError, KeyError, OSError) as error:
            print(f"[ERROR] {error}", file=sys.stderr)
            return 2
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="List EDF/BDF channels, duration and annotations")
    inspect_parser.add_argument("input", type=Path)
    analyze_parser = commands.add_parser("analyze", help="Analyze explicit sessions/segments from a JSON manifest")
    analyze_parser.add_argument("--config", required=True, type=Path)
    analyze_parser.add_argument("--output", type=Path, default=Path("hrv_results"))
    demo_parser = commands.add_parser("demo", help="Run reproducible synthetic experimental/control recordings")
    demo_parser.add_argument("--output", type=Path, default=Path("hrv_results/synthetic_demo"))
    preview_parser = commands.add_parser("preview", help="Check one ECG recording without assigning an experimental condition")
    preview_parser.add_argument("input", type=Path)
    preview_parser.add_argument("--ecg-channel", required=True)
    preview_parser.add_argument("--ecg-reference-channel")
    preview_parser.add_argument("--resp-channel")
    preview_parser.add_argument("--start", type=float, required=True)
    preview_parser.add_argument("--end", type=float, required=True)
    preview_parser.add_argument("--output", type=Path, default=Path("hrv_results/recording_preview"))
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            if args.input.suffix.lower() not in {".edf", ".bdf"}:
                raise ValueError("inspect accepts EDF/BDF; CSV column names are specified in HRV_README.md.")
            reader = mne.io.read_raw_bdf if args.input.suffix.lower() == ".bdf" else mne.io.read_raw_edf
            raw = reader(str(args.input), preload=False, verbose="ERROR")
            print(json.dumps({"duration_s": raw.n_times / raw.info["sfreq"], "sampling_rate_hz": raw.info["sfreq"],
                              "channels": raw.ch_names, "annotations": [str(a) for a in raw.annotations]}, indent=2))
            return 0
        if args.command == "demo":
            args.output.mkdir(parents=True, exist_ok=True)
            manifest = demo_manifest(args.output)
            print("[DEMO] Synthetic data only; these are not participant results.")
            run_analysis(manifest, Path.cwd(), args.output)
        elif args.command == "preview":
            manifest = {"sessions": [{"input": str(args.input.resolve()), "participant_id": "UNSPECIFIED",
                         "session_id": "signal_preview", "condition": "unspecified",
                         "ecg_channel": args.ecg_channel, "ecg_reference_channel": args.ecg_reference_channel,
                         "resp_channel": args.resp_channel, "segments": [{"name": "during", "start_s": args.start,
                         "end_s": args.end, "breathing_mode": "unknown"}]}]}
            print("[PREVIEW] 'during' is a technical preview window, not a verified protocol label. No relaxation comparison or score.")
            run_analysis(manifest, Path.cwd(), args.output)
        else:
            manifest = json.loads(args.config.read_text(encoding="utf-8-sig"))
            run_analysis(manifest, args.config.resolve().parent, args.output)
    except (ValueError, KeyError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
