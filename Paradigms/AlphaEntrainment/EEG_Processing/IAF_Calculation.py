import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import mne
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.signal import welch



EDF_PATH = r"C:\Users\saiik\Downloads\VYKHARI\SUB-2VYKHARI~ _b2102c63-2618-414e-8b1c-e90e7b171048.edf"

# Manually inspected clean windows (seconds from recording start)
EYES_CLOSED_TMIN = 73.0
EYES_CLOSED_TMAX = 133.0   # aim for 60 s; must be >= MIN_CLEAN_EYES_CLOSED_SEC

EYES_OPEN_TMIN = 5.0
EYES_OPEN_TMAX = 70.0

# Optional repeat attempt if the first eyes-closed segment is invalid
# (protocol: repeat the 60 s eyes-closed recording once, else exclude).
# Leave as None to skip auto-retry.
RETRY_EYES_CLOSED_TMIN = None
RETRY_EYES_CLOSED_TMAX = None

REFERENCE_CHANNEL = "Cz"
REREFERENCE_CHANNELS = ["O1", "O2"]   # re-derived as (channel - Cz)
POSTERIOR_ROI = ["O1", "Oz", "O2"]    # Oz used as-recorded unless also
                                       # added to REREFERENCE_CHANNELS

BANDPASS = (1.0, 40.0)
MAINS_NOTCH_HZ = 50.0   # India mains; use 60.0 for US recordings

MIN_CLEAN_EYES_CLOSED_SEC = 45.0
WINDOW_SEC = 4.0
WINDOW_OVERLAP = 0.5   # 50%

ALPHA_SEARCH_BAND = (7.0, 13.0)
CROSS_CHANNEL_TOL_HZ = 0.5
MIN_COMPATIBLE_CHANNELS = 2

OUTPUT_DIR = "iaf_results"

# ============================================================
# Pipeline
# ============================================================


def load_raw(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(f"[INFO] Loaded {edf_path} | channels: {raw.ch_names} | "
          f"sfreq={raw.info['sfreq']} Hz | duration={raw.times[-1]:.1f}s")
    return raw


def rereference_to_channel(raw, channels, ref_channel):
    """Re-derive `channels` in-place as bipolar (channel - ref_channel).
    The original channel and the reference channel are replaced by the
    new derivation (mne requires this to reuse the same channel name).
    Everything outside `channels` and `ref_channel` is untouched."""
    raw = raw.copy()
    present = [ch for ch in channels if ch in raw.ch_names]
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        print(f"[WARN] Re-reference channels not found, skipping: {missing}")
    if ref_channel not in raw.ch_names:
        raise ValueError(f"Reference channel '{ref_channel}' not found in recording.")
    if present:
        # Use temporary names for the derived channels, then rename back
        # to the original names — reusing the anode's own name directly
        # as ch_name trips a channel-merge conflict in mne.
        tmp_names = [f"{ch}__bipolar_tmp" for ch in present]
        raw = mne.set_bipolar_reference(
            raw, anode=present, cathode=[ref_channel] * len(present),
            ch_name=tmp_names, drop_refs=True, verbose=False,
        )
        raw.rename_channels(dict(zip(tmp_names, present)))
        print(f"[INFO] Re-referenced to {ref_channel}: {present}")
    return raw


def apply_filters(raw, bandpass, notch_hz):
    raw = raw.copy()
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[notch_hz], verbose=False)
    return raw


def crop(raw, tmin, tmax):
    return raw.copy().crop(tmin=tmin, tmax=tmax)


def channel_quality_log(raw, roi_channels, ref_channel, rereferenced):
    """Log which posterior channels were used / missing / re-referenced.
    Extend this if you start rejecting noisy channels and swapping in
    frontal electrodes, or regressing out an ocular ERP via an EOG
    channel — log the substitution here."""
    log = {}
    for ch in roi_channels:
        if ch not in raw.ch_names:
            log[ch] = "missing - excluded from ROI"
        elif ch in rereferenced:
            log[ch] = f"used (re-referenced to {ref_channel})"
        else:
            log[ch] = "used (as-recorded reference)"
    return log


def welch_psd(data, sfreq, window_sec, overlap):
    """4 s Hann windows, 50% overlap, averaged -> this is exactly what
    scipy.signal.welch does internally in one call."""
    nperseg = int(round(window_sec * sfreq))
    noverlap = int(round(nperseg * overlap))
    freqs, psd = welch(data, fs=sfreq, window="hann", nperseg=nperseg,
                        noverlap=noverlap, detrend="constant")
    return freqs, psd


def compute_channel_psds(raw, channels, window_sec, overlap):
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=channels)
    freqs = None
    psds = {}
    for i, ch in enumerate(channels):
        f, p = welch_psd(data[i], sfreq, window_sec, overlap)
        freqs = f
        psds[ch] = p
    return freqs, psds


def _parabolic_vertex(freqs, log_psd, idx):
    """Fit a parabola to log-power at idx-1, idx, idx+1.
    Return (vertex_freq, is_concave)."""
    if idx <= 0 or idx >= len(freqs) - 1:
        return None, False
    x = freqs[idx - 1: idx + 2]
    y = log_psd[idx - 1: idx + 2]
    a, b, c = np.polyfit(x, y, 2)
    concave = a < 0
    if not concave:
        return None, False
    vertex = -b / (2 * a)
    return vertex, concave


def _largest_local_max_in_band(freqs, psd, band):
    lo, hi = band
    band_idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    best_idx, best_val = None, -np.inf
    for i in band_idx:
        if i == 0 or i == len(freqs) - 1:
            continue
        if psd[i] > psd[i - 1] and psd[i] > psd[i + 1] and psd[i] > best_val:
            best_val = psd[i]
            best_idx = i
    return best_idx


def find_channel_peak(freqs, psd, band):
    """Largest valid local max in-band -> parabolic vertex on log-power,
    kept only if concave and inside the search band."""
    log_psd = np.log10(psd + 1e-20)
    idx = _largest_local_max_in_band(freqs, psd, band)
    if idx is None:
        return None
    vertex, concave = _parabolic_vertex(freqs, log_psd, idx)
    if vertex is None or not (band[0] <= vertex <= band[1]):
        return None
    return vertex


def find_iaf(freqs, channel_psds, roi_channels, band, tol, min_channels):
    flags = []
    posterior_avg = np.mean([channel_psds[ch] for ch in roi_channels], axis=0)
    iaf_candidate = find_channel_peak(freqs, posterior_avg, band)
    per_channel_peaks = {ch: find_channel_peak(freqs, channel_psds[ch], band) for ch in roi_channels}

    if iaf_candidate is None:
        flags.append("No valid concave alpha peak in posterior-average PSD within 7-13 Hz")
        return {
            "iaf_hz": None, "valid": False, "posterior_candidate_hz": None,
            "per_channel_peaks_hz": {ch: None for ch in roi_channels},
            "compatible_channels": [], "quality_flags": flags,
        }

    compatible = [ch for ch, pk in per_channel_peaks.items()
                  if pk is not None and abs(pk - iaf_candidate) <= tol]
    valid = len(compatible) >= min_channels

    if not valid:
        flags.append(f"Only {len(compatible)}/{len(roi_channels)} channels within "
                      f"+/-{tol} Hz of posterior candidate ({min_channels} required)")
    else:
        flags.append(f"{len(compatible)}/{len(roi_channels)} channels compatible within +/-{tol} Hz")

    return {
        "iaf_hz": round(float(iaf_candidate), 3) if valid else None,
        "valid": valid,
        "posterior_candidate_hz": round(float(iaf_candidate), 3),
        "per_channel_peaks_hz": {ch: (round(float(v), 3) if v is not None else None)
                                  for ch, v in per_channel_peaks.items()},
        "compatible_channels": compatible,
        "quality_flags": flags,
    }


def analyze_eyes_closed(raw_full, tmin, tmax, roi_channels, ref_channel, rereferenced,
                         min_clean_sec, window_sec, overlap, band, tol, min_channels):
    duration = tmax - tmin
    result = {"condition": "eyes_closed", "tmin": tmin, "tmax": tmax, "clean_seconds": duration}

    if duration < min_clean_sec:
        result["valid"] = False
        result["quality_flags"] = [f"Segment length {duration:.1f}s < required {min_clean_sec}s"]
        return result

    seg = crop(raw_full, tmin, tmax)
    freqs, psds = compute_channel_psds(seg, roi_channels, window_sec, overlap)
    iaf_result = find_iaf(freqs, psds, roi_channels, band, tol, min_channels)

    result.update(iaf_result)
    result["channels_used"] = roi_channels
    result["freqs"] = freqs
    result["channel_psds"] = psds
    result["channel_quality_log"] = channel_quality_log(raw_full, roi_channels, ref_channel, rereferenced)
    return result


def analyze_eyes_open_baseline(raw_full, tmin, tmax, roi_channels, window_sec, overlap, band):
    """QC / baseline only. Not gated by the IAF validity rules."""
    duration = tmax - tmin
    seg = crop(raw_full, tmin, tmax)
    freqs, psds = compute_channel_psds(seg, roi_channels, window_sec, overlap)
    posterior_avg = np.mean([psds[ch] for ch in roi_channels], axis=0)
    band_peak = find_channel_peak(freqs, posterior_avg, band)
    return {
        "condition": "eyes_open_baseline",
        "tmin": tmin, "tmax": tmax, "clean_seconds": duration,
        "channels_used": roi_channels,
        "descriptive_alpha_peak_hz": round(float(band_peak), 3) if band_peak else None,
        "note": "Eyes-open interval - QC / baseline reference only, not gated by IAF validity rules.",
        "freqs": freqs, "channel_psds": psds,
    }


def plot_condition(result, band, out_path):
    freqs = result["freqs"]
    psds = result["channel_psds"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch, p in psds.items():
        ax.semilogy(freqs, p, label=ch, alpha=0.6)
    posterior_avg = np.mean(list(psds.values()), axis=0)
    ax.semilogy(freqs, posterior_avg, label="posterior avg", color="k", lw=2)
    ax.axvspan(band[0], band[1], color="gray", alpha=0.15, label="7-13 Hz search band")
    peak = result.get("iaf_hz") or result.get("descriptive_alpha_peak_hz") or result.get("posterior_candidate_hz")
    if peak:
        ax.axvline(peak, color="red", ls="--", label=f"peak = {peak:.2f} Hz")
    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(result["condition"])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(config):
    out_dir = Path(config["OUTPUT_DIR"])
    out_dir.mkdir(exist_ok=True, parents=True)

    raw = load_raw(config["EDF_PATH"])
    raw = rereference_to_channel(raw, config["REREFERENCE_CHANNELS"], config["REFERENCE_CHANNEL"])
    raw = apply_filters(raw, config["BANDPASS"], config["MAINS_NOTCH_HZ"])

    roi = [ch for ch in config["POSTERIOR_ROI"] if ch in raw.ch_names]
    missing_roi = [ch for ch in config["POSTERIOR_ROI"] if ch not in raw.ch_names]
    if missing_roi:
        print(f"[WARN] Posterior ROI channels missing from recording: {missing_roi}")

    ec_result = analyze_eyes_closed(
        raw, config["EYES_CLOSED_TMIN"], config["EYES_CLOSED_TMAX"], roi,
        config["REFERENCE_CHANNEL"], config["REREFERENCE_CHANNELS"],
        config["MIN_CLEAN_EYES_CLOSED_SEC"], config["WINDOW_SEC"], config["WINDOW_OVERLAP"],
        config["ALPHA_SEARCH_BAND"], config["CROSS_CHANNEL_TOL_HZ"], config["MIN_COMPATIBLE_CHANNELS"],
    )

    if not ec_result.get("valid", False) and config.get("RETRY_EYES_CLOSED_TMIN") is not None:
        print("[INFO] First eyes-closed attempt invalid - trying repeat interval per protocol.")
        retry = analyze_eyes_closed(
            raw, config["RETRY_EYES_CLOSED_TMIN"], config["RETRY_EYES_CLOSED_TMAX"], roi,
            config["REFERENCE_CHANNEL"], config["REREFERENCE_CHANNELS"],
            config["MIN_CLEAN_EYES_CLOSED_SEC"], config["WINDOW_SEC"], config["WINDOW_OVERLAP"],
            config["ALPHA_SEARCH_BAND"], config["CROSS_CHANNEL_TOL_HZ"], config["MIN_COMPATIBLE_CHANNELS"],
        )
        retry["retry_of"] = {"tmin": ec_result["tmin"], "tmax": ec_result["tmax"]}
        ec_result = retry
        if not ec_result.get("valid", False):
            ec_result["participant_status"] = "EXCLUDE - no valid IAF after repeat"
            print("[RESULT] Participant EXCLUDED - no valid IAF after repeat attempt.")
    elif not ec_result.get("valid", False):
        ec_result["participant_status"] = (
            "INVALID - repeat the 60s eyes-closed recording once, "
            "then exclude participant if still invalid"
        )

    if "freqs" in ec_result:
        plot_condition(ec_result, config["ALPHA_SEARCH_BAND"], out_dir / "eyes_closed_psd.png")

    eo_result = analyze_eyes_open_baseline(
        raw, config["EYES_OPEN_TMIN"], config["EYES_OPEN_TMAX"], roi,
        config["WINDOW_SEC"], config["WINDOW_OVERLAP"], config["ALPHA_SEARCH_BAND"],
    )
    plot_condition(eo_result, config["ALPHA_SEARCH_BAND"], out_dir / "eyes_open_baseline_psd.png")

    def strip_arrays(d):
        return {k: v for k, v in d.items() if k not in ("freqs", "channel_psds")}

    summary = {
        "edf_path": config["EDF_PATH"],
        "eyes_closed": strip_arrays(ec_result),
        "eyes_open_baseline": strip_arrays(eo_result),
    }

    with open(out_dir / "iaf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    if ec_result.get("valid"):
        print(f"\n[RESULT] Final IAF for this visit: {ec_result['iaf_hz']} Hz "
              f"(channels: {ec_result['compatible_channels']}, "
              f"clean: {ec_result['clean_seconds']:.1f}s) "
              f"-> use this value for ACTIVE flicker.")
    return summary


if __name__ == "__main__":
    config = {k: v for k, v in globals().items() if k.isupper()}
    run(config)