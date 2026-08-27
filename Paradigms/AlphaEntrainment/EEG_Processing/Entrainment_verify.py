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


# ============================================================
# Configuration
# ============================================================

EDF_PATH = r"C:\Users\saiik\Downloads\mridul_after_flash.edf"

# Enter the start and stop time (in seconds from recording start)
# of the segment you want to analyse.
SEGMENT_TMIN = 13    # <-- change me
SEGMENT_TMAX = 246   # <-- change me

# A short label used in plot titles and output filenames.
SEGMENT_LABEL = "entrainment_window"

REFERENCE_CHANNEL = "Cz"
REREFERENCE_CHANNELS = ["O1", "O2"]   # re-derived as (channel - Cz)
POSTERIOR_ROI = ["O1", "O2"]          # channels whose PSD will be computed

BANDPASS = (1.0, 40.0)
MAINS_NOTCH_HZ = 50.0   # India mains; use 60.0 for US recordings

WINDOW_SEC = 4.0
WINDOW_OVERLAP = 0.5   # 50 %

ALPHA_SEARCH_BAND = (7.0, 13.0)
CROSS_CHANNEL_TOL_HZ = 0.5
MIN_COMPATIBLE_CHANNELS = 2

OUTPUT_DIR = r"C:\Users\saiik\Downloads\Entrainment Verify"

# ============================================================
# Pipeline helpers  (unchanged from IAF_Calculation.py)
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


def apply_filters(raw, bandpass, notch_hz):
    raw = raw.copy()
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[notch_hz], verbose=False)
    return raw


def crop(raw, tmin, tmax):
    return raw.copy().crop(tmin=tmin, tmax=tmax)


def channel_quality_log(raw, roi_channels, ref_channel, rereferenced):
    """Log which posterior channels were used / missing / re-referenced."""
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
    """4 s Hann windows, 50% overlap, averaged -> scipy.signal.welch."""
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
    per_channel_peaks = {ch: find_channel_peak(freqs, channel_psds[ch], band)
                         for ch in roi_channels}

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


# ============================================================
# Segment analysis  (replaces eyes-open / eyes-closed split)
# ============================================================


def analyze_segment(raw_full, tmin, tmax, label, roi_channels,
                    ref_channel, rereferenced,
                    window_sec, overlap, band, tol, min_channels):
    """Compute PSD and detect alpha peak for the user-specified time window."""
    duration = tmax - tmin
    result = {
        "condition": label,
        "tmin": tmin,
        "tmax": tmax,
        "duration_seconds": duration,
    }

    seg = crop(raw_full, tmin, tmax)
    freqs, psds = compute_channel_psds(seg, roi_channels, window_sec, overlap)
    iaf_result = find_iaf(freqs, psds, roi_channels, band, tol, min_channels)

    result.update(iaf_result)
    result["channels_used"] = roi_channels
    result["freqs"] = freqs
    result["channel_psds"] = psds
    result["channel_quality_log"] = channel_quality_log(
        raw_full, roi_channels, ref_channel, rereferenced)
    return result


def plot_segment(result, band, out_path):
    freqs = result["freqs"]
    psds = result["channel_psds"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch, p in psds.items():
        ax.semilogy(freqs, p, label=ch, alpha=0.6)
    posterior_avg = np.mean(list(psds.values()), axis=0)
    ax.semilogy(freqs, posterior_avg, label="posterior avg", color="k", lw=2)
    ax.axvspan(band[0], band[1], color="gray", alpha=0.15,
               label=f"{band[0]}-{band[1]} Hz search band")
    peak = result.get("iaf_hz") or result.get("posterior_candidate_hz")
    if peak:
        ax.axvline(peak, color="red", ls="--", label=f"peak = {peak:.2f} Hz")
    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"{result['condition']}  [{result['tmin']}s \u2013 {result['tmax']}s]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved -> {out_path}")


# ============================================================
# Main entry point
# ============================================================


def run(config):
    out_dir = Path(config["OUTPUT_DIR"])
    out_dir.mkdir(exist_ok=True, parents=True)

    raw = load_raw(config["EDF_PATH"])
    raw = rereference_to_channel(raw, config["REREFERENCE_CHANNELS"],
                                 config["REFERENCE_CHANNEL"])
    raw = apply_filters(raw, config["BANDPASS"], config["MAINS_NOTCH_HZ"])

    roi = [ch for ch in config["POSTERIOR_ROI"] if ch in raw.ch_names]
    missing_roi = [ch for ch in config["POSTERIOR_ROI"] if ch not in raw.ch_names]
    if missing_roi:
        print(f"[WARN] Posterior ROI channels missing from recording: {missing_roi}")

    result = analyze_segment(
        raw,
        tmin=config["SEGMENT_TMIN"],
        tmax=config["SEGMENT_TMAX"],
        label=config["SEGMENT_LABEL"],
        roi_channels=roi,
        ref_channel=config["REFERENCE_CHANNEL"],
        rereferenced=config["REREFERENCE_CHANNELS"],
        window_sec=config["WINDOW_SEC"],
        overlap=config["WINDOW_OVERLAP"],
        band=config["ALPHA_SEARCH_BAND"],
        tol=config["CROSS_CHANNEL_TOL_HZ"],
        min_channels=config["MIN_COMPATIBLE_CHANNELS"],
    )

    plot_filename = f"{config['SEGMENT_LABEL']}_psd.png"
    plot_segment(result, config["ALPHA_SEARCH_BAND"], out_dir / plot_filename)

    def strip_arrays(d):
        return {k: v for k, v in d.items() if k not in ("freqs", "channel_psds")}

    summary = {
        "edf_path": config["EDF_PATH"],
        "segment": strip_arrays(result),
    }

    json_filename = f"{config['SEGMENT_LABEL']}_psd_summary.json"
    with open(out_dir / json_filename, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))

    if result.get("valid"):
        print(f"\n[RESULT] Alpha peak detected: {result['iaf_hz']} Hz "
              f"(channels: {result['compatible_channels']}, "
              f"window: {result['tmin']}s \u2013 {result['tmax']}s)")
    else:
        print("\n[RESULT] No valid alpha peak found in the specified window.")

    return summary


if __name__ == "__main__":
    config = {k: v for k, v in globals().items() if k.isupper()}
    run(config)
