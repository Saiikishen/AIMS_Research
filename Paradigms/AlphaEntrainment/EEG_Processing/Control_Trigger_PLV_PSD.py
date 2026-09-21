#!/usr/bin/env python3
"""EDF-only EEG PLV and PSD, including after-flashing PSD from the final 60 s.

After-flashing timing always uses the original EDF, independent of --start/end.
Each PSD condition excludes 2 s at each edge after filtering. The final 60 s
therefore contribute 56 s of PSD data. No stimulus CSV is required.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import trapezoid
from scipy.signal import butter, hilbert, iirnotch, sosfiltfilt, tf2sos, welch

EDF_PATH = r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\CONTROL\SUB15~ JACOB_0e04f0e8-fb17-4f50-8a72-31c9e4d27116.edf"
OUTPUT_DIR = Path(__file__).resolve().parent / "control_results"
CHANNELS = ("O1", "O2")
REFERENCE_CHANNEL = "Cz"
ANALYSIS_START_S = 0.0
ANALYSIS_END_S = None  # None means the end of the EDF; no condition inferred
PLV_BAND_HZ = (8.0, 13.0)
PLV_WINDOW_S = 10.0
EDGE_EXCLUDE_S = 2.0
PSD_WINDOW_S = 4.0
AFTER_FLASH_DURATION_S = 60.0  # last 60 seconds of the original EDF
TRIGGER_CHANNEL = "DC7"
TRIGGER_THRESHOLD_V = 0.005
MIN_TRIGGER_ISI_S = 0.1


def detect_triggers(signal, sfreq, threshold):
    """Report pulse onsets of either polarity, relative to EDF recording start."""
    if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(signal).all():
        raise ValueError("Trigger threshold must be positive and DC signal finite.")
    baseline = float(np.median(signal))
    above = np.abs(signal - baseline) > threshold
    rising = np.flatnonzero(np.diff(above.astype(int), prepend=0) == 1) / sfreq
    kept = []
    for time in rising:
        if not kept or time - kept[-1] >= MIN_TRIGGER_ISI_S:
            kept.append(time)
    return np.asarray(kept), rising, baseline


def band_power(frequencies, psd, low, high):
    """Integrate PSD with interpolated endpoints at the exact band limits."""
    x = np.r_[low, frequencies[(frequencies > low) & (frequencies < high)], high]
    return float(trapezoid(np.interp(x, frequencies, psd), x))


def phase_locking_value(phase_a, phase_b):
    """Time-domain PLV between two simultaneous EEG phase series."""
    return float(np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))


def broadband_filter(eeg, sfreq):
    """Use identical broadband preprocessing for every PSD condition."""
    eeg = np.asarray(eeg, dtype=float)
    if not np.isfinite(eeg).all() or not np.isfinite(sfreq) or sfreq <= 60:
        raise ValueError("PSD requires finite EEG and sampling rate above 60 Hz.")
    broad = eeg.copy()
    if sfreq > 100:
        b, a = iirnotch(50, 30, fs=sfreq)
        broad = sosfiltfilt(tf2sos(b, a), broad, axis=-1)
    return sosfiltfilt(butter(4, [3, 30], btype="bandpass", fs=sfreq, output="sos"), broad, axis=-1)


def condition_psds(eeg, sfreq):
    """Compute separate PSDs before and within the EDF's final 60 seconds."""
    n_samples = eeg.shape[1]
    after_samples = int(round(AFTER_FLASH_DURATION_S * sfreq))
    if n_samples < after_samples:
        raise ValueError("EDF is shorter than 60 seconds; a full after-flashing interval is unavailable.")
    split = n_samples - after_samples
    edge = int(round(EDGE_EXCLUDE_S * sfreq))
    win = int(round(PSD_WINDOW_S * sfreq))
    conditions = []
    for name, label, first, last in (
        ("earlier_recording", "Earlier recording", 0, split),
        ("after_flashing", "After flashing (last 60 s)", split, n_samples),
    ):
        metadata = dict(condition=name, label=label, start_edf_s=first / sfreq,
                        end_edf_s=last / sfreq, status="available")
        if last - first - 2 * edge < win:
            metadata.update(status="insufficient_duration", usable_start_edf_s=None,
                            usable_end_edf_s=None, usable_duration_s=0.0, psd_windows=0)
            conditions.append(dict(metadata=metadata, frequencies=None, psd=None))
            continue
        # Filter each condition separately so neither filtering nor Welch
        # windows combine samples across the condition boundary.
        broad = broadband_filter(eeg[:, first:last], sfreq)[:, edge:-edge]
        f, p = welch(broad, fs=sfreq, window="hann", nperseg=win, noverlap=win // 2,
                      nfft=win, detrend="constant", scaling="density", axis=-1)
        metadata.update(usable_start_edf_s=(first + edge) / sfreq,
                        usable_end_edf_s=(last - edge) / sfreq,
                        usable_duration_s=broad.shape[1] / sfreq,
                        psd_windows=1 + (broad.shape[1] - win) // (win - win // 2))
        conditions.append(dict(metadata=metadata, frequencies=f, psd=p))
    return conditions


def analyze_signals(eeg, sfreq, band, window_s, start_s=0.0):
    """Analyze two re-referenced signals; return PLV, windows, PSD and power."""
    eeg = np.asarray(eeg, dtype=float)
    band = np.asarray(band, dtype=float)
    if eeg.ndim != 2 or eeg.shape[0] != 2 or not np.isfinite(eeg).all():
        raise ValueError("Expected two finite EEG channels. Clean nonfinite data before analysis.")
    if not np.isfinite(sfreq) or sfreq <= 60:
        raise ValueError("Sampling rate must exceed 60 Hz for the 3-30 Hz bandpass.")
    if band.shape != (2,) or not np.isfinite(band).all() or not 3 <= band[0] < band[1] <= 30:
        raise ValueError("PLV band must have two ordered frequencies within 3-30 Hz.")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("PLV window duration must be positive and finite.")
    edge = int(round(EDGE_EXCLUDE_S * sfreq))
    window = int(round(window_s * sfreq))
    psd_window = int(round(PSD_WINDOW_S * sfreq))
    if window < 2 or eeg.shape[1] - 2 * edge < max(window, psd_window):
        raise ValueError("Selected interval is too short after excluding 2 s at each end.")
    broad = broadband_filter(eeg, sfreq)
    narrow = sosfiltfilt(butter(4, band, btype="bandpass", fs=sfreq, output="sos"), broad, axis=-1)
    analytic = hilbert(narrow, axis=-1)
    if np.any(np.std(narrow[:, edge:-edge], axis=-1) <= 1e-15):
        raise ValueError("A channel has no measurable signal in the PLV band; phase is undefined.")
    phase = np.angle(analytic[:, edge:-edge])
    broad = broad[:, edge:-edge]
    rows = []
    for offset in range(0, phase.shape[1] - window + 1, window):
        sl = slice(offset, offset + window)
        begin = start_s + (edge + offset) / sfreq
        rows.append(dict(window_number=len(rows) + 1, start_edf_s=begin,
                         end_edf_s=begin + window / sfreq,
                         plv=phase_locking_value(phase[0, sl], phase[1, sl])))
    windows = pd.DataFrame(rows)
    f, psd = welch(broad, fs=sfreq, window="hann", nperseg=psd_window,
                   noverlap=psd_window // 2, nfft=psd_window, detrend="constant",
                   scaling="density", axis=-1)
    psd_count = 1 + (broad.shape[1] - psd_window) // (psd_window - psd_window // 2)
    powers = []
    for p in psd:
        alpha = (f >= 8) & (f <= 13)
        powers.append(dict(alpha_8_13_power_uV2=band_power(f, p, 8, 13) * 1e12,
                           plv_band_power_uV2=band_power(f, p, *band) * 1e12,
                           alpha_peak_frequency_hz=float(f[alpha][np.argmax(p[alpha])]),
                           psd_windows=psd_count))
    summary = dict(overall_plv=phase_locking_value(phase[0], phase[1]),
                   mean_window_plv=float(windows.plv.mean()),
                   complete_plv_windows=len(windows), usable_duration_s=phase.shape[1] / sfreq,
                   plv_window_s=window / sfreq,
                   trailing_s_excluded_from_window_summary=(phase.shape[1] % window) / sfreq,
                   usable_start_edf_s=start_s + edge / sfreq,
                   usable_end_edf_s=start_s + (eeg.shape[1] - edge) / sfreq)
    return summary, windows, f, psd, powers


def plot_condition_psd(axes, conditions, channels, reference):
    for index, (ax, channel) in enumerate(zip(axes, channels)):
        for condition, color in zip(conditions, ("#157f99", "#da7b36")):
            meta = condition["metadata"]
            if condition["psd"] is None:
                ax.text(.02, .03, f"{meta['label']}: insufficient duration", transform=ax.transAxes, fontsize=8)
                continue
            f, p = condition["frequencies"], condition["psd"][index]
            display = (f >= 3) & (f <= 30)
            ax.semilogy(f[display], p[display] * 1e12, color=color,
                        label=f"{meta['label']}\n{meta['start_edf_s']:g}-{meta['end_edf_s']:g} s")
        label = f"{channel} minus {reference}" if reference else channel
        ax.set(title=f"{label}: Welch PSD", xlabel="Frequency on EDF clock (Hz)",
               ylabel="PSD (microvolt squared / Hz)", xlim=(3, 30))
        ax.legend(fontsize=8)


def save_figure(output, windows, conditions, channels, reference, band, summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    centers = (windows.start_edf_s + windows.end_edf_s) / 2
    axes[0].plot(centers, windows.plv, "o-", color="#157f99", markersize=3)
    axes[0].axhline(summary["overall_plv"], color="#da7b36", ls="--", label="Overall PLV")
    axes[0].set(xlabel="EDF time (s)", ylabel=f"{channels[0]}-{channels[1]} PLV",
                ylim=(0, 1), title=f"EEG phase locking: {band[0]:g}-{band[1]:g} Hz")
    axes[0].legend(fontsize=8)
    plot_condition_psd(axes[1:], conditions, channels, reference)
    label = f"referenced to {reference}" if reference else "as-recorded reference"
    fig.suptitle(f"Control EDF: {channels[0]} and {channels[1]}, {label}")
    fig.savefig(output / "eeg_plv_psd.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", sharey=True)
    plot_condition_psd(axes, conditions, channels, reference)
    fig.suptitle("Control PSD: earlier recording and after flashing\n2 s excluded at each condition edge")
    fig.savefig(output / "psd_before_after.png", dpi=160)
    plt.close(fig)


def save_condition_tables(output, conditions, channels, reference):
    rows, spectra = [], []
    for condition in conditions:
        meta = condition["metadata"]
        for index, channel in enumerate(channels):
            alpha_power = None
            if condition["psd"] is not None:
                f, p = condition["frequencies"], condition["psd"][index]
                alpha_power = band_power(f, p, 8, 13) * 1e12
                display = (f >= 3) & (f <= 30)
                spectra.append(pd.DataFrame(dict(channel=channel, condition=meta["condition"],
                                                  frequency_hz=f[display], psd_V2_per_Hz=p[display],
                                                  psd_uV2_per_Hz=p[display] * 1e12)))
            rows.append(dict(channel=channel, reference=reference or "as-recorded", **meta,
                             alpha_8_13_power_uV2=alpha_power))
    pd.DataFrame(rows).to_csv(output / "psd_condition_summary.csv", index=False)
    pd.concat(spectra, ignore_index=True).to_csv(output / "psd_by_condition.csv", index=False)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edf", type=Path, default=Path(EDF_PATH))
    parser.add_argument("--out", type=Path, help="Output folder (default: subject-specific EDF-only folder).")
    parser.add_argument("--channels", nargs=2, default=CHANNELS, metavar=("CHANNEL_A", "CHANNEL_B"))
    parser.add_argument("--reference", default=REFERENCE_CHANNEL, help="Reference channel, or 'none'.")
    parser.add_argument("--start", type=float, default=ANALYSIS_START_S, help="Interval start in EDF seconds.")
    parser.add_argument("--end", type=float, default=ANALYSIS_END_S, help="Exclusive end in EDF seconds.")
    parser.add_argument("--band", nargs=2, type=float, default=PLV_BAND_HZ, metavar=("LOW_HZ", "HIGH_HZ"))
    parser.add_argument("--window", type=float, default=PLV_WINDOW_S, help="PLV window duration in seconds.")
    parser.add_argument("--threshold", type=float, default=TRIGGER_THRESHOLD_V)
    parser.add_argument("--check-only", action="store_true", help="Inspect EDF/triggers without PLV/PSD.")
    args = parser.parse_args(argv)
    channels = list(args.channels)
    reference = None if args.reference.lower() == "none" else args.reference
    if channels[0] == channels[1] or reference in channels:
        parser.error("Choose two distinct channels and a separate reference (or 'none').")
    if not np.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be positive and finite.")
    raw = mne.io.read_raw_edf(args.edf, preload=False, verbose="ERROR")
    needed = channels + ([reference] if reference else [])
    missing = set(needed) - set(raw.ch_names)
    if missing:
        raise ValueError(f"Missing EDF channels: {sorted(missing)}. Available: {raw.ch_names}")
    has_trigger = TRIGGER_CHANNEL in raw.ch_names
    if has_trigger:
        needed.append(TRIGGER_CHANNEL)
    raw.pick(list(dict.fromkeys(needed)))
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    end_s = duration if args.end is None else args.end
    if not np.isfinite([args.start, end_s]).all() or not 0 <= args.start < end_s <= duration:
        parser.error(f"Require 0 <= --start < --end <= EDF duration ({duration:.3f} s).")
    first = int(np.ceil(args.start * sfreq))
    last = min(int(np.ceil(end_s * sfreq)), raw.n_times)
    output = args.out if args.out else OUTPUT_DIR / args.edf.stem
    output.mkdir(parents=True, exist_ok=True)
    warnings = ["PLV compares two EEG channels; no stimulus phase locking is estimated.",
                "After flashing is the final 60 EDF seconds; earlier recording may include multiple conditions.",
                "No automated artifact or annotation rejection, or significance testing, is applied.",
                "Common reference and volume conduction can contribute to inter-channel PLV."]
    triggers, edges, baseline = np.array([]), np.array([]), None
    if has_trigger:
        dc = raw.get_data(picks=[TRIGGER_CHANNEL])[0]
        if np.isfinite(dc).all():
            triggers, edges, baseline = detect_triggers(dc, sfreq, args.threshold)
        else:
            warnings.append("DC7 contains nonfinite samples; trigger inspection was skipped.")
    else:
        warnings.append("DC7 is absent; EEG analysis proceeds without trigger inspection.")
    pd.DataFrame(dict(edf_rising_edge_s=edges)).to_csv(output / "edf_trigger_edges.csv", index=False)
    pd.DataFrame(dict(marker_number=np.arange(1, len(triggers) + 1), edf_time_s=triggers,
                      previous_interval_s=np.r_[np.nan, np.diff(triggers)] if len(triggers) else [],
                      in_selected_interval=(triggers >= args.start) & (triggers < end_s))).to_csv(
        output / "edf_triggers.csv", index=False)
    report = dict(edf=str(args.edf.resolve()), analysis_mode="EEG-to-EEG PLV and channel PSD",
                  channels=channels, reference_channel=reference, sfreq_hz=sfreq, duration_edf_s=duration,
                  selected_start_edf_s=first / sfreq, selected_end_edf_s=last / sfreq,
                  plv_band_hz=list(args.band), requested_plv_window_s=args.window,
                  trigger_channel_present=has_trigger, edf_pulses=len(triggers),
                  trigger_baseline_V=baseline, trigger_threshold_V=args.threshold,
                  trigger_comparison_performed=False, clock_correction_applied=False,
                  edge_exclude_s=EDGE_EXCLUDE_S, psd_window_s=PSD_WINDOW_S, broadband_filter_hz=[3, 30],
                  after_flash_duration_s=AFTER_FLASH_DURATION_S,
                  psd_condition_split_source="User-defined: final 60 s of the original EDF are after flashing",
                  notch_hz=50 if sfreq > 100 else None, analysis_completed=False, warnings=warnings,
                  versions={"mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__})
    report_path = output / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"EDF: {args.edf.name}; {duration:.3f} s at {sfreq:g} Hz")
    print(f"Detected {len(triggers)} DC7 pulses. No external trigger comparison or clock correction.")
    print(f"Selected EDF interval: {first / sfreq:.3f}-{last / sfreq:.3f} s")
    for warning in warnings:
        print(f"NOTE: {warning}")
    if args.check_only:
        print(f"EDF inspection saved to {output.resolve()}")
        return report
    full_eeg = raw.get_data(picks=channels)
    if reference:
        full_eeg -= raw.get_data(picks=[reference])
    conditions = condition_psds(full_eeg, sfreq)
    eeg = full_eeg[:, first:last]
    summary, windows, f, psd, powers = analyze_signals(eeg, sfreq, args.band, args.window, first / sfreq)
    windows.insert(0, "channel_pair", "-".join(channels))
    windows["band_low_hz"], windows["band_high_hz"] = args.band
    windows.to_csv(output / "eeg_plv_windows.csv", index=False)
    pd.DataFrame([dict(channel_pair="-".join(channels), **summary)]).to_csv(output / "eeg_plv_summary.csv", index=False)
    power_table = pd.DataFrame([dict(channel=channel, reference=reference or "as-recorded", **power)
                                for channel, power in zip(channels, powers)])
    power_table.to_csv(output / "channel_psd_summary.csv", index=False)
    display = (f >= 3) & (f <= 30)
    pd.concat([pd.DataFrame(dict(channel=channel, frequency_hz=f[display],
                                 psd_V2_per_Hz=p[display], psd_uV2_per_Hz=p[display] * 1e12))
               for channel, p in zip(channels, psd)], ignore_index=True).to_csv(output / "psd_spectra.csv", index=False)
    condition_rows = save_condition_tables(output, conditions, channels, reference)
    save_figure(output, windows, conditions, channels, reference, args.band, summary)
    report.update(analysis_completed=True, plv_summary=summary, psd_summary=power_table.to_dict(orient="records"),
                  psd_conditions=condition_rows)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{channels[0]}-{channels[1]} PLV: overall={summary['overall_plv']:.4f}; "
          f"mean of {len(windows)} complete windows={summary['mean_window_plv']:.4f}")
    print(power_table.to_string(index=False))
    for condition in conditions:
        meta = condition["metadata"]
        print(f"{meta['label']}: {meta['start_edf_s']:.3f}-{meta['end_edf_s']:.3f} s; "
              f"PSD uses {meta['usable_duration_s']:.3f} s ({meta['psd_windows']} Welch windows).")
    print(f"PSD comparison figure: {(output / 'psd_before_after.png').resolve()}")
    print(f"Saved EDF-only results to {output.resolve()}")
    return report


if __name__ == "__main__":
    main()
