"""sEEG PSD: edit the settings below, then run python SEEG/psd.py.

Pipeline: load -> optional reference -> MNE FIR filters -> crop -> Welch PSD.
CLI override: python SEEG/psd.py --edf recording.edf --tmin 360 --tmax 370
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
from scipy.signal import welch


# ============================================================
# Settings -- edit these, just like in your IAF script
# ============================================================

EDF_PATH = r"C:\Users\saiik\Downloads\Event2.edf"
TMIN = 362.0  # Seconds from EDF start, inclusive. None = prompt.
TMAX = 378.0  # Seconds from EDF start, exclusive. None = prompt.

CHANNEL_GROUPS = {
   # "AF": [f"AF{i}" for i in range(1, 13)],
    #"dPO": [f"dPO{i}" for i in range(1, 17)],
    #"I": [f"I{i}" for i in range(1, 17)],
    "AF": ["AF8", "AF9","AF10","AF11","AF12"],
}
REFERENCE_MODE = "bipolar"  # "bipolar", "original", or "channel".
REFERENCE_CHANNEL = None   # EDF reference label, only for mode "channel".
BAD_CHANNELS = []         # Manually inspected bad contacts, e.g. ["AF3"].
ALLOW_MISSING_CHANNELS = False

BANDPASS = (1.0, 100.0)
MAINS_NOTCH_HZ = 50.0
NOTCH_WIDTH_HZ = 2.0      # FIR stopband 49-51 Hz, allowing modest mains drift.
NOTCH_TRANS_BANDWIDTH_HZ = 1.0

WINDOW_SEC = 2.0
WINDOW_OVERLAP = 0.5
PSD_UNIT = "uV2/Hz"     # Plot/export dB re 1 µV²/Hz. Alternative: "V2/Hz".
OUTPUT_DIR = None         # None = beside EDF; otherwise choose a parent folder.


# ============================================================
# Pipeline
# ============================================================

def canonical_contact(label):
    """Accept aliases such as EEG AF01-Ref, but not bipolar channel labels."""
    label = re.sub(r"^(?:EEG|SEEG)\s+", "", label.strip(), flags=re.I)
    label = re.sub(r"[\s_-]*REF$", "", label, flags=re.I)
    match = re.fullmatch(r"(AF|dPO|I)[\s_-]*0*(\d+)", label, flags=re.I)
    if not match:
        return None
    shaft = {"af": "AF", "dpo": "dPO", "i": "I"}[match[1].lower()]
    return f"{shaft}{int(match[2])}"


def load_raw(edf_path, requested, reference_channel=None, bad_channels=(), allow_missing=False):
    """Load the full recording for requested contacts before filtering."""
    path = Path(edf_path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".edf":
        raise ValueError(f"EDF file not found or incorrect extension: {path}")
    with mne.io.read_raw_edf(path, preload=False, verbose=False) as header:
        mapping = {}
        for label in header.ch_names:
            contact = canonical_contact(label)
            if contact in requested:
                if contact in mapping:
                    raise ValueError(f"Multiple EDF labels match {contact}.")
                mapping[contact] = label
        reference = None
        if reference_channel:
            reference = (reference_channel if reference_channel in header.ch_names
                         else mapping.get(canonical_contact(reference_channel)))
            if reference is None:
                raise ValueError(f"Reference channel not found: {reference_channel}")
    missing = [ch for ch in requested if ch not in mapping]
    if missing:
        if not allow_missing:
            raise ValueError(f"Missing channels: {missing}. Check labels or set ALLOW_MISSING_CHANNELS=True.")
        print(f"[WARN] Missing channels: {missing}")
    if set(bad_channels) - set(requested):
        raise ValueError("BAD_CHANNELS must use requested contact names such as AF3.")
    channels = [ch for ch in requested if ch in mapping and ch not in bad_channels]
    if reference:
        channels = [ch for ch in channels if mapping[ch] != reference]
        if reference in [mapping[ch] for ch in bad_channels if ch in mapping]:
            raise ValueError("A bad channel cannot be used as the reference.")
    if not channels:
        raise ValueError("No usable analysis channels remain.")
    include = list(dict.fromkeys([mapping[ch] for ch in channels] + ([reference] if reference else [])))
    raw = mne.io.read_raw_edf(path, include=include, preload=True, verbose=False)
    units = getattr(raw, "_orig_units", {}) or {}
    if any(units.get(ch) not in {"V", "mV", "uV", "µV", "μV", "nV"} for ch in include):
        raise ValueError("Selected EDF signals need valid voltage units; verify the EDF header.")
    raw.rename_channels({mapping[ch]: ch for ch in channels})
    raw.set_channel_types({ch: "seeg" for ch in raw.ch_names}, verbose=False)
    print(f"[INFO] Loaded {len(channels)} contacts | sfreq={raw.info['sfreq']:g} Hz | "
          f"duration={raw.n_times / raw.info['sfreq']:.3f} s")
    details = {"channel_mapping": {ch: mapping[ch] for ch in channels},
               "missing_channels": missing, "bad_channels": list(bad_channels),
               "edf_physical_units": {ch: units[ch] for ch in include}}
    return raw, channels, reference, details


def rereference_to_channel(raw, channels, ref_channel):
    """Optional contact-minus-reference derivations, as in the IAF example."""
    if not ref_channel:
        print("[INFO] Using the as-recorded EDF reference.")
        return raw
    temporary = [f"{ch}__referenced" for ch in channels]
    raw = mne.set_bipolar_reference(raw, anode=channels,
                                    cathode=[ref_channel] * len(channels),
                                    ch_name=temporary, drop_refs=True, verbose=False)
    raw.rename_channels(dict(zip(temporary, channels)))
    print(f"[INFO] Re-referenced analysis contacts to {ref_channel}.")
    return raw


def rereference_bipolar(raw, groups):
    """Adjacent contacts on each shaft: AF1-AF2, dPO1-dPO2, I10-I11, etc.

    Pair the configured consecutive positions before checking availability,
    so an absent/bad I11 can never turn I10 into I10-I12.
    """
    anodes, cathodes, pair_names = [], [], []
    bipolar_groups, derivations = {}, {}
    for group, contacts in groups.items():
        bipolar_groups[group] = []
        for anode, cathode in zip(contacts[:-1], contacts[1:]):
            # Only consecutive contact numbers from the same shaft may pair.
            a = re.fullmatch(r"(AF|dPO|I)(\d+)", anode)
            b = re.fullmatch(r"(AF|dPO|I)(\d+)", cathode)
            if not a or not b or a[1] != b[1] or int(b[2]) != int(a[2]) + 1:
                continue
            if anode not in raw.ch_names or cathode not in raw.ch_names:
                continue
            name = f"{anode}-{cathode}"
            anodes.append(anode)
            cathodes.append(cathode)
            pair_names.append(name)
            bipolar_groups[group].append(name)
            derivations[name] = {"anode": anode, "cathode": cathode,
                                 "operation": f"{anode} minus {cathode}"}
    if not pair_names:
        raise ValueError("No valid adjacent contacts remain for bipolar referencing.")
    raw = mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes,
                                    ch_name=pair_names, drop_refs=True, verbose=False)
    raw.pick(pair_names)  # Exclude isolated contacts without a valid adjacent pair.
    print(f"[INFO] Created {len(pair_names)} bipolar channels (contact n minus contact n+1).")
    return raw, bipolar_groups, derivations


def validate_window(raw, tmin, tmax, window_sec):
    """Require a manually inspected interval away from annotated artifacts."""
    sfreq = raw.info["sfreq"]
    if not np.isfinite([tmin, tmax, window_sec]).all() or not 0 <= tmin < tmax <= raw.n_times / sfreq:
        raise ValueError("Require finite times with 0 <= TMIN < TMAX <= EDF duration.")
    if window_sec < 2 or tmax - tmin < window_sec:
        raise ValueError("WINDOW_SEC must be >= 2 s and fit inside the selected interval.")
    # A 10 s margin covers the support of the default FIR filters.
    for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration,
                                           raw.annotations.description):
        onset -= raw.first_time
        if str(description).lower().startswith(("bad", "edge")):
            if onset <= tmax + 10 and onset + duration >= tmin - 10:
                raise ValueError(f"{description!r} at {onset:g} s is within 10 s of the window. "
                                 "Choose a clean interval away from annotated artifacts/boundaries.")
    if tmin < 10 or tmax > raw.n_times / sfreq - 10:
        print("[WARN] Window is near a recording boundary; filter edge effects may remain.")


def apply_filters(raw, bandpass, notch_hz, notch_width=2.0, transition_bandwidth=1.0):
    """Same MNE FIR filtering concepts as the supplied IAF script."""
    nyquist = raw.info["sfreq"] / 2
    if not 0 < bandpass[0] < bandpass[1] < nyquist:
        raise ValueError(f"Bandpass must satisfy 0 < low < high < {nyquist:g} Hz.")
    if not 0 < notch_width or not 0 < transition_bandwidth:
        raise ValueError("Notch width and transition bandwidth must be positive.")
    if not (notch_width / 2 + transition_bandwidth / 2 < notch_hz <
            nyquist - notch_width / 2 - transition_bandwidth / 2):
        raise ValueError("Notch and its transitions must lie between 0 Hz and Nyquist.")
    raw = raw.copy()
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], picks="all", method="fir",
               phase="zero", fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[notch_hz], picks="all", method="fir", phase="zero",
                     fir_design="firwin", notch_widths=notch_width,
                     trans_bandwidth=transition_bandwidth, verbose=False)
    print(f"[INFO] MNE FIR bandpass {bandpass[0]:g}-{bandpass[1]:g} Hz; notch {notch_hz:g} Hz.")
    return raw


def crop(raw, tmin, tmax):
    """Select samples in [tmin, tmax), without an extra right-endpoint sample."""
    sfreq = raw.info["sfreq"]
    start = int(np.ceil(tmin * sfreq - 1e-9))
    stop = int(np.ceil(tmax * sfreq - 1e-9))
    return raw.copy().crop(tmin=start / sfreq, tmax=(stop - 1) / sfreq)


def welch_psd(data, sfreq, window_sec, overlap):
    """Hann windows, mean removal and linear-power averaging; output V^2/Hz."""
    if not np.isfinite([window_sec, overlap]).all() or window_sec < 2 or not 0 <= overlap < 1:
        raise ValueError("Require WINDOW_SEC >= 2 and 0 <= WINDOW_OVERLAP < 1.")
    nperseg = int(round(window_sec * sfreq))
    noverlap = int(round(nperseg * overlap))
    if len(data) < nperseg or noverlap >= nperseg:
        raise ValueError("Not enough samples for a full Welch window, or overlap is too large.")
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite or BAD samples in the interval; inspect a clean window.")
    return welch(data, fs=sfreq, window="hann", nperseg=nperseg, noverlap=noverlap,
                 nfft=nperseg, detrend="constant", scaling="density", average="mean")


def compute_channel_psds(raw, channels, window_sec, overlap):
    data = raw.get_data(picks=channels, reject_by_annotation="NaN", verbose=False)
    psds = {}
    for i, ch in enumerate(channels):
        freqs, psds[ch] = welch_psd(data[i], raw.info["sfreq"], window_sec, overlap)
    return freqs, psds


def to_db(psd, unit):
    scale = {"V2/Hz": 1.0, "uV2/Hz": 1e12}[unit]
    return 10 * np.log10(np.maximum(psd * scale, np.finfo(float).tiny))


def plot_psds(freqs, psds, groups, config, out_dir):
    """Individual traces and a linear-power group average, like posterior_avg."""
    unit_label = "V²/Hz" if config["PSD_UNIT"] == "V2/Hz" else "µV²/Hz"
    keep = (freqs >= config["BANDPASS"][0]) & (freqs <= config["BANDPASS"][1])
    for group, requested in groups.items():
        channels = [ch for ch in requested if ch in psds]
        if not channels:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, ch in enumerate(channels):
            ax.plot(freqs[keep], to_db(psds[ch][keep], config["PSD_UNIT"]),
                    label=ch, color=plt.get_cmap("tab20")(index), alpha=0.75, lw=0.8)
        group_avg = np.mean([psds[ch] for ch in channels], axis=0)
        ax.plot(freqs[keep], to_db(group_avg[keep], config["PSD_UNIT"]),
                color="black", lw=2, label=f"{group} mean PSD")
        ax.set(xlim=(0, config["BANDPASS"][1]), xlabel="Frequency (Hz)",
               ylabel=f"PSD (dB re 1 {unit_label})",
               title=f"{group} | {config['TMIN']:g}-{config['TMAX']:g} s | "
                     f"{config['REFERENCE_MODE']} reference | Welch PSD")
        ax.set_xticks(np.arange(0, config["BANDPASS"][1] + 1e-9, 5.0))
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"PSD_{group}.png", dpi=150)
        plt.close(fig)


def save_psds(freqs, psds, bandpass, unit, out_dir):
    keep = (freqs >= bandpass[0]) & (freqs <= bandpass[1])
    names = list(psds)
    values = np.array([psds[ch][keep] for ch in names])
    for filename, data in (("psd_V2_per_Hz.csv", values),
                           ("psd_uV2_per_Hz.csv", values * 1e12),
                           ("psd_dB.csv", to_db(values, unit))):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frequency_Hz", *names])
            writer.writerows(zip(freqs[keep], *data))


def plot_signal_and_psd(freqs, psds, filtered, config, out_dir):
    """One two-panel image per channel: PSD and the corresponding waveform."""
    channels = list(psds)
    filtered_data = filtered.get_data(picks=channels) * 1e6
    times = filtered.times + filtered.first_time
    unit_label = "V²/Hz" if config["PSD_UNIT"] == "V2/Hz" else "µV²/Hz"
    keep = (freqs >= config["BANDPASS"][0]) & (freqs <= config["BANDPASS"][1])
    files = []
    for index, channel in enumerate(channels):
        fig, (ax_psd, ax_signal) = plt.subplots(2, 1, figsize=(12, 8),
                                               layout="constrained")
        ax_psd.plot(freqs[keep], to_db(psds[channel][keep], config["PSD_UNIT"]),
                    color="tab:blue", linewidth=1)
        ax_psd.set(title="Welch PSD of the filtered signal", xlim=(0, config["BANDPASS"][1]),
                   xlabel="Frequency (Hz)", ylabel=f"PSD (dB re 1 {unit_label})")
        ax_psd.set_xticks(np.arange(0, config["BANDPASS"][1] + 1e-9, 5.0))
        ax_signal.plot(times, filtered_data[index], color="tab:blue", linewidth=0.8,
                       label="Filtered signal used for PSD")
        ax_signal.set(title="Filtered signal used for PSD", xlim=(config["TMIN"], config["TMAX"]),
                      xlabel="Time from EDF start (s)", ylabel="Amplitude (µV)")
        for ax in (ax_psd, ax_signal):
            ax.grid(True, linestyle="--", alpha=0.3)
        fig.suptitle(f"{channel} | {config['TMIN']:g}-{config['TMAX']:g} s | "
                     f"{config['REFERENCE_MODE']} reference", fontsize=13)
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", channel)
        filename = f"PSD_signal_{safe_name}.png"
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
        files.append(filename)
    return files


def run(config):
    if config["PSD_UNIT"] not in ("V2/Hz", "uV2/Hz"):
        raise ValueError("PSD_UNIT must be 'V2/Hz' or 'uV2/Hz'.")
    mode = config["REFERENCE_MODE"]
    if mode not in ("bipolar", "original", "channel"):
        raise ValueError("REFERENCE_MODE must be 'bipolar', 'original', or 'channel'.")
    if mode == "channel" and not config["REFERENCE_CHANNEL"]:
        raise ValueError("Set REFERENCE_CHANNEL when REFERENCE_MODE is 'channel'.")
    requested = [ch for group in config["CHANNEL_GROUPS"].values() for ch in group]
    raw, channels, reference, details = load_raw(
        config["EDF_PATH"], requested, config["REFERENCE_CHANNEL"] if mode == "channel" else None,
        config["BAD_CHANNELS"], config["ALLOW_MISSING_CHANNELS"])
    validate_window(raw, config["TMIN"], config["TMAX"], config["WINDOW_SEC"])
    groups, derivations = config["CHANNEL_GROUPS"], {}
    if mode == "bipolar":
        raw, groups, derivations = rereference_bipolar(raw, groups)
        channels = [ch for group in groups.values() for ch in group]
        reference = "adjacent bipolar: contact n minus contact n+1"
    else:
        raw = rereference_to_channel(raw, channels, reference)
    raw = apply_filters(raw, config["BANDPASS"], config["MAINS_NOTCH_HZ"],
                        config["NOTCH_WIDTH_HZ"], config["NOTCH_TRANS_BANDWIDTH_HZ"])
    segment = crop(raw, config["TMIN"], config["TMAX"])
    freqs, psds = compute_channel_psds(segment, channels, config["WINDOW_SEC"], config["WINDOW_OVERLAP"])
    sfreq = float(raw.info["sfreq"])
    nperseg = int(round(config["WINDOW_SEC"] * sfreq))
    noverlap = int(round(nperseg * config["WINDOW_OVERLAP"]))
    n_segments = 1 + (segment.n_times - nperseg) // (nperseg - noverlap)
    if n_segments < 5:
        print(f"[WARN] Only {n_segments} Welch segments/channel; use a longer stationary window if appropriate.")

    parent = Path(config["OUTPUT_DIR"]) if config["OUTPUT_DIR"] else Path(config["EDF_PATH"]).parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = parent / f"{Path(config['EDF_PATH']).stem}_PSD_FIR_{mode}_{config['TMIN']:g}-{config['TMAX']:g}s_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    plot_psds(freqs, psds, groups, config, out_dir)
    combined_plots = plot_signal_and_psd(freqs, psds, segment, config, out_dir)
    save_psds(freqs, psds, config["BANDPASS"], config["PSD_UNIT"], out_dir)
    summary = {"config": config, "channels_used": channels, "reference": reference or "as-recorded",
               "output_groups": groups, "bipolar_derivations": derivations,
               "signal_and_psd_plots": combined_plots,
               "waveform_units": "uV",
               "waveform_traces": ["after referencing and filters; used for PSD"],
               "sfreq_hz": sfreq, "window_samples": segment.n_times,
               "sample_interval_s": [segment.first_samp / sfreq,
                                     (segment.first_samp + segment.n_times) / sfreq],
               "welch_segments_per_channel": n_segments, "frequency_bin_spacing_hz": sfreq / nperseg,
               "filter": "MNE FIR, firwin, zero phase, full recording before crop",
               "welch": "Hann, constant detrend, density scaling, mean in linear power, no zero padding",
               "linear_psd_units": "V^2/Hz (also exported in uV^2/Hz)",
               "db_reference": f"1 {config['PSD_UNIT']}",
               "group_average": "arithmetic mean of contact PSDs before log conversion",
               "artifact_policy": "manually inspected window; reject BAD/EDGE annotations within 10 s",
               "software": {"mne": mne.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
               "output_dir": str(out_dir.resolve()), **details}
    (out_dir / "psd_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[RESULT] {len(channels)} channels | {n_segments} Welch segments/channel | "
          f"{sfreq / nperseg:g} Hz bins\n[OUTPUT] {out_dir.resolve()}")
    return summary


if __name__ == "__main__":
    config = {k: v for k, v in globals().items() if k.isupper()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--output", help="Parent directory for a new results folder")
    parser.add_argument("--reference", choices=("bipolar", "original", "channel"))
    parser.add_argument("--reference-channel", help="EDF reference label for mode 'channel'")
    args = parser.parse_args()
    for key, value in (("EDF_PATH", args.edf), ("TMIN", args.tmin),
                       ("TMAX", args.tmax), ("OUTPUT_DIR", args.output),
                       ("REFERENCE_MODE", args.reference), ("REFERENCE_CHANNEL", args.reference_channel)):
        if value is not None:
            config[key] = value
    try:
        if not config["EDF_PATH"]:
            config["EDF_PATH"] = input("EDF file path: ").strip().strip('\"').strip("'")
        for key in ("TMIN", "TMAX"):
            if config[key] is None:
                config[key] = float(input(f"{key} (seconds): "))
        run(config)
    except (ValueError, OSError, EOFError) as exc:
        parser.exit(2, f"Error: {exc}")
