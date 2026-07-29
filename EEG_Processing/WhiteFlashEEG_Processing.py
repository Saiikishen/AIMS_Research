"""
EEG Alpha-Band (8-14 Hz) Analysis: PSD + Time-Frequency per activity interval
==============================================================================

Loads a single .edf recording, applies a 50 Hz notch filter, then for each
labeled time interval (normal activity + 5 task intervals) generates a
SEPARATE figure containing:
    (1) Power Spectral Density (Welch), with the 8-14 Hz alpha band shaded
    (2) Time-frequency representation (Morlet wavelets), with the alpha
        band marked by dashed horizontal lines

Requirements:
    pip install mne matplotlib numpy

Tested against MNE >= 1.7 (uses the modern `raw.compute_psd()` /
`raw.compute_tfr()` API). If you're on an older MNE version, upgrade with:
    pip install --upgrade mne
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
import matplotlib.pyplot as plt
import mne

# ----------------------------------------------------------------------------
# 1. CONFIG — edit these for your setup
# ----------------------------------------------------------------------------

EDF_PATH = r"C:\Users\saiik\Downloads\NIBY-Photic\NIBY-Photic\NIBY~ VEEG_399df063-f9bb-44a4-adec-3fc7b5954c7d.edf"          # <-- path to your .edf recording
OUTPUT_DIR = "eeg_figures"          # where PNGs get saved
PICKS = ["O1", "O2"]              # channel selection (list of channel names e.g., ['OP1', 'OP2'])

NOTCH_FREQ = 50.0                   # mains hum, per your setup
ALPHA_BAND = (8.0, 22.0)            # band of interest

# PSD display range (gives context around alpha, not just the band itself)
PSD_FMIN, PSD_FMAX = 1.0, 40.0

# Time-frequency frequency range (also gives a bit of context around alpha)
TFR_FMIN, TFR_FMAX, TFR_STEP = 5.0, 16.0, 1.0

# Time intervals to analyze: name -> (tmin, tmax) in seconds, as given
PERIODS = {
    "normal_activity": (1, 294),
    "activity_1": (357, 361),
    "activity_2": (376, 382),
    "activity_3": (397, 402),
    "activity_4": (417, 425),
    "activity_5": (437, 445),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. LOAD + PREPROCESS (notch filter only, as requested)
# ----------------------------------------------------------------------------

print(f"Loading {EDF_PATH} ...")
raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)

# If channel types weren't inferred correctly from the EDF header, uncomment:
# raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

print(raw.info)
print("Channels:", raw.ch_names)



print(f"Applying {NOTCH_FREQ} Hz notch filter on channels: {PICKS} ...")
raw.notch_filter(freqs=NOTCH_FREQ, picks=PICKS, verbose=False)

sfreq = raw.info["sfreq"]
alpha_summary = {}  # collects mean alpha power per period for a quick printout

# ----------------------------------------------------------------------------
# 3. PER-PERIOD ANALYSIS
# ----------------------------------------------------------------------------

for name, (tmin, tmax) in PERIODS.items():

    if tmax > raw.times[-1]:
        print(f"[skip] {name}: tmax={tmax}s exceeds recording length "
              f"({raw.times[-1]:.1f}s)")
        continue

    print(f"\nProcessing '{name}' ({tmin}-{tmax}s) ...")
    seg = raw.copy().crop(tmin=tmin, tmax=tmax)

    # --- PSD (Welch) ---------------------------------------------------
    n_samples = seg.n_times
    n_fft = min(2048, n_samples)  # keep n_fft valid for short segments

    spectrum = seg.compute_psd(
        method="welch",
        picks=PICKS,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        n_fft=n_fft,
        verbose=False,
    )
    psd_data, psd_freqs = spectrum.get_data(return_freqs=True)  # (n_ch, n_freqs)
    psd_db = 10 * np.log10(psd_data)

    # mean alpha-band power (linear, not dB) per channel -> overall mean
    alpha_mask = (psd_freqs >= ALPHA_BAND[0]) & (psd_freqs <= ALPHA_BAND[1])
    alpha_summary[name] = psd_data[:, alpha_mask].mean()

    # --- Time-Frequency (Morlet wavelets) -------------------------------
    tfr_freqs = np.arange(TFR_FMIN, TFR_FMAX + TFR_STEP, TFR_STEP)
    n_cycles = tfr_freqs / 2.0

    tfr = seg.compute_tfr(
        method="morlet",
        freqs=tfr_freqs,
        n_cycles=n_cycles,
        picks=PICKS,
        verbose=False,
    )
    # tfr.data shape: (n_channels, n_freqs, n_times) -> average across channels
    tfr_power = tfr.data.mean(axis=0)
    tfr_times = tfr.times

    # --- Figure: one per period, PSD on top / TFR on bottom ------------
    fig, (ax_psd, ax_tfr) = plt.subplots(2, 1, figsize=(9, 7))
    fig.suptitle(f"{name}  ({tmin}-{tmax}s)", fontsize=13, fontweight="bold")

    # PSD subplot
    for ch_idx, ch_name in enumerate(spectrum.ch_names):
        ax_psd.plot(psd_freqs, psd_db[ch_idx], linewidth=0.8, alpha=0.7,
                    label=ch_name)
    ax_psd.axvspan(*ALPHA_BAND, color="orange", alpha=0.2, label="Alpha (8-14 Hz)")
    ax_psd.set_xlim(PSD_FMIN, PSD_FMAX)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (dB)")
    ax_psd.set_title("Power Spectral Density (Welch)")
    if len(spectrum.ch_names) <= 10:
        ax_psd.legend(fontsize=7, loc="upper right", ncol=2)

    # TFR subplot
    mesh = ax_tfr.pcolormesh(tfr_times, tfr_freqs, tfr_power,
                              shading="auto", cmap="viridis")
    ax_tfr.axhline(ALPHA_BAND[0], color="white", linestyle="--", linewidth=1)
    ax_tfr.axhline(ALPHA_BAND[1], color="white", linestyle="--", linewidth=1)
    ax_tfr.set_xlabel("Time (s)")
    ax_tfr.set_ylabel("Frequency (Hz)")
    ax_tfr.set_title("Time-Frequency Power (Morlet wavelets, dashed lines = alpha band)")
    fig.colorbar(mesh, ax=ax_tfr, label="Power")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(out_path, dpi=150)
    print(f"  saved -> {out_path}")

# ----------------------------------------------------------------------------
# 4. SUMMARY
# ----------------------------------------------------------------------------

print("\nMean alpha-band (8-12 Hz) power by period:")
for name, val in alpha_summary.items():
    print(f"  {name:>16s}: {val:.4e}")

plt.show()