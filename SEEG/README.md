# sEEG PSD using the IAF script's structure

Edit EDF_PATH, TMIN, and TMAX at the top of psd.py, then run:

    python -m pip install -r SEEG/requirements.txt
    python SEEG/psd.py

Edit the settings for your EDF path, desired interval and contact selection.
To analyze only I10-I11, set CHANNEL_GROUPS = {"I": ["I10", "I11"]} and
REFERENCE_MODE = "bipolar".
Set EDF_PATH to an empty string and TMIN/TMAX to None for interactive prompts.
Command-line overrides are also supported:

    python SEEG/psd.py --edf "C:\data\recording.edf" --tmin 60 --tmax 120

## Pipeline

1. Load the full recording for AF1-AF12, dPO1-dPO16, and I1-I16.
2. By default, apply adjacent-contact bipolar referencing within each shaft:
   AF1-AF2 through AF11-AF12, dPO1-dPO2 through dPO15-dPO16, and I1-I2 through
   I15-I16. In particular, I10-I11 means the I10 voltage minus the I11 voltage.
   This produces 41 bipolar channels from 44 contacts. The final contact on
   each shaft serves as the previous contact's reference; there is no wraparound.
   Missing/bad contacts remove affected pairs and are never bridged.
3. Apply MNE zero-phase FIR filters with firwin: 1-100 Hz bandpass and a 50 Hz
   notch, before cropping. NOTCH_WIDTH_HZ=2.0 sets a 49-51 Hz stopband to handle
   the observed mains drift. The FIR transition bandwidth is 1 Hz.
4. Crop samples in [TMIN, TMAX), seconds from EDF start.
5. Calculate each channel's PSD using scipy.signal.welch: 4-second Hann
   windows, 50% overlap, constant detrending, one-sided density scaling,
   arithmetic averaging in linear power, and no zero padding.
6. Plot each group with its bipolar channels and a black mean PSD, like the
   posterior average in the IAF example. Average power before converting to dB.

These FIR filters replace the previous script's SciPy IIR filters. Their results
differ near filter edges. There is no IAF peak search or eyes-closed validity
rule: this script calculates sEEG PSD for the requested interval.

REFERENCE_MODE="bipolar" is the default. Set REFERENCE_MODE="original" to use
the recorded reference and keep 44 individual contacts. Set REFERENCE_MODE="channel"
and REFERENCE_CHANNEL to an EDF label for a single common reference, like the
IAF example. A reference contact is excluded from its own PSD calculation.
The command-line equivalents are --reference bipolar, --reference original,
or --reference channel --reference-channel Cz. Use Cz only if appropriate and
actually present; the default does not assume a Cz electrode.

Bipolar referencing operates on voltage signals before PSD calculation. It is
not subtraction of the two PSD curves. Confirm that consecutive contact numbers
correspond to physically adjacent contacts on each shaft.

## Units

PSD_UNIT="uV2/Hz" is the default: MNE supplies volts, Welch returns V²/Hz,
and plots use 10*log10(PSD * 1e12), labeled dB re 1 µV²/Hz. This is numerically
120 dB higher than dB re 1 V²/Hz for the same nonzero power.

Set PSD_UNIT="V2/Hz" to use the original IAF example's display units. Negative dB is valid.
Linear CSVs are exported in both units regardless of the display choice.
The logarithm uses a tiny numerical floor only for zero power.

## Outputs

Each run creates a timestamped folder beside the EDF, or inside OUTPUT_DIR:

- PSD_AF.png, PSD_dPO.png, PSD_I.png: channel curves and group mean PSD.
- PSD_signal_I10-I11.png (one image per selected output channel): two subplots,
  with filtered-signal PSD above and the filtered time-domain waveform below.
  The waveform panel shows only the filtered signal used for PSD, in µV.
  For bipolar mode, this is filtered I10 minus I11. Its time axis
  is seconds from the EDF start, using the same TMIN/TMAX interval as the PSD.
- psd_V2_per_Hz.csv and psd_uV2_per_Hz.csv: 1-100 Hz frequency rows and bipolar columns.
- psd_dB.csv: dB values using the selected PSD_UNIT.
- psd_summary.json: configuration, mapping, reference, exact sample interval,
  bipolar anode/cathode mapping, Welch segment counts, units, package versions
  and output location.

## Clean intervals

Use a visually inspected interval appropriate to the experiment. The script
refuses windows within 10 seconds of BAD/EDGE annotations, instead of joining
disjoint samples. This margin covers the default FIR supports; reassess it if
you narrow the filter transition bands. Nonfinite samples in the selected
interval cause an error. Unannotated artifacts are not automatically detected.

Set BAD_CHANNELS to exclude inspected bad contacts. Missing channels require
ALLOW_MISSING_CHANNELS=True; ambiguous labels always raise an error. A bad
channel cannot be used as a reference.

The EDF sampling rate must exceed 200 Hz with the default bandpass, and EDF
voltage calibration must be correct. Selected contacts and any reference should
share a sampling rate; MNE upsamples mixed-rate signals. Loading the full selected
recording, as in the supplied example, can require substantial RAM for long files.

At least one full Welch window is required. A 10-second interval gives four
overlapping 4-second windows; longer stationary intervals provide more averaging.
The default frequency-bin spacing is 0.25 Hz; Hann smoothing is broader than
bin spacing. Group mean PSD is the mean of individual PSDs, not the PSD of an
averaged voltage signal.

Power near the notch and bandpass transitions is altered. Only 50 Hz is notched;
interference near the 100 Hz harmonic can remain.

References:
[MNE FIR/notch filtering](https://mne.tools/stable/generated/mne.filter.notch_filter.html),
[SciPy Welch PSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
