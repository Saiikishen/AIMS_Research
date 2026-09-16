#!/usr/bin/env python3


from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from . import PhaseLocking as phase
else:
    import PhaseLocking as phase


# CONFIG - retain the matching EDF and stimulus CSV for the current subject.
EDF_PATH = (
    r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\CONTROL\SUB15~ JACOB_0e04f0e8-fb17-4f50-8a72-31c9e4d27116.edf"
)
STIMULUS_CSV_PATH = (
    r"C:\Users\saiik\Downloads\miriam expt edf\Jacob_sub15\control excel\stimulus_log.csv"
)

REFERENCE_CHANNEL = "Cz"
REREFERENCE_CHANNELS = ["O1", "O2"]
EEG_CHANNELS = ["O1", "O2"]
DC7_CHANNEL_NAME = "DC7"
DC7_THRESHOLD = 0.005
MIN_TRIGGER_ISI_S = 1.0
TRIGGER_MATCH_WARN_TOL_S = 0.05
MERGE_ASOF_TOL_S = 0.05

WELCH_NPERSEG_S = phase.PSD_WINDOW_S
BOUNDARY_EXCLUDE_S = phase.BOUNDARY_EXCLUDE_S
AFTER_FLASH_DELAY_S = phase.AFTER_FLASH_DELAY_S
OUTPUT_DIR = "flash_eeg_plv_psd_results"


def collapse_close_triggers(times, min_isi_s=MIN_TRIGGER_ISI_S):
    """Apply the EDF detector's refractory period to paired CSV markers too.

    control.py logs the last frame of one block and the first of the next.
    Their TTLs are only a frame apart; keep the first, as the EDF detector does.
    """
    kept = []
    for value in np.sort(np.asarray(times, dtype=float)):
        if not kept or value - kept[-1] >= min_isi_s:
            kept.append(value)
    return np.asarray(kept)


def load_stimulus_log(csv_path):
    df = pd.read_csv(csv_path)
    required = {"global_time_s", "block_num", "frequency_hz", "stimulus_state", "trigger_sent"}
    if not required <= set(df):
        raise ValueError(f"The control stimulus log is missing columns: {sorted(required - set(df))}")
    if len(df) < 2:
        raise ValueError("The stimulus log needs at least two frames.")
    df = df.sort_values("global_time_s").reset_index(drop=True)
    times = df.global_time_s.to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Stimulus frame times must be finite and strictly increasing.")
    if not df.stimulus_state.isin(["ON", "OFF"]).all():
        raise ValueError("stimulus_state must contain ON or OFF.")
    if df.block_num.isna().any() or not df.trigger_sent.isin([0, 1]).all():
        raise ValueError("Block labels cannot be missing and trigger_sent must be 0 or 1.")
    df["state_bin"] = (df.stimulus_state == "ON").astype(float)
    rows = []
    previous_stop = -1
    for block_num, group in df.groupby("block_num", sort=False):
        indices = group.index.to_numpy()
        if np.any(np.diff(indices) != 1) or indices[0] <= previous_stop:
            raise ValueError("Each block_num must identify one contiguous frequency block.")
        previous_stop = indices[-1]
        frequencies = group.frequency_hz.to_numpy(dtype=float)
        if not np.isfinite(frequencies).all() or not np.allclose(frequencies, frequencies[0], rtol=0, atol=1e-6):
            raise ValueError(f"Block {block_num} must have one finite frequency.")
        rows.append({"block_num": block_num, "frequency_hz": float(frequencies[0]),
                     "t_start_csv": float(group.global_time_s.iloc[0]),
                     "t_last_frame_csv": float(group.global_time_s.iloc[-1]),
                     "n_frames": len(group)})
    epochs = pd.DataFrame(rows)
    # Hold the final frame until the next frame/blank screen. For the final
    # block, estimate this one-frame duration from the logged frame spacing.
    frame_period = float(np.median(np.diff(times)))
    epochs["t_end_csv"] = np.r_[epochs.t_start_csv.to_numpy()[1:], times[-1] + frame_period]
    logged_triggers = df.loc[df.trigger_sent == 1, "global_time_s"].to_numpy(dtype=float)
    triggers = collapse_close_triggers(logged_triggers)
    if len(triggers) < 3:
        raise ValueError("At least three distinct control boundary markers are needed for clock alignment.")
    print(f"[stimulus] {len(df)} frames, {len(epochs)} frequency blocks; "
          f"{len(logged_triggers)} logged markers -> {len(triggers)} distinct boundaries")
    return df, epochs, triggers


def detect_dc7_triggers(raw):
    return phase.detect_dc7_triggers(raw, dc7_name=DC7_CHANNEL_NAME,
                                     threshold=DC7_THRESHOLD, min_isi_s=MIN_TRIGGER_ISI_S)


def align_clocks(csv_trigger_times, eeg_trigger_times, warn_tol_s=TRIGGER_MATCH_WARN_TOL_S):
    """Fit offset and drift using unique marker pairs, permitting missing pulses."""
    csv_times = collapse_close_triggers(csv_trigger_times)
    eeg_times = np.asarray(eeg_trigger_times, dtype=float)
    if len(csv_times) < 3 or len(eeg_times) < 3:
        raise ValueError("Clock alignment requires at least three markers on each clock.")
    best, best_key = None, None
    for ct0 in csv_times[:5]:
        for et0 in eeg_times[:10]:
            offset = et0 - ct0
            csv_match, eeg_match, used = [], [], set()
            for ct in csv_times:
                distances = abs(eeg_times - (ct + offset))
                index = int(np.argmin(distances))
                if distances[index] < .2 and index not in used:
                    used.add(index)
                    csv_match.append(ct)
                    eeg_match.append(eeg_times[index])
            if len(csv_match) < 3:
                continue
            slope, intercept = np.polyfit(csv_match, eeg_match, 1)
            residuals = np.asarray(eeg_match) - (slope * np.asarray(csv_match) + intercept)
            key = (len(csv_match), -float(np.mean(residuals ** 2)))
            if slope > 0 and (best_key is None or key > best_key):
                best_key = key
                best = (slope, intercept, np.asarray(eeg_match), residuals)
    if best is None:
        raise ValueError("Could not align the control CSV and EDF triggers. Check the input pair and DC7 threshold.")
    slope, intercept, matched, residuals = best
    max_ms = np.max(abs(residuals)) * 1000
    print(f"[align] {len(matched)}/{len(csv_times)} distinct CSV boundaries matched "
          f"to {len(eeg_times)} EDF pulses; maximum residual {max_ms:.2f} ms")
    print(f"[align] t_edf = {slope:.8f} * t_csv + {intercept:.4f}")
    if max_ms > warn_tol_s * 1000:
        print("[align][WARNING] Alignment residual exceeds the configured tolerance.")
    if len(matched) < len(csv_times):
        print("[align] Missing markers: block boundaries remain mapped from the CSV clock fit.")
    return best


def analyze_channel(ch_name, raw, csv_df, epochs, slope, intercept, out_dir):
    """Analyze every block, then write exactly one channel summary image."""
    sfreq = float(raw.info["sfreq"])
    eeg = raw.get_data(picks=[ch_name])[0]
    times = np.arange(len(eeg)) / sfreq
    starts = slope * epochs.t_start_csv.to_numpy() + intercept
    stops = slope * epochs.t_end_csv.to_numpy() + intercept
    flash_stop = float(stops[-1])
    after_start = flash_stop + AFTER_FLASH_DELAY_S
    # An unobserved final hardware pulse does not truncate the last CSV block.
    ref = phase.reference_square_wave(csv_df, (times - intercept) / slope, tol_s=MERGE_ASOF_TOL_S).copy()
    ref[(times < starts[0]) | (times >= flash_stop)] = np.nan
    stimulus_mask = np.isfinite(ref)
    ref_filled = np.nan_to_num(ref, nan=.5)

    cache = {}
    filtered_eeg = None
    psd_mask = np.zeros(len(eeg), dtype=bool)
    block_results = []
    for i, row in enumerate(epochs.itertuples(index=False)):
        frequency = float(row.frequency_hz)
        if frequency not in cache:
            # Full finite runs are filtered before applying block masks. This
            # is the same EEG/reference phase extraction as PhaseLocking.py.
            broad, eeg_phase, ref_phase = phase.filter_and_extract_phase(eeg, ref_filled, sfreq, frequency)
            if filtered_eeg is None:
                filtered_eeg = broad
            cache[frequency] = (eeg_phase, ref_phase)
        eeg_phase, ref_phase = cache[frequency]
        inside = (times >= starts[i]) & (times < stops[i])
        usable = (inside & stimulus_mask & (times - starts[i] > BOUNDARY_EXCLUDE_S)
                  & (stops[i] - times > BOUNDARY_EXCLUDE_S) & np.isfinite(filtered_eeg))
        psd_mask |= usable
        valid = usable & np.isfinite(eeg_phase) & np.isfinite(ref_phase)
        plv = phase.circular_plv(ref_phase, eeg_phase, valid) if valid.sum() > sfreq else np.nan
        block_results.append({"block_num": row.block_num, "frequency_hz": frequency,
                              "plv": plv, "valid_phase_s": valid.sum() / sfreq})

    blocks = pd.DataFrame(block_results)
    available = np.isfinite(blocks.plv).sum()
    mean_plv = float(blocks.plv.mean()) if available else np.nan
    f_on, p_on, n_on = phase.contiguous_welch(filtered_eeg, sfreq, psd_mask, window_s=WELCH_NPERSEG_S)
    after_mask = (times > after_start) & np.isfinite(filtered_eeg)
    f_after, p_after, n_after = phase.contiguous_welch(filtered_eeg, sfreq, after_mask, window_s=WELCH_NPERSEG_S)
    print(f"[{ch_name}] PLV: {available}/{len(blocks)} blocks, mean block PLV={mean_plv:.3f}; "
          f"PSD: {n_on} During Flash / {n_after} After Flash windows")
    print(blocks.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"[{ch_name}] mapped flash offset {flash_stop:.3f}s; After Flash starts {after_start:.3f}s")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    reference = f" (re-ref to {REFERENCE_CHANNEL})" if REFERENCE_CHANNEL else ""
    fmin, fmax = blocks.frequency_hz.min(), blocks.frequency_hz.max()
    fig.suptitle(f"Channel {ch_name}{reference}  |  control flicker {fmin:g}-{fmax:g} Hz  |  "
                 f"mean block PLV={mean_plv:.3f}")

    x = np.arange(1, len(blocks) + 1)
    axes[0].bar(x, blocks.plv, color="#1f77b4")
    axes[0].set(title=f"PLV per frequency block ({available}/{len(blocks)})", ylabel="PLV",
                xlabel="block # / frequency (Hz)", ylim=(0, 1), xlim=(.5, len(blocks) + .5))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{b}\n{f:g}" for b, f in zip(blocks.block_num, blocks.frequency_hz)], fontsize=7)
    if available < len(blocks):
        axes[0].scatter(x[~np.isfinite(blocks.plv)], np.full(len(blocks) - available, .025),
                        marker="x", color="gray", label="Insufficient phase data")
        axes[0].legend(fontsize=7)

    for frequencies, power, label, color in ((f_on, p_on, "During Flash", "#1f77b4"),
                                             (f_after, p_after, "After Flash", "#ff7f0e")):
        if power is not None:
            band = (frequencies >= phase.BANDPASS_LOW_HZ) & (frequencies <= phase.BANDPASS_HIGH_HZ)
            axes[1].semilogy(frequencies[band], power[band], label=label, color=color)
    if p_on is None or p_after is None:
        missing = " / ".join(label for power, label in ((p_on, "During Flash"), (p_after, "After Flash")) if power is None)
        axes[1].text(.03, .96, f"{missing}: no full PSD windows", transform=axes[1].transAxes, va="top", fontsize=8)
    if np.isclose(fmin, fmax):
        axes[1].axvline(fmin, color="r", ls="--", lw=1, label="f0")
        axes[1].axvline(2 * fmin, color="r", ls=":", lw=1, label="2*f0")
    else:
        axes[1].axvspan(fmin, fmax, color="red", alpha=.07, label="Stimulus frequencies")
    axes[1].set(title="EEG PSD (pooled frequency blocks)", xlabel="Hz", ylabel="Power (V²/Hz)",
                xlim=(phase.BANDPASS_LOW_HZ, phase.BANDPASS_HIGH_HZ))
    axes[1].legend(fontsize=8)
    fig.text(.5, .015,
             f"Preprocessing / PSD: {phase.NOTCH_FREQ_HZ:g} Hz notch + {phase.BANDPASS_LOW_HZ:g}-{phase.BANDPASS_HIGH_HZ:g} Hz  |  "
             f"PLV: each block's f0 ± {phase.PHASE_HALF_BANDWIDTH_HZ:g} Hz\n"
             f"Usable phase: {blocks.valid_phase_s.sum():.1f}s  |  {WELCH_NPERSEG_S:g}s PSD windows: "
             f"During Flash {n_on}, After Flash {n_after}  |  After Flash: mapped flash offset + {AFTER_FLASH_DELAY_S:g}s",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .09, 1, .95))
    path = Path(out_dir) / f"{ch_name}_phase_locking.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"channel": ch_name, "blocks": blocks, "mean_block_plv": mean_plv,
            "during_psd_windows": n_on, "after_psd_windows": n_after,
            "flash_stop_s": flash_stop, "after_flash_start_s": after_start, "figure": path}


def main():
    csv_df, epochs, csv_triggers = load_stimulus_log(STIMULUS_CSV_PATH)
    raw = phase.load_edf(EDF_PATH)
    missing = [ch for ch in EEG_CHANNELS if ch not in raw.ch_names]
    if missing or not EEG_CHANNELS or len(set(EEG_CHANNELS)) != len(EEG_CHANNELS):
        raise ValueError(f"Specify distinct, available EEG channels. Missing: {missing}")
    eeg_triggers, _, _ = detect_dc7_triggers(raw)
    slope, intercept, _, _ = align_clocks(csv_triggers, eeg_triggers)
    raw = phase.rereference_to_channel(raw, REREFERENCE_CHANNELS, REFERENCE_CHANNEL)
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    results = [analyze_channel(ch, raw, csv_df, epochs, slope, intercept, output) for ch in EEG_CHANNELS]
    print(f"\n[main] Saved {len(results)} channel images in {output.resolve()}")
    for result in results:
        print(f"  {result['figure'].name}")
    return results


if __name__ == "__main__":
    main()
