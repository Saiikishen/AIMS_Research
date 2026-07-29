#!/usr/bin/env python3
"""
exg_visualizer.py

Real-time dual-channel EEG + ECG visualizer for a BioAmp EXG Pill setup
wired into a Seeed XIAO ESP32C6:
    A0 -> EEG electrode channel
    A1 -> ECG electrode channel
streamed over USB serial. Pairs with the exg_dual_channel firmware sketch.

Pipeline (per channel):
    ESP32C6 (one "eeg_raw,ecg_raw" CSV line per sample pair)
        -> background SerialReader thread -> thread-safe sample queue
        -> StreamingFilter (causal IIR notch + bandpass via sosfilt with
           persistent zi state -- no per-frame edge transients)
        -> rolling display buffer
        -> matplotlib FuncAnimation (time-domain trace + live PSD, per channel)

Firmware assumption (edit _parse_line() if yours differs):
    Serial.print(eeg_raw); Serial.print(','); Serial.println(ecg_raw);
  printed at a FIXED sample rate (micros()-paced). The --fs argument MUST
  match whatever rate the firmware actually samples at, or the filter
  cutoffs and time/frequency axes will all be wrong.

Run with real hardware:
    python exg_visualizer.py --port /dev/ttyACM0 --fs 250
    python exg_visualizer.py --port COM5 --notch 50

Run without hardware attached (synthetic test signal, useful for tuning
filters before you trust it against real electrodes):
    python exg_visualizer.py --simulate

While the window is open:
    press 'n' -> toggle the notch filter on/off (both channels)
    press 'b' -> toggle the bandpass filter on/off (both channels)
  (compare raw vs. filtered live, and watch the PSD panels to confirm
  the mains notch and band edges are where you expect)

Dependencies:
    pip install pyserial numpy scipy matplotlib
    # optional, only needed if you use to_mne_raw() for offline analysis:
    pip install mne
"""

import argparse
import sys
import threading
import time
from collections import deque

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------------------------------------------------------------
# Per-channel presets -- edit if your electrode placement/goals differ.
# --------------------------------------------------------------------------
EEG_PRESET = dict(lowcut=0.5, highcut=90.0, label="EEG (A0)")
ECG_PRESET = dict(lowcut=0.5, highcut=40.0, label="ECG (A1)")

ADC_MAX = 4095   # ESP32C6 ADC is 12-bit by default (0-4095) under Arduino core
ADC_VREF = 3.3   # volts, only used by the optional counts_to_millivolts() helper


def counts_to_millivolts(adc_counts, gain=1.0):
    """
    Rough ADC-counts -> mV conversion. The BioAmp EXG Pill's output isn't
    inherently calibrated to a known physiological voltage without knowing
    its analog gain stage and exactly how it's wired into the ESP32C6 ADC
    input, so treat this as approximate -- fine for relative comparisons,
    not for clinical-grade amplitude readings.
    """
    volts = (np.asarray(adc_counts, dtype=float) / ADC_MAX) * ADC_VREF
    return (volts / gain) * 1000.0


# --------------------------------------------------------------------------
# Serial acquisition (background thread so plotting never blocks on I/O)
# --------------------------------------------------------------------------
class SerialReader(threading.Thread):
    """
    Reads one "eeg_raw,ecg_raw" CSV line per sample pair from the ESP32C6
    and pushes (eeg, ecg) tuples onto a queue. deque.append()/.popleft()
    are individually atomic in CPython (protected by the GIL), so no
    extra lock is needed for this single-producer/single-consumer pattern.
    """

    def __init__(self, port, baud, out_queue, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.ser = None

    @staticmethod
    def _parse_line(line):
        try:
            eeg_str, ecg_str = line.strip().split(",")
            return float(eeg_str), float(ecg_str)
        except (ValueError, AttributeError):
            return None  # ignore boot banners / garbage / partial lines

    def run(self):
        import serial  # local import: --simulate mode doesn't need pyserial installed
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as e:
            print(f"[SerialReader] could not open {self.port}: {e}", file=sys.stderr)
            self.stop_event.set()
            return

        time.sleep(2)  # let the ESP32 finish its reset-on-open before trusting the stream
        self.ser.reset_input_buffer()

        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore")
            except serial.SerialException as e:
                print(f"[SerialReader] read error: {e}", file=sys.stderr)
                break
            sample = self._parse_line(raw)
            if sample is not None:
                self.out_queue.append(sample)

        if self.ser and self.ser.is_open:
            self.ser.close()


class SimulatedReader(threading.Thread):
    """
    Drop-in replacement for SerialReader when no hardware is attached.
    Generates an EEG-like waveform on one channel and an ECG-like waveform
    on the other, both with mains interference and noise, at the
    configured sample rate -- so the filter/plot pipeline can be tuned and
    sanity-checked before it's trusted against real electrodes.
    """

    def __init__(self, fs, mains_hz, out_queue, stop_event):
        super().__init__(daemon=True)
        self.fs = fs
        self.mains_hz = mains_hz
        self.out_queue = out_queue
        self.stop_event = stop_event

    def run(self):
        t = 0.0
        dt = 1.0 / self.fs
        while not self.stop_event.is_set():
            eeg_val = (
                30 * np.sin(2 * np.pi * 10 * t)               # ~alpha rhythm
                + 10 * np.sin(2 * np.pi * 20 * t)             # ~beta rhythm
                + 15 * np.sin(2 * np.pi * self.mains_hz * t)  # mains hum
                + np.random.normal(0, 8)
            )
            phase = (t % (1.0 / 1.2)) * 1.2                   # ~72 bpm QRS-like spike train
            spike = 400 * np.exp(-((phase - 0.05) ** 2) / (2 * 0.002 ** 2))
            ecg_val = spike + 15 * np.sin(2 * np.pi * self.mains_hz * t) + np.random.normal(0, 5)

            self.out_queue.append((2048 + eeg_val, 2048 + ecg_val))  # centered on ADC mid-scale
            t += dt
            time.sleep(dt)


# --------------------------------------------------------------------------
# Streaming (causal) notch + bandpass filter with persistent state
# --------------------------------------------------------------------------
class StreamingFilter:
    """
    Butterworth bandpass + IIR notch, both in second-order-sections form,
    applied with scipy.signal.sosfilt and a persistent `zi` state carried
    across calls.

    Deliberately NOT filtfilt-on-the-whole-buffer: recomputing a zero-
    phase filter over the rolling window every redraw reintroduces a
    transient at the window edge on *every frame* -- the same class of
    cascaded-filter edge-transient problem from the SEEG/MATLAB work.
    Carrying zi forward instead makes this a true continuous causal
    stream, at the cost of a small constant group delay -- the right
    trade-off for a live display.
    """

    def __init__(self, fs, lowcut, highcut, notch_hz, notch_q=30.0, order=4):
        self.sos_band = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
        b_notch, a_notch = signal.iirnotch(notch_hz, notch_q, fs=fs)
        self.sos_notch = signal.tf2sos(b_notch, a_notch)

        self._zi_band_proto = signal.sosfilt_zi(self.sos_band)
        self._zi_notch_proto = signal.sosfilt_zi(self.sos_notch)
        self.zi_band = None   # lazily scaled to the first real sample, see process()
        self.zi_notch = None

        self.bandpass_enabled = True
        self.notch_enabled = True

    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=float)
        if chunk.size == 0:
            return chunk

        # Each stage's zi is lazily initialised to the DC level of *its*
        # actual input, not the original raw input.  This matters when
        # the bandpass runs first: it strips DC (~2048 -> ~0), so the
        # notch's zi must be scaled to ~0 (the bandpass output) rather
        # than ~2048 (the raw ADC level).  The old code initialised both
        # zi's from the raw chunk[0] before any filtering, which caused
        # a ~2048-count transient at the notch stage on startup.
        if self.bandpass_enabled:
            if self.zi_band is None:
                self.zi_band = self._zi_band_proto * chunk[0]
            chunk, self.zi_band = signal.sosfilt(self.sos_band, chunk, zi=self.zi_band)

        if self.notch_enabled:
            if self.zi_notch is None:
                self.zi_notch = self._zi_notch_proto * chunk[0]
            chunk, self.zi_notch = signal.sosfilt(self.sos_notch, chunk, zi=self.zi_notch)

        return chunk



def to_mne_raw(buffers, fs, ch_types=None):
    """
    Wrap channel buffers as an MNE RawArray so you can drop into MNE's
    tools (PSD, ICA, epoching, browsing) once you've captured a segment
    you care about. Only imports mne when this is actually called, so the
    live visualizer above doesn't need mne installed to run.

    buffers: dict like {"EEG": eeg_array, "ECG": ecg_array}
    """
    import mne
    ch_names = list(buffers.keys())
    if ch_types is None:
        ch_types = ["eeg" if "eeg" in n.lower() else "ecg" for n in ch_names]
    data = np.array([np.asarray(buffers[n], dtype=float) for n in ch_names])
    info = mne.create_info(ch_names, sfreq=fs, ch_types=ch_types)
    return mne.io.RawArray(data, info)


# --------------------------------------------------------------------------
# One time-domain + PSD column per channel
# --------------------------------------------------------------------------
class ChannelView:
    def __init__(self, ax_time, ax_psd, fs, window_sec, label, filt):
        self.fs = fs
        self.filt = filt
        self.n = int(window_sec * fs)

        self.raw_buf = deque([float("nan")] * self.n, maxlen=self.n)
        self.filt_buf = deque([float("nan")] * self.n, maxlen=self.n)
        self.t = np.arange(-self.n, 0) / fs

        (self.line_raw,) = ax_time.plot(self.t, list(self.raw_buf), lw=0.6, alpha=0.35, color="gray", label="raw")
        (self.line_filt,) = ax_time.plot(self.t, list(self.filt_buf), lw=1.2, color="C0", label="filtered")
        ax_time.set_xlim(self.t[0], self.t[-1])
        ax_time.set_xlabel("time (s)")
        ax_time.set_title(label)
        ax_time.legend(loc="upper right", fontsize=8)
        ax_time.grid(alpha=0.3)
        self.ax_time = ax_time

        (self.line_psd,) = ax_psd.plot([], [])
        ax_psd.set_xlabel("frequency (Hz)")
        ax_psd.set_ylabel("PSD (dB)")
        ax_psd.set_xlim(0, min(60, fs / 2))
        ax_psd.grid(alpha=0.3)
        self.ax_psd = ax_psd

    def push(self, new_raw_samples):
        if not new_raw_samples:
            return
        filtered = self.filt.process(new_raw_samples)
        self.raw_buf.extend(new_raw_samples)
        self.filt_buf.extend(filtered)

    def redraw(self):
        raw_data = np.array(self.raw_buf)
        filt_data = np.array(self.filt_buf)

        # DC-subtract the raw trace so it shares the same baseline as
        # the filtered signal (whose DC has already been removed by the
        # bandpass).  Without this the Y-axis auto-range spans the full
        # ~2048-count DC offset and the actual physiological oscillations
        # (tens of counts peak-to-peak) are invisible.
        raw_mean = np.nanmean(raw_data)
        if np.isfinite(raw_mean):
            self.line_raw.set_ydata(raw_data - raw_mean)
        else:
            self.line_raw.set_ydata(raw_data)
        self.line_filt.set_ydata(filt_data)
        self.ax_time.relim()
        self.ax_time.autoscale_view(scalex=False, scaley=True)

        valid = filt_data[np.isfinite(filt_data)]
        if len(valid) > 64:
            f, pxx = signal.welch(valid, fs=self.fs, nperseg=min(512, len(valid)))
            pxx_db = 10 * np.log10(pxx + 1e-12)
            self.line_psd.set_data(f, pxx_db)
            self.ax_psd.relim()
            self.ax_psd.autoscale_view()


class LivePlotter:
    """EEG-only view: time domain on top, PSD on the bottom."""

    def __init__(self, fs, window_sec, eeg_filt, ecg_filt):
        self.fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        self.fig.suptitle("EEG live view  --  'n' toggle notch, 'b' toggle bandpass")

        self.eeg = ChannelView(axes[0], axes[1], fs, window_sec, EEG_PRESET["label"], eeg_filt)
        # self.ecg = ChannelView(..., ecg_filt)  # ECG plot disabled

        self.fig.tight_layout()

    def push(self, eeg_samples, ecg_samples):
        self.eeg.push(eeg_samples)
        # self.ecg.push(ecg_samples)

    def redraw(self, _frame):
        self.eeg.redraw()
        # self.ecg.redraw()
        return (self.eeg.line_raw, self.eeg.line_filt, self.eeg.line_psd)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Live dual-channel EEG+ECG visualizer for BioAmp EXG Pill + ESP32C6")
    ap.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyACM0 (Linux), COM5 (Windows), /dev/cu.usbmodemXXXX (Mac)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fs", type=float, default=250.0, help="sample rate the firmware actually outputs at (Hz) -- must match reality")
    ap.add_argument("--notch", type=float, default=50.0, help="mains frequency to notch out on both channels (50 India/EU, 60 US)")
    ap.add_argument("--window", type=float, default=6.0, help="seconds visible on screen")
    ap.add_argument("--eeg-lowcut", type=float, default=EEG_PRESET["lowcut"])
    ap.add_argument("--eeg-highcut", type=float, default=EEG_PRESET["highcut"])
    ap.add_argument("--ecg-lowcut", type=float, default=ECG_PRESET["lowcut"])
    ap.add_argument("--ecg-highcut", type=float, default=ECG_PRESET["highcut"])
    ap.add_argument("--simulate", action="store_true", help="use synthetic data, no hardware needed")
    args = ap.parse_args()

    if not args.simulate and not args.port:
        ap.error("--port is required unless --simulate is set")

    sample_queue = deque()
    stop_event = threading.Event()

    if args.simulate:
        reader = SimulatedReader(args.fs, args.notch, sample_queue, stop_event)
    else:
        reader = SerialReader(args.port, args.baud, sample_queue, stop_event)
    reader.start()

    eeg_filt = StreamingFilter(args.fs, args.eeg_lowcut, args.eeg_highcut, args.notch)
    ecg_filt = StreamingFilter(args.fs, args.ecg_lowcut, args.ecg_highcut, args.notch)
    plotter = LivePlotter(args.fs, args.window, eeg_filt, ecg_filt)

    def on_key(event):
        if event.key == "n":
            new_state = not eeg_filt.notch_enabled
            eeg_filt.notch_enabled = ecg_filt.notch_enabled = new_state
            print(f"[toggle] notch {'ON' if new_state else 'OFF'}")
        elif event.key == "b":
            new_state = not eeg_filt.bandpass_enabled
            eeg_filt.bandpass_enabled = ecg_filt.bandpass_enabled = new_state
            print(f"[toggle] bandpass {'ON' if new_state else 'OFF'}")

    plotter.fig.canvas.mpl_connect("key_press_event", on_key)

    def animate(frame):
        eeg_pending, ecg_pending = [], []
        while sample_queue:
            eeg_val, ecg_val = sample_queue.popleft()
            eeg_pending.append(eeg_val)
            ecg_pending.append(ecg_val)
        plotter.push(eeg_pending, ecg_pending)
        return plotter.redraw(frame)

    ani = FuncAnimation(plotter.fig, animate, interval=30, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        stop_event.set()
        reader.join(timeout=2)


if __name__ == "__main__":
    main()
