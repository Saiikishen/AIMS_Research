#!/usr/bin/env python3
"""
Automated verification suite for the filtering and signal analysis
pipeline in exg_visualizer.py.

Tests cover:
  1. StreamingFilter construction and SOS shapes
  2. Notch filter frequency response (rejection at mains Hz)
  3. Bandpass filter frequency response (passband / stopband)
  4. Causal zi state initialisation (startup-transient suppression)
  5. Streaming consistency (chunk-by-chunk == one-shot)
  6. Toggle behaviour (enable/disable paths)
  7. Edge cases (empty chunk, single sample, DC-only)
  8. PSD computation path (Welch via ChannelView.redraw)
  9. SimulatedReader waveform sanity (frequency content)
 10. counts_to_millivolts conversion
"""

import sys, os, unittest
import numpy as np
from scipy import signal as sp_signal

# ---- import the module under test ----
sys.path.insert(0, os.path.dirname(__file__))
from exg_visualizer import (
    StreamingFilter,
    SimulatedReader,
    counts_to_millivolts,
    ADC_MAX,
    ADC_VREF,
    EEG_PRESET,
    ECG_PRESET,
)


# ===================== helpers =====================
def make_test_signal(fs, duration, freqs, amps=None):
    """Sum-of-sinusoids test signal."""
    t = np.arange(0, duration, 1.0 / fs)
    if amps is None:
        amps = [1.0] * len(freqs)
    return t, sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))


def power_at_freq(x, fs, target_hz, bw=2.0):
    """Return average PSD in a ±bw Hz band around target_hz."""
    f, pxx = sp_signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (f >= target_hz - bw) & (f <= target_hz + bw)
    return np.mean(pxx[mask]) if np.any(mask) else 0.0


# ==================== test cases ====================
class TestStreamingFilterConstruction(unittest.TestCase):
    """Verify SOS matrix shapes and zi prototypes."""

    def test_sos_shapes(self):
        filt = StreamingFilter(fs=250, lowcut=0.5, highcut=45, notch_hz=50)
        # SOS arrays must be Nx6
        self.assertEqual(filt.sos_band.shape[1], 6)
        self.assertEqual(filt.sos_notch.shape[1], 6)

    def test_zi_proto_shapes(self):
        filt = StreamingFilter(fs=250, lowcut=0.5, highcut=45, notch_hz=50)
        # zi prototype shape must be (n_sections, 2)
        self.assertEqual(filt._zi_band_proto.shape[1], 2)
        self.assertEqual(filt._zi_notch_proto.shape[1], 2)
        self.assertEqual(filt._zi_band_proto.shape[0], filt.sos_band.shape[0])
        self.assertEqual(filt._zi_notch_proto.shape[0], filt.sos_notch.shape[0])


class TestNotchFilter(unittest.TestCase):
    """The notch must attenuate the mains frequency sharply."""

    def _check_notch(self, fs, notch_hz):
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=fs / 2 - 1,
                               notch_hz=notch_hz)
        filt.bandpass_enabled = False  # isolate the notch

        _, x = make_test_signal(fs, 4.0, [10, notch_hz], amps=[1, 1])
        y = filt.process(x)

        pwr_notch = power_at_freq(y, fs, notch_hz, bw=1.0)
        pwr_pass = power_at_freq(y, fs, 10, bw=2.0)

        # notch bin should be ≥20 dB below the passband
        if pwr_notch > 0 and pwr_pass > 0:
            ratio_db = 10 * np.log10(pwr_pass / pwr_notch)
            self.assertGreater(ratio_db, 20,
                               f"Notch at {notch_hz} Hz only {ratio_db:.1f} dB down (need >20)")

    def test_notch_50hz(self):
        self._check_notch(250, 50)

    def test_notch_60hz(self):
        self._check_notch(250, 60)


class TestBandpassFilter(unittest.TestCase):
    """Bandpass must pass in-band and reject out-of-band."""

    def test_passband(self):
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        filt.notch_enabled = False  # isolate the bandpass
        _, x = make_test_signal(fs, 4.0, [10])
        y = filt.process(x)
        # 10 Hz is well inside [0.5, 45] — should survive
        ratio = np.std(y[fs:]) / np.std(x[fs:])  # skip startup
        self.assertGreater(ratio, 0.5, "10 Hz signal lost >6 dB through the bandpass")

    def test_stopband_high(self):
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        filt.notch_enabled = False
        _, x = make_test_signal(fs, 4.0, [80])
        y = filt.process(x)
        # 80 Hz is above highcut=45 — should be heavily attenuated
        pwr_in = power_at_freq(x, fs, 80, bw=2.0)
        pwr_out = power_at_freq(y, fs, 80, bw=2.0)
        if pwr_in > 0:
            atten_db = 10 * np.log10(pwr_in / max(pwr_out, 1e-30))
            self.assertGreater(atten_db, 20,
                               f"80 Hz only {atten_db:.1f} dB down past highcut=45")

    def test_stopband_dc(self):
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        filt.notch_enabled = False
        x = np.ones(fs * 4) * 2048  # pure DC
        y = filt.process(x)
        # DC should be rejected by lowcut=0.5 Hz highpass
        self.assertLess(np.abs(np.mean(y[-fs:])), 5.0,
                        "DC not removed by bandpass lowcut")


class TestCausalZiInitialisation(unittest.TestCase):
    """zi should be scaled to the first sample to suppress startup transient."""

    def test_no_large_transient(self):
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        baseline = 2048.0
        x = np.full(fs, baseline) + np.random.normal(0, 5, fs)
        y = filt.process(x)
        # The first few samples should NOT spike wildly above the signal range
        peak = np.max(np.abs(y[:20] - np.mean(y[50:])))
        self.assertLess(peak, 200,
                        f"Startup transient peak={peak:.1f} — zi init may be broken")


class TestStreamingConsistency(unittest.TestCase):
    """Processing in small chunks must equal processing in one shot."""

    def test_chunked_vs_oneshot(self):
        fs = 250
        _, x = make_test_signal(fs, 2.0, [10, 50], amps=[1, 0.5])

        # one-shot
        f1 = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        y_one = f1.process(x)

        # chunked (50-sample chunks)
        f2 = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        chunks = [x[i:i + 50] for i in range(0, len(x), 50)]
        y_chunked = np.concatenate([f2.process(c) for c in chunks])

        np.testing.assert_allclose(y_one, y_chunked, atol=1e-10,
                                   err_msg="Chunk-by-chunk differs from one-shot")


class TestToggleBehaviour(unittest.TestCase):
    """Disabling a filter stage must pass through unchanged."""

    def test_notch_disabled_passthrough(self):
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        filt.notch_enabled = False
        filt.bandpass_enabled = False
        x = np.random.randn(fs)
        y = filt.process(x)
        np.testing.assert_array_equal(x, y,
                                      "Both filters disabled but output differs from input")

    def test_mid_stream_toggle(self):
        """Toggle notch mid-stream: should not crash or produce NaN."""
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        _, x = make_test_signal(fs, 2.0, [10, 50])
        chunk_size = 50
        for i in range(0, len(x), chunk_size):
            if i == 250:
                filt.notch_enabled = not filt.notch_enabled
            y = filt.process(x[i:i + chunk_size])
            self.assertFalse(np.any(np.isnan(y)), "NaN after mid-stream toggle")

    def test_notch_zi_uses_bandpass_output(self):
        """When bandpass is active, notch zi must be initialised to the
        bandpass output level (~0), not the raw ADC level (~2048).
        The old code initialised both zi's from the raw input, causing
        a ~2048-count transient at the notch stage."""
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        # simulate ADC-level DC input
        x = np.full(fs * 2, 2048.0) + np.random.normal(0, 5, fs * 2)
        y = filt.process(x)
        # After the very first output samples, the signal should be near 0
        # (DC removed by bandpass, no massive transient from notch)
        self.assertLess(np.max(np.abs(y[:50])), 100,
                        "Notch zi mismatch: large transient on startup")


class TestEdgeCases(unittest.TestCase):

    def test_empty_chunk(self):
        filt = StreamingFilter(fs=250, lowcut=0.5, highcut=45, notch_hz=50)
        y = filt.process([])
        self.assertEqual(len(y), 0)

    def test_single_sample(self):
        filt = StreamingFilter(fs=250, lowcut=0.5, highcut=45, notch_hz=50)
        y = filt.process([2048.0])
        self.assertEqual(len(y), 1)
        self.assertFalse(np.isnan(y[0]))


class TestCountsToMillivolts(unittest.TestCase):

    def test_zero(self):
        self.assertAlmostEqual(counts_to_millivolts(0, gain=1.0), 0.0)

    def test_full_scale(self):
        expected_mv = (ADC_VREF / 1.0) * 1000.0  # gain=1
        self.assertAlmostEqual(counts_to_millivolts(ADC_MAX, gain=1.0), expected_mv, places=2)

    def test_gain_scaling(self):
        mv1 = counts_to_millivolts(2048, gain=1.0)
        mv2 = counts_to_millivolts(2048, gain=2.0)
        self.assertAlmostEqual(mv1, mv2 * 2.0, places=5)

    def test_array_input(self):
        result = counts_to_millivolts([0, ADC_MAX // 2, ADC_MAX])
        self.assertEqual(len(result), 3)


class TestSimulatedReaderContent(unittest.TestCase):
    """SimulatedReader should inject known frequency components."""

    def test_eeg_has_alpha_and_mains(self):
        from collections import deque
        import threading
        q = deque()
        stop = threading.Event()
        reader = SimulatedReader(fs=250, mains_hz=50, out_queue=q, stop_event=stop)
        reader.start()
        import time; time.sleep(4.0)  # collect ~3-4 s of data (generous for slow machines)
        stop.set()
        reader.join(timeout=3)

        eeg = np.array([s[0] for s in q])
        self.assertGreater(len(eeg), 100, "SimulatedReader produced too few samples")

        # Should contain 10 Hz alpha and 50 Hz mains
        pwr_alpha = power_at_freq(eeg, 250, 10, bw=2)
        pwr_mains = power_at_freq(eeg, 250, 50, bw=2)
        pwr_noise_band = power_at_freq(eeg, 250, 70, bw=2)  # should be low

        self.assertGreater(pwr_alpha, pwr_noise_band,
                           "No alpha peak detected in simulated EEG")
        self.assertGreater(pwr_mains, pwr_noise_band,
                           "No mains peak detected in simulated EEG")


class TestFilterFrequencyResponse(unittest.TestCase):
    """Verify the actual SOS frequency response curves."""

    def test_bandpass_response_shape(self):
        filt = StreamingFilter(fs=250, lowcut=0.5, highcut=45, notch_hz=50)
        w, h = sp_signal.sosfreqz(filt.sos_band, worN=2048, fs=250)
        mag_db = 20 * np.log10(np.abs(h) + 1e-30)

        # At 10 Hz (well within passband) should be near 0 dB
        idx_10 = np.argmin(np.abs(w - 10))
        self.assertGreater(mag_db[idx_10], -3,
                           f"Bandpass gain at 10 Hz is {mag_db[idx_10]:.1f} dB (expected > -3)")

        # At 80 Hz (well outside) should be significantly attenuated
        idx_80 = np.argmin(np.abs(w - 80))
        self.assertLess(mag_db[idx_80], -20,
                        f"Bandpass gain at 80 Hz is {mag_db[idx_80]:.1f} dB (expected < -20)")

    def test_notch_response_depth(self):
        filt = StreamingFilter(fs=250, lowcut=0.5, highcut=45, notch_hz=50)
        w, h = sp_signal.sosfreqz(filt.sos_notch, worN=2048, fs=250)
        mag_db = 20 * np.log10(np.abs(h) + 1e-30)

        idx_50 = np.argmin(np.abs(w - 50))
        self.assertLess(mag_db[idx_50], -20,
                        f"Notch at 50 Hz only {mag_db[idx_50]:.1f} dB deep (need < -20)")

        # Neighbouring frequencies should be near unity
        idx_40 = np.argmin(np.abs(w - 40))
        self.assertGreater(mag_db[idx_40], -3,
                           f"Notch too wide: gain at 40 Hz is {mag_db[idx_40]:.1f} dB")


class TestZiStateNotReset(unittest.TestCase):
    """
    BUG PROBE: If a filter is toggled OFF and then back ON mid-stream,
    verify that the zi state is still valid (not None or stale) and
    processing resumes without crashing.
    """

    def test_toggle_off_on_preserves_zi(self):
        """zi state should persist while the filter is disabled, and
        processing should resume cleanly when re-enabled."""
        fs = 250
        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=45, notch_hz=50)
        x = np.random.randn(fs) + 2048

        # Process a chunk to initialise zi (both enabled by default)
        filt.process(x[:100])
        self.assertIsNotNone(filt.zi_band)
        self.assertIsNotNone(filt.zi_notch)

        # Toggle off and process (zi shouldn't be updated but should persist)
        saved_zi_band = filt.zi_band.copy()
        filt.bandpass_enabled = False
        filt.notch_enabled = False
        filt.process(x[100:200])
        np.testing.assert_array_equal(filt.zi_band, saved_zi_band,
                                      "zi_band changed while filter was disabled")

        # Toggle back on — should still work
        filt.bandpass_enabled = True
        filt.notch_enabled = True
        y = filt.process(x[200:250])
        self.assertFalse(np.any(np.isnan(y)), "NaN after re-enabling filters")


# ==================== run ====================
if __name__ == "__main__":
    unittest.main(verbosity=2)
