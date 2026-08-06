#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FREQUENCY SWEEP AROUND A TARGET Hz
-----------------------------------
1. On launch, shows a dialog box asking for the desired/target frequency (Hz),
   plus Subject/Session/Day/Run info for logging (same fields as the other task).
2. Builds an 11-step sweep from (target - 0.5 Hz) to (target + 0.5 Hz) in
   0.1 Hz increments:  -0.5, -0.4, -0.3, -0.2, -0.1, 0, +0.1, +0.2, +0.3, +0.4, +0.5
3. Each step: flashes a full-screen white rectangle at that frequency for
   FLASH_DUR (10s), then a plain black screen for BLACK_DUR (5s), then moves to
   the next step. The final step's black screen ends the session.
4. Sends a hardware TTL trigger (0x01) via COM3 on EVERY state change -- i.e.
   right as each flash block begins AND right as each black-screen period
   begins (22 triggers total for the 11-step sweep) -- using the identical
   send_ttl() function/trigger byte as white_flash.py.
5. Logs exact calendar timestamps and event durations to a CSV, same style
   as white_flash.py.
6. Same frame-count-based, deadline-paced flashing loop as white_flash.py
   for accurate, vsync-independent timing. No jitter in this script -- every
   step is a plain constant-frequency square wave (see white_flash.py's
   jittered version if you want jitter added into this sweep too).

ASSUMPTIONS (flag if you want these changed):
- Stimulus is always a full-screen white rectangle (same fallback as
  white_flash.py) -- no separate image-file option here.
- No "press space to begin" screen -- flashing starts immediately once the
  frequency dialog is confirmed, per your description.
- The black-screen periods are plain black (no fixation cross).
"""

import os, csv, time, re
from datetime import datetime
# pyrefly: ignore [missing-import]
from psychopy import visual, core, event, gui

try:
    import serial, serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── HARDWARE & CONFIG ─────────────────────────────────────────────────────────
SERIAL_PORT   = 'COM3'
BAUD_RATE     = 115200
FULLSCREEN    = True
FALLBACK_SCREEN_SIZE = [1920, 1200]
OUTPUT_DIR    = 'data'

# ── SWEEP CONFIG ──────────────────────────────────────────────────────────────
FLASH_DUR     = 10.0   # seconds of flashing per step
BLACK_DUR     = 5.0    # seconds of black screen after each flashing step
FREQ_STEP_HZ  = 0.1    # increment between consecutive steps
FREQ_RANGE_HZ = 0.5    # sweep spans target +/- this amount

NOMINAL_REFRESH_HZ  = 120.0
MIN_PLAUSIBLE_HZ    = 30.0
MAX_PLAUSIBLE_HZ    = 300.0

# ── DISPLAY RESOLUTION ────────────────────────────────────────────────────────
def get_native_resolution(fallback=FALLBACK_SCREEN_SIZE):
    """
    Query the true native pixel resolution on Windows. Must declare DPI
    awareness first -- otherwise GetSystemMetrics silently returns a
    DPI-scaled logical resolution (e.g. 1280x800 instead of 1920x1200 at
    150% scaling), which reproduces the same letterboxing/artifact problem
    this fix is meant to solve.
    """
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()        # older Windows fallback
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        if width > 0 and height > 0:
            return [int(width), int(height)]
    except Exception as e:
        print(f'[DISPLAY WARNING] Could not query native resolution ({e}); '
              f'using fallback {fallback}.')
    return list(fallback)

# ── RUNTIME PROMPT ────────────────────────────────────────────────────────────
def prompt_info():
    """
    Ask for Subject/Session/Day/Run (for logging, same as white_flash.py) plus
    the target frequency to sweep around. Retries up to 3 times if the
    frequency field isn't a valid positive number that keeps the whole sweep
    (target - FREQ_RANGE_HZ .. target + FREQ_RANGE_HZ) above 0 Hz.
    """
    for attempt in range(3):
        dlg = gui.Dlg(title='Frequency Sweep Task')
        # dlg.addField('Subject ID:', 'S01')
        # dlg.addField('Run (1 or 2):', '1')
        # dlg.addField('Session:', '1')
        # dlg.addField('Day:', '1')
        dlg.addField('Target Frequency (Hz):', '8.0')
        data = dlg.show()
        if not dlg.OK:
            core.quit()

        # Hardcode default values for commented out fields
        subj, run, ses, day = 'S01', '1', '1', '1'
        freq_str = str(data[0]).strip()
        try:
            target_freq = float(freq_str)
            if target_freq - FREQ_RANGE_HZ <= 0:
                raise ValueError('target frequency minus the sweep range must stay above 0 Hz')
            return subj, run, ses, day, target_freq
        except ValueError as e:
            print(f'[INPUT WARNING] Invalid frequency "{freq_str}" ({e}); please re-enter.')
    print('[INPUT ERROR] No valid frequency entered after 3 attempts. Exiting.')
    core.quit()

# ── SERIAL TRIGGER ────────────────────────────────────────────────────────────
_ser = None

def init_serial():
    global _ser
    if not SERIAL_AVAILABLE:
        print('[TTL] NO-TRIGGER mode (serial library not installed)')
        return
    try:
        _ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(0.1)
        print(f'[TTL] {SERIAL_PORT} opened successfully')
    except serial.serialutil.SerialException as e:
        print(f'[TTL WARNING] Could not open {SERIAL_PORT}: {e}')
        _ser = None

def send_ttl():
    global _ser
    if _ser:
        try:
            _ser.write(b'\x01')
            print('[TTL] Trigger successfully sent')
        except Exception as e:
            print(f'[TTL ERROR] Failed to send trigger: {e}')
    else:
        print('[TTL] (Simulation) Trigger successfully sent')

def close_serial():
    if _ser:
        try:
            _ser.close()
        except Exception:
            pass

# ── LOGGING ───────────────────────────────────────────────────────────────────
_writer = _fh = _clk = None
_FIELDS = ['timestamp', 'subj', 'ses', 'day', 'task', 'run',
           'step_num', 'frequency_hz', 'event', 'onset_s', 'duration_s']

def init_log(subj, ses, day, run):
    global _writer, _fh, _clk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = os.path.join(OUTPUT_DIR,
         f'task_freq_sweep_{subj}_ses{ses}_day{day}_run{run}_{ts}.csv')
    _fh = open(fn, 'w', newline='', encoding='utf-8')
    _writer = csv.DictWriter(_fh, fieldnames=_FIELDS)
    _writer.writeheader()
    _fh.flush()
    _clk = core.Clock()
    print(f'[LOG] Created log file: {fn}')

def log(subj, ses, day, run, step_num='', frequency_hz='',
        event_label='', onset='', duration=''):
    if not _writer:
        return
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    _writer.writerow({
        'timestamp': now_str,
        'subj': subj, 'ses': ses, 'day': day,
        'task': 'freq_sweep', 'run': run,
        'step_num': step_num, 'frequency_hz': frequency_hz,
        'event': event_label,
        'onset_s': round(float(onset), 4) if onset != '' else '',
        'duration_s': round(float(duration), 4) if duration != '' else ''
    })
    _fh.flush()

def close_log():
    if _fh:
        try:
            _fh.close()
        except Exception:
            pass

# ── HELPERS ───────────────────────────────────────────────────────────────────
def abort(win, subj, ses, day, run):
    send_ttl()
    log(subj, ses, day, run, event_label='abort', onset=_clk.getTime())
    close_log()
    close_serial()
    win.close()
    core.quit()

def check_esc(win, subj, ses, day, run):
    if event.getKeys(['escape']):
        abort(win, subj, ses, day, run)

def measure_refresh_rate(win, fallback_hz=NOMINAL_REFRESH_HZ):
    """
    Measure the monitor's actual refresh rate once at startup. Frame-locked
    flashing needs the REAL rate -- but if the GPU driver/compositor isn't
    honoring vsync, win.flip() returns almost instantly and this measurement
    comes back as a nonsense value (e.g. 600+ Hz on a 120 Hz panel). Reject
    anything outside a plausible range for a real display and fall back to
    the confirmed panel spec instead of trusting a broken measurement.
    """
    print('[DISPLAY] Measuring actual monitor refresh rate (please wait)...')
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                       nWarmUpFrames=15, threshold=1)
    if measured is None or not (MIN_PLAUSIBLE_HZ <= measured <= MAX_PLAUSIBLE_HZ):
        print(f'[DISPLAY WARNING] Measured value ({measured}) is not a real panel '
              f'refresh rate -- your GPU driver/compositor is not honoring vsync for '
              f'this process, so win.flip() is returning without waiting for the real '
              f'screen refresh. This is a Windows/driver-level issue this script cannot '
              f'fix by itself. Falling back to the confirmed panel spec: {fallback_hz} Hz. '
              f'See the console notes printed after the run for how to fix vsync.')
        return float(fallback_hz)
    print(f'[DISPLAY] Measured refresh rate: {measured:.3f} Hz')
    return float(measured)

def wait_flash(win, stims_to_draw, freq, dur, refresh_hz, subj, ses, day, run):
    """
    Frame-locked square-wave (ON/OFF) flashing at `freq` Hz for `dur` seconds.
    Decides ON/OFF by FRAME COUNT rather than elapsed time, and paces each
    flip() to the real target time as a safety net against a driver/compositor
    that isn't honoring vsync -- same approach as white_flash.py.
    """
    frames_per_cycle = refresh_hz / float(freq)
    half_cycle_frames = frames_per_cycle / 2.0
    total_frames = int(round(dur * refresh_hz))
    frame_period = 1.0 / refresh_hz
    t_start = _clk.getTime()

    for frame_n in range(total_frames):
        check_esc(win, subj, ses, day, run)
        phase = frame_n % frames_per_cycle
        if phase < half_cycle_frames:
            for s in stims_to_draw:
                s.draw()
        target_t = t_start + (frame_n + 1) * frame_period
        while _clk.getTime() < target_t:
            pass
        win.flip()

def wait_black(win, dur, refresh_hz, subj, ses, day, run):
    """
    Plain black screen (nothing drawn) for `dur` seconds, deadline-paced the
    same way as wait_flash so its real duration doesn't drift if vsync isn't
    being honored.
    """
    total_frames = int(round(dur * refresh_hz))
    frame_period = 1.0 / refresh_hz
    t_start = _clk.getTime()

    for frame_n in range(total_frames):
        check_esc(win, subj, ses, day, run)
        target_t = t_start + (frame_n + 1) * frame_period
        while _clk.getTime() < target_t:
            pass
        win.flip()

# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_sweep():
    subj, run, ses, day, target_freq = prompt_info()
    init_serial()
    init_log(subj, ses, day, run)

    # Build the 11-step sweep: target - 0.5 .. target + 0.5 in 0.1 Hz steps
    n_steps = int(round((2 * FREQ_RANGE_HZ) / FREQ_STEP_HZ)) + 1  # 11
    freq_sequence = [round(target_freq - FREQ_RANGE_HZ + i * FREQ_STEP_HZ, 4)
                      for i in range(n_steps)]
    print(f'[SWEEP] Target {target_freq} Hz -> {n_steps} steps: '
          f'{[round(f,2) for f in freq_sequence]}')

    # Setup Window — use the true detected native resolution instead of a
    # hardcoded 16:9 size, so nothing gets letterboxed on the 16:10 panel.
    screen_size = get_native_resolution()
    print(f'[DISPLAY] Detected native resolution: {screen_size[0]}x{screen_size[1]}')
    win = visual.Window(screen_size, fullscr=FULLSCREEN,
                        color='black', units='height', allowGUI=False,
                        waitBlanking=True, useFBO=True, winType='pyglet')
    win.mouseVisible = False
    print(f'[DISPLAY] Window buffer resolution: {win.size[0]}x{win.size[1]}')

    # Measure the real refresh rate once, before anything is shown to the subject.
    refresh_hz = measure_refresh_rate(win)

    # Full-screen white rectangle stimulus (same fallback as white_flash.py)
    rect_stim = visual.Rect(win, units='norm', width=2.0, height=2.0, pos=(0, 0),
                             fillColor='white', lineColor='white')
    stims_to_flash = [rect_stim]

    # ── Sweep Loop ──────────────────────────────────────────────────────────
    for idx, freq in enumerate(freq_sequence, start=1):
        print(f"[STEP {idx}/{n_steps}] Flashing at {freq} Hz for {FLASH_DUR}s, "
              f"then black for {BLACK_DUR}s...")

        # ── Trigger + Flash state (state change: black/start -> flashing) ──
        send_ttl()
        t_flash = _clk.getTime()
        log(subj, ses, day, run, step_num=idx, frequency_hz=freq,
            event_label='flash_onset', onset=t_flash, duration=FLASH_DUR)
        wait_flash(win, stims_to_flash, freq, FLASH_DUR, refresh_hz, subj, ses, day, run)
        log(subj, ses, day, run, step_num=idx, frequency_hz=freq,
            event_label='flash_offset', onset=_clk.getTime())

        # ── Trigger + Black state (state change: flashing -> black) ────────
        send_ttl()
        t_black = _clk.getTime()
        log(subj, ses, day, run, step_num=idx, frequency_hz=freq,
            event_label='black_onset', onset=t_black, duration=BLACK_DUR)
        wait_black(win, BLACK_DUR, refresh_hz, subj, ses, day, run)
        log(subj, ses, day, run, step_num=idx, frequency_hz=freq,
            event_label='black_offset', onset=_clk.getTime())

    # ── Cleanup ───────────────────────────────────────────────────────────
    log(subj, ses, day, run, event_label='experiment_end', onset=_clk.getTime())
    close_log()
    close_serial()
    win.close()
    core.quit()

if __name__ == '__main__':
    run_sweep()