#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FEATURES
--------
1. Flashes an image at varying frequencies (default: 8 Hz for 5s -> 12 Hz for 5s -> 9 Hz for 5s).
2. Includes a 3-second delay (inter-block interval with fixation cross) between switching frequencies.
3. Sends hardware TTL trigger (0x01) via COM3 every time just before each flashing block begins, exactly like `object_naming.py`.
4. Prints confirmation to console every time a trigger is successfully sent.
5. Logs exact calendar date/time timestamps and event durations down to the microsecond.
6. Robust square-wave (ON/OFF) frame-by-frame rendering loop prevents screen glitches and accurate frequency modulation across any screen refresh rate.

"""

import os, csv, time, random, re
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
IMAGE_DIR     = 'data'

# Flashing Color Configuration (HEX or named color). Use '#FFDCA8' for a warm tone, or 'white' for pure white.
FLASH_COLOR   = '#babfe0'

# Flashing Sequence Configuration: list of (frequency_in_Hz, duration_in_seconds)
FREQ_SEQUENCE = [
    (8, 20.0),   
    # (9, 20.0),  
    # (10, 20.0),
    # (11, 20.0),
    # (12, 20.0),   
]

DELAY_DUR           = 9.5   # 9.5s delay in between switching frequencies
FIXATION_DUR        = 5.0   # 5.0s initial pre-stimulus fixation before the first flash block
WELCOME_DUR         = 1.0   # 1.0s initial welcome display
GOODBYE_DUR         = 1.0   # 1.0s goodbye screen
FREQ_ANNOUNCE_DUR   = 2.0   # 2.0s display showing what frequency is about to be flashed
PRE_FLASH_FIX_DUR   = 1.0   # 1.0s brief fixation cross right after announcement before flashing begins


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
    # dlg = gui.Dlg(title='Frequency Flashing Task')
    # dlg.addField('Subject ID:', 'S01')
    # dlg.addField('Run (1 or 2):', '1')
    # dlg.addField('Session:', '1')
    # dlg.addField('Day:', '1')
    # dlg.addField('Image Name:', 'blank-white-screen.png')
    # data = dlg.show()
    # if not dlg.OK:
    #     core.quit()
    # return (str(data[0]).strip(), str(data[1]).strip(),
    #         str(data[2]).strip(), str(data[3]).strip(), str(data[4]).strip())    
    # Bypassing the GUI dialogue box as requested
    return ('S01', '1', '1', '1', 'blank-white-screen.png')

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
           'block_num', 'frequency_hz', 'image_name', 'event', 'onset_s', 'duration_s']

def init_log(subj, ses, day, run):
    global _writer, _fh, _clk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = os.path.join(OUTPUT_DIR,
         f'task6_white_flash_{subj}_ses{ses}_day{day}_run{run}_{ts}.csv')
    _fh = open(fn, 'w', newline='', encoding='utf-8')
    _writer = csv.DictWriter(_fh, fieldnames=_FIELDS)
    _writer.writeheader()
    _fh.flush()
    _clk = core.Clock()
    print(f'[LOG] Created log file: {fn}')

def log(subj, ses, day, run, block_num='', frequency_hz='', image_name='',
        event_label='', onset='', duration=''):
    if not _writer:
        return
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    _writer.writerow({
        'timestamp': now_str,
        'subj': subj, 'ses': ses, 'day': day,
        'task': 'frequency_flashing', 'run': run,
        'block_num': block_num, 'frequency_hz': frequency_hz, 'image_name': image_name,
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

def wait_space(win, stims_to_draw, subj, ses, day, run):
    """Draw stims every frame until the space bar is pressed."""
    event.clearEvents()
    while True:
        check_esc(win, subj, ses, day, run)
        for s in stims_to_draw:
            s.draw()
        win.flip()
        keys = event.getKeys(keyList=['space'])
        if keys:
            break

def wait_fix(win, fix, dur, subj, ses, day, run):
    t_end = _clk.getTime() + dur
    while _clk.getTime() < t_end:
        check_esc(win, subj, ses, day, run)
        fix.draw()
        win.flip()

def wait_stims_timed(win, stims_to_draw, dur, subj, ses, day, run):
    """Draw stims every frame for dur seconds."""
    t_end = _clk.getTime() + dur
    while _clk.getTime() < t_end:
        check_esc(win, subj, ses, day, run)
        for s in stims_to_draw:
            s.draw()
        win.flip()

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

    Unlike wall-clock polling, this decides ON/OFF by FRAME COUNT rather
    than elapsed time, so it stays exactly locked to the intended cycle
    structure (dur * freq is always a whole number of cycles in
    FREQ_SEQUENCE). It also paces each flip() to the real target time as a
    safety net: if the GPU driver/compositor isn't honoring vsync (flip()
    returning early instead of waiting for the real screen refresh), an
    unpaced loop would finish the block in the wrong amount of REAL time,
    which would silently break EEG epoch alignment even though the frame
    count was "correct". The explicit deadline below guarantees the block's
    actual wall-clock duration matches `dur` regardless of driver behavior.
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

# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_flashing():
    subj, run, ses, day, image_name = prompt_info()
    init_serial()
    init_log(subj, ses, day, run)

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

    # Create visual stimuli
    msg = visual.TextStim(win, text='', height=0.05,
                          color='white', alignText='center', pos=(0, 0))
    fix = visual.TextStim(win, text='+', height=0.08, color='white')
    img_stim = visual.ImageStim(win, image=None, units='norm', size=(2.0, 2.0), pos=(0, 0))

    # Resolve image path
    img_path = image_name
    if not os.path.isabs(img_path) and not os.path.exists(img_path):
        for candidate_dir in ['', 'data', os.path.join('data', 'images')]:
            candidate = os.path.join(candidate_dir, image_name) if candidate_dir else image_name
            if os.path.exists(candidate):
                img_path = candidate
                break

    if "blank-white-screen.png" in image_name:
        print(f"[STIMULUS] Using full screen rectangle ({FLASH_COLOR}) instead of blank-white-screen.png for performance.")
        rect_stim = visual.Rect(win, units='norm', width=2.0, height=2.0, pos=(0, 0), fillColor=FLASH_COLOR, lineColor=FLASH_COLOR)
        stims_to_flash = [rect_stim]
    elif os.path.exists(img_path):
        print(f"[STIMULUS] Loaded image from {img_path} (Full Screen Mode)")
        img_stim.setImage(img_path)
        stims_to_flash = [img_stim]
    else:
        print(f"[STIMULUS WARNING] Image '{image_name}' not found. Using full screen rectangle ({FLASH_COLOR}) instead.")
        rect_stim = visual.Rect(win, units='norm', width=2.0, height=2.0, pos=(0, 0), fillColor=FLASH_COLOR, lineColor=FLASH_COLOR)
        stims_to_flash = [rect_stim]

    # ── 1. Welcome Screen ─────────────────────────────────────────────────
    msg.text = "Frequency Flashing\n\nExperiment Start Now."
    t_start = _clk.getTime()
    log(subj, ses, day, run, event_label='experiment_start',
        onset=t_start, duration=WELCOME_DUR)
    wait_stims_timed(win, [msg], WELCOME_DUR, subj, ses, day, run)

    # ── 2. Spacebar Prompt ────────────────────────────────────────────────
    msg.text = "To start.\n\nPress the space bar."
    log(subj, ses, day, run, event_label='waiting_for_space', onset=_clk.getTime())
    wait_space(win, [msg], subj, ses, day, run)
    log(subj, ses, day, run, event_label='space_pressed', onset=_clk.getTime())

    # ── 3. Initial Pre-stimulus Fixation (3.5s) ───────────────────────────
    t_fix = _clk.getTime()
    log(subj, ses, day, run, image_name=image_name,
        event_label='initial_fixation_onset', onset=t_fix, duration=FIXATION_DUR)
    wait_fix(win, fix, FIXATION_DUR, subj, ses, day, run)

    # ── 4. Frequency Flashing Blocks Loop ─────────────────────────────────
    for idx, (freq, dur) in enumerate(FREQ_SEQUENCE, start=1):
        n_frames = int(round(dur * refresh_hz))
        print(f"[BLOCK {idx}] Flashing at {freq} Hz for {dur} seconds "
              f"({n_frames} frames @ {refresh_hz:.2f} Hz)...")

        # ── Show Frequency Before Flashing Starts ─────────────────────
        print(f"[ANNOUNCEMENT] Displaying upcoming frequency: {freq} Hz...")
        msg.text = f"Flashing Frequency:\n\n{freq} Hz"
        t_announce = _clk.getTime()
        log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
            event_label='frequency_announcement_onset', onset=t_announce, duration=FREQ_ANNOUNCE_DUR)
        wait_stims_timed(win, [msg], FREQ_ANNOUNCE_DUR, subj, ses, day, run)
        log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
            event_label='frequency_announcement_offset', onset=_clk.getTime())

        # Brief fixation before the flash and trigger
        t_pre_fix = _clk.getTime()
        log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
            event_label='pre_flash_fixation_onset', onset=t_pre_fix, duration=PRE_FLASH_FIX_DUR)
        wait_fix(win, fix, PRE_FLASH_FIX_DUR, subj, ses, day, run)

        # ── Trigger Sent Just Before Flashing Begins ──────────────────
        send_ttl()
        t_flash = _clk.getTime()
        log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
            event_label='trigger_sent', onset=t_flash)
        log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
            event_label='flash_onset', onset=t_flash, duration=dur)

        # ── Flashing Block (e.g. 8 Hz for 5s) ─────────────────────────
        wait_flash(win, stims_to_flash, freq, dur, refresh_hz, subj, ses, day, run)

        log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
            event_label='flash_offset', onset=_clk.getTime())

        # ── 3-Second Delay Between Switching Frequency ────────────────
        if idx < len(FREQ_SEQUENCE):
            print(f"[INTER-BLOCK DELAY] Waiting {DELAY_DUR} seconds before next frequency...")
            t_delay = _clk.getTime()
            log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
                event_label='delay_onset', onset=t_delay, duration=DELAY_DUR)
            wait_fix(win, fix, DELAY_DUR, subj, ses, day, run)
            log(subj, ses, day, run, block_num=idx, frequency_hz=freq, image_name=image_name,
                event_label='delay_offset', onset=_clk.getTime())

    # ── 5. Goodbye Screen ─────────────────────────────────────────────────
    msg.text = "End of the session.\nThank you."
    t_end = _clk.getTime()
    log(subj, ses, day, run, event_label='experiment_end',
        onset=t_end, duration=GOODBYE_DUR)
    wait_stims_timed(win, [msg], GOODBYE_DUR, subj, ses, day, run)

    # ── Cleanup ───────────────────────────────────────────────────────────
    close_log()
    close_serial()
    win.close()
    core.quit()

if __name__ == '__main__':
    run_flashing()