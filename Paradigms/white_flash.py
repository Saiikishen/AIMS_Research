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
SCREEN_SIZE   = [1920, 1080]
OUTPUT_DIR    = 'data'
IMAGE_DIR     = 'data'
DEFAULT_IMAGE = os.path.join('data', 'blank-white-screen.png')

# Flashing Sequence Configuration: list of (frequency_in_Hz, duration_in_seconds)
FREQ_SEQUENCE = [
    (8, 10.0),   # 8 Hz for 5 seconds
    (9, 10.0),  # 12 Hz for 5 seconds
    (10, 10.0),
    (11, 10.0), 
    (12, 10.0),  # 9 Hz for 5 seconds
]

DELAY_DUR           = 5.0   # 5.0s delay in between switching frequencies
FIXATION_DUR        = 5.0   # 5.0s initial pre-stimulus fixation before the first flash block
WELCOME_DUR         = 1.0   # 1.0s initial welcome display
GOODBYE_DUR         = 1.0   # 1.0s goodbye screen
FREQ_ANNOUNCE_DUR   = 2.0   # 2.0s display showing what frequency is about to be flashed
PRE_FLASH_FIX_DUR   = 1.0   # 1.0s brief fixation cross right after announcement before flashing begins

# ── RUNTIME PROMPT ────────────────────────────────────────────────────────────
def prompt_info():
    dlg = gui.Dlg(title='Frequency Flashing Task')
    dlg.addField('Subject ID:', 'S01')
    dlg.addField('Run (1 or 2):', '1')
    dlg.addField('Session:', '1')
    dlg.addField('Day:', '1')
    dlg.addField('Image Name:', DEFAULT_IMAGE)
    data = dlg.show()
    if not dlg.OK:
        core.quit()
    return (str(data[0]).strip(), str(data[1]).strip(),
            str(data[2]).strip(), str(data[3]).strip(), str(data[4]).strip())

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
        core.wait(0.001, hogCPUperiod=0.0)

def wait_fix(win, fix, dur, subj, ses, day, run):
    t_end = _clk.getTime() + dur
    while _clk.getTime() < t_end:
        check_esc(win, subj, ses, day, run)
        fix.draw()
        win.flip()
        core.wait(0.001, hogCPUperiod=0.0)

def wait_stims_timed(win, stims_to_draw, dur, subj, ses, day, run):
    """Draw stims every frame for dur seconds."""
    t_end = _clk.getTime() + dur
    while _clk.getTime() < t_end:
        check_esc(win, subj, ses, day, run)
        for s in stims_to_draw:
            s.draw()
        win.flip()
        core.wait(0.001, hogCPUperiod=0.0)

def wait_flash(win, stims_to_draw, freq, dur, subj, ses, day, run):
    """
    Draw stims flashing at `freq` Hz (50% duty cycle ON/OFF square wave) for `dur` seconds.
    During each cycle T = 1/freq:
    - Stimulus is ON (drawn) for the first half of the cycle (0 to T/2).
    - Stimulus is OFF (not drawn / background screen) for the second half of the cycle (T/2 to T).
    """
    t_start = _clk.getTime()
    t_end = t_start + dur
    cycle_dur = 1.0 / float(freq)
    half_cycle = cycle_dur / 2.0

    while _clk.getTime() < t_end:
        check_esc(win, subj, ses, day, run)
        t_elapsed = _clk.getTime() - t_start
        # If inside the first half of the cycle, draw the stimulus
        if (t_elapsed % cycle_dur) < half_cycle:
            for s in stims_to_draw:
                s.draw()
        win.flip()
        core.wait(0.001, hogCPUperiod=0.0)

# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_flashing():
    subj, run, ses, day, image_name = prompt_info()
    init_serial()
    init_log(subj, ses, day, run)

    # Setup Window
    win = visual.Window(SCREEN_SIZE, fullscr=FULLSCREEN,
                        color='black', units='height', allowGUI=False)
    win.mouseVisible = False

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

    if os.path.exists(img_path):
        print(f"[STIMULUS] Loaded image from {img_path} (Full Screen Mode)")
        img_stim.setImage(img_path)
        stims_to_flash = [img_stim]
    else:
        print(f"[STIMULUS WARNING] Image '{image_name}' not found. Using full screen white rectangle instead.")
        rect_stim = visual.Rect(win, units='norm', width=2.0, height=2.0, pos=(0, 0), fillColor='white', lineColor='white')
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
        print(f"[BLOCK {idx}] Flashing at {freq} Hz for {dur} seconds...")

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
        wait_flash(win, stims_to_flash, freq, dur, subj, ses, day, run)

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
