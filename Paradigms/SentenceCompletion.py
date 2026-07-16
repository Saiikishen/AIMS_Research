#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FEATURES
--------
1. Loads 30 images (Picture1.png to Picture30.png) from `data/images` sequentially.
2. Sends hardware TTL trigger (0x01) via COM3 when each picture
   is shown (after a 3.5s fixation cross).
3. Prints confirmation to console every time a trigger is successfully sent.
4. Logs exact calendar date/time timestamps down to the microsecond for every event.
5. Robust frame-by-frame rendering loop prevents blank screen glitches.
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
IMAGE_DIR     = os.path.join('data', 'block-2 sentence completion')

FIXATION_DUR  = 3.5   # 3.5s pre-stimulus fixation
IMAGE_DUR     = 2.0   # 4.0s picture display
WELCOME_DUR   = 1.0   # 1.0s initial welcome display
GOODBYE_DUR   = 1.0   # 1.0s goodbye screen

# ── RUNTIME PROMPT ────────────────────────────────────────────────────────────
def prompt_info():
    dlg = gui.Dlg(title='Sentence Completion')
    dlg.addField('Subject ID:', 'S01')
    dlg.addField('Run (1 or 2):', '1')
    dlg.addField('Session:', '1')
    dlg.addField('Day:', '1')
    data = dlg.show()
    if not dlg.OK:
        core.quit()
    return (str(data[0]).strip(), str(data[1]).strip(),
            str(data[2]).strip(), str(data[3]).strip())

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
           'trial_num', 'image_name', 'event', 'onset_s', 'duration_s']

def init_log(subj, ses, day, run):
    global _writer, _fh, _clk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = os.path.join(OUTPUT_DIR,
         f'task5_picture_naming_{subj}_ses{ses}_day{day}_run{run}_{ts}.csv')
    _fh = open(fn, 'w', newline='', encoding='utf-8')
    _writer = csv.DictWriter(_fh, fieldnames=_FIELDS)
    _writer.writeheader()
    _fh.flush()
    _clk = core.Clock()
    print(f'[LOG] Created log file: {fn}')

def log(subj, ses, day, run, trial_num='', image_name='',
        event_label='', onset='', duration=''):
    if not _writer:
        return
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    _writer.writerow({
        'timestamp': now_str,
        'subj': subj, 'ses': ses, 'day': day,
        'task': 'Sentence Completion', 'run': run,
        'trial_num': trial_num, 'image_name': image_name,
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

# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_picture_naming():
    subj, run, ses, day = prompt_info()
    init_serial()
    init_log(subj, ses, day, run)

    # Discover and sort image files (Picture1.png to Picture30.png)
    if not os.path.exists(IMAGE_DIR):
        print(f"[ERROR] Image directory not found: {IMAGE_DIR}")
        close_log()
        close_serial()
        core.quit()

    img_files = [f for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Sort numerically by extracting integers from filenames (e.g., Picture1 -> 1, Picture2 -> 2)
    img_files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    if not img_files:
        print(f"[ERROR] No image files found in {IMAGE_DIR}")
        close_log()
        close_serial()
        core.quit()

    print(f"[STIMULI] Loaded {len(img_files)} images from {IMAGE_DIR}")

    # Setup Window
    win = visual.Window(SCREEN_SIZE, fullscr=FULLSCREEN,
                        color='black', units='height', allowGUI=False)
    win.mouseVisible = False

    # Create visual stimuli
    msg = visual.TextStim(win, text='', height=0.05,
                          color='white', alignText='center', pos=(0, 0))
    fix = visual.TextStim(win, text='+', height=0.08, color='white')
    img_stim = visual.ImageStim(win, image=None, size=(0.5, 0.5))

    # ── 1. Welcome Screen ─────────────────────────────────────────────────
    msg.text = "Sentence Completion\n\nExperiment Start Now."
    t_start = _clk.getTime()
    log(subj, ses, day, run, event_label='experiment_start',
        onset=t_start, duration=WELCOME_DUR)
    wait_stims_timed(win, [msg], WELCOME_DUR, subj, ses, day, run)

    # ── 2. Spacebar Prompt ────────────────────────────────────────────────
    msg.text = "To start.\n\nPress the space bar."
    log(subj, ses, day, run, event_label='waiting_for_space', onset=_clk.getTime())
    wait_space(win, [msg], subj, ses, day, run)
    log(subj, ses, day, run, event_label='space_pressed', onset=_clk.getTime())

    # ── 3. Picture Naming Trial Loop ──────────────────────────────────────
    for tri, img_name in enumerate(img_files, start=1):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_stim.setImage(img_path)

        # ── Pre-stimulus fixation (3.5s) ──────────────────────────────
        t_fix = _clk.getTime()
        log(subj, ses, day, run, trial_num=tri, image_name=img_name,
            event_label='fixation_onset', onset=t_fix, duration=FIXATION_DUR)
        wait_fix(win, fix, FIXATION_DUR, subj, ses, day, run)

        # ── Picture Presentation (2.0s) + TTL Trigger ─────────────────
        send_ttl()
        t_stim = _clk.getTime()
        log(subj, ses, day, run, trial_num=tri, image_name=img_name,
            event_label='trigger_sent', onset=t_stim)
        log(subj, ses, day, run, trial_num=tri, image_name=img_name,
            event_label='stimulus_onset', onset=t_stim, duration=IMAGE_DUR)
        wait_stims_timed(win, [img_stim], IMAGE_DUR, subj, ses, day, run)

        log(subj, ses, day, run, trial_num=tri, image_name=img_name,
            event_label='stimulus_offset', onset=_clk.getTime())

    # ── 4. Goodbye Screen ─────────────────────────────────────────────────
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
    run_picture_naming()
