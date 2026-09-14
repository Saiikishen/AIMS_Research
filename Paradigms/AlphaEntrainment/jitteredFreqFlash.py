#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FEATURES
--------
1. Flashes an image/rectangle at varying frequencies (default: 8 Hz -> 11 Hz -> 10 Hz -> 9 Hz -> 12 Hz, 55s each).
2. Continuous, seamless transitions between frequency blocks with NO intermediate gaps, announcements, or fixation crosses.
3. Sends hardware TTL trigger (0x01) via COM5 every 10 seconds throughout continuous flashing, synchronized to frame flips.
4. Comprehensive frame-by-frame stimulus log (stimulus_log.csv) recording instantaneous frequency, time, phase, ON/OFF state, and trigger events for phase locking calculation.
5. High-level event CSV log recording calendar timestamps and event markers.
6. Robust square-wave (ON/OFF) frame-by-frame rendering loop with optional inter-flash jitter.
7. Post-flashing resting phase: 30s eyes-open with fixation cross (showing "Close Your Eyes" in final second), followed by 1 min (60s) eyes-closed resting period with synchronized TTL triggers and auditory cue beeps.
"""

import os, sys, csv, time, random, re, math
from datetime import datetime
# pyrefly: ignore [missing-import]
from psychopy import visual, core, event, gui

try:
    import serial, serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── HARDWARE & CONFIG ─────────────────────────────────────────────────────────
SERIAL_PORT   = 'COM5'
BAUD_RATE     = 115200

# Periodic TTL trigger interval during continuous flashing (seconds)
TRIGGER_INTERVAL_S = 10.0

FULLSCREEN    = True
FALLBACK_SCREEN_SIZE = [1920, 1200]
OUTPUT_DIR    = 'data'
IMAGE_DIR     = 'data'

# Flashing Color Configuration (HEX or named color). Use '#FFDCA8' for a warm tone, or 'white' for pure white.
FLASH_COLOR   = '#5b8cff'

# Flashing Sequence Configuration: list of (frequency_in_Hz, duration_in_seconds)
FREQ_SEQUENCE = [
    (8, 55.0),
    (11, 55.0),  
    (10, 55.0),
    (9, 55.0),
    (12, 55.0),   
]

# ── INTER-FLASH JITTER CONFIG ─────────────────────────────────────────────────
# By default each flash lands at a perfectly constant interval (e.g. exactly every
# 125ms at 8 Hz). Turning jitter on varies the SPACING between individual flashes
# from cycle to cycle while keeping the OVERALL rate across each block locked to
# exactly `freq` Hz -- same total flash count, same block duration, every time.
JITTER_ENABLED   = True
JITTER_FRACTION  = 0.2      # Max deviation from the nominal inter-flash interval
JITTER_MODE      = 'random'  # 'random' or 'alternating'
JITTER_SEED      = None     # Set an int for a reproducible jitter sequence; None for fresh random

WELCOME_DUR         = 1.0   # 1.0s initial welcome display
GOODBYE_DUR         = 1.0   # 1.0s goodbye screen

# ── POST-FLASHING REST & EYES-CLOSED CONFIGURATION ───────────────────────────
POST_FLASH_FIXATION_DUR = 30.0  # seconds for fixation screen (showing "Close Your Eyes" at final second)
EYES_CLOSED_DUR         = 60.0  # seconds for blank screen eyes-closed rest (1 minute)

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

_stimulus_log = []
_session_ts = None

def init_log(subj, ses, day, run):
    global _writer, _fh, _clk, _stimulus_log, _session_ts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = os.path.join(OUTPUT_DIR,
         f'task6_white_flash_{subj}_ses{ses}_day{day}_run{run}_{_session_ts}.csv')
    _fh = open(fn, 'w', newline='', encoding='utf-8')
    _writer = csv.DictWriter(_fh, fieldnames=_FIELDS)
    _writer.writeheader()
    _fh.flush()
    _clk = core.Clock()
    _stimulus_log = []
    print(f'[LOG] Created event log file: {fn}')

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

def save_stimulus_log(subj, ses, day, run):
    global _stimulus_log, _session_ts
    if not _stimulus_log:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = _session_ts or datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Primary path for PhaseLocking.py
    csv_path_primary = os.path.join(OUTPUT_DIR, 'stimulus_log.csv')
    # 2. Timestamped session archive path so data is preserved across runs
    csv_path_archive = os.path.join(
        OUTPUT_DIR,
        f'stimulus_log_{subj}_ses{ses}_day{day}_run{run}_{ts}.csv'
    )

    fieldnames = [
        'time_s', 'global_time_s', 'frame', 'block_frame',
        'block_num', 'frequency_hz', 'stimulus_state',
        'phase_rad', 'phase_deg', 'trigger_sent',
        'cycle', 'breath_phase'
    ]

    for path in [csv_path_primary, csv_path_archive]:
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(_stimulus_log)
            print(f'[LOG] Stimulus log saved: {path} ({len(_stimulus_log)} rows)')
        except Exception as e:
            print(f'[LOG ERROR] Failed to save stimulus log to {path}: {e}')

def close_log():
    if _fh:
        try:
            _fh.close()
        except Exception:
            pass

# ── HELPERS ───────────────────────────────────────────────────────────────────
def abort(win, subj, ses, day, run):
    send_ttl()
    if _clk:
        log(subj, ses, day, run, event_label='abort', onset=_clk.getTime())
    save_stimulus_log(subj, ses, day, run)
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
              f'fix by itself. Falling back to the confirmed panel spec: {fallback_hz} Hz.')
        return float(fallback_hz)
    print(f'[DISPLAY] Measured refresh rate: {measured:.3f} Hz')
    return float(measured)

def build_jittered_cycle_frames(n_flashes, nominal_period_s, refresh_hz,
                                 jitter_fraction, mode, rng):
    """
    Build the number of display frames each individual flash cycle (one
    ON+OFF period, i.e. one inter-flash interval) should last, so the
    SPACING between consecutive flashes varies while the AVERAGE rate over
    the whole block stays locked to the nominal frequency.
    """
    max_delta_s = nominal_period_s * min(jitter_fraction, 0.45)
    periods_s = []
    i = 0
    while i < n_flashes:
        if i + 1 < n_flashes:
            delta = (max_delta_s if mode == 'alternating'
                      else rng.uniform(-max_delta_s, max_delta_s))
            periods_s.append(nominal_period_s + delta)
            periods_s.append(nominal_period_s - delta)
            i += 2
        else:
            # Odd flash left over with no partner -- give it the exact nominal period.
            periods_s.append(nominal_period_s)
            i += 1

    frame_counts = []
    cum_time_s = 0.0
    prev_boundary = 0
    for p in periods_s:
        cum_time_s += p
        boundary = int(round(cum_time_s * refresh_hz))
        frame_counts.append(max(2, boundary - prev_boundary))  # >=2 so there's room for an ON and an OFF frame
        prev_boundary = boundary
    return frame_counts, periods_s

def build_block_frames(freq, dur, refresh_hz, jitter_enabled, jitter_fraction, jitter_mode, rng):
    """
    Build the list of boolean (True=ON, False=OFF) states for every frame in a block
    of duration `dur` at frequency `freq`.
    Guarantees exact total frame count = round(dur * refresh_hz).
    """
    nominal_period_s = 1.0 / float(freq)
    total_frames = int(round(dur * refresh_hz))

    if not jitter_enabled:
        frames_per_cycle = refresh_hz / float(freq)
        half_cycle_frames = frames_per_cycle / 2.0
        frame_states = []
        for f in range(total_frames):
            phase = f % frames_per_cycle
            frame_states.append(phase < half_cycle_frames)
        return frame_states
    else:
        n_flashes = int(round(dur / nominal_period_s))
        cycle_frames, periods_s = build_jittered_cycle_frames(
            n_flashes, nominal_period_s, refresh_hz, jitter_fraction, jitter_mode, rng
        )
        print(f'[JITTER] {freq} Hz block: {n_flashes} flashes, intervals ranged '
              f'{min(periods_s)*1000:.1f}-{max(periods_s)*1000:.1f} ms '
              f'(nominal {nominal_period_s*1000:.1f} ms, mode={jitter_mode})')
        frame_states = []
        for n_cycle_frames in cycle_frames:
            half_cycle_frames = n_cycle_frames / 2.0
            for i in range(n_cycle_frames):
                frame_states.append(i < half_cycle_frames)

        # Guard against minor rounding boundary discrepancies
        if len(frame_states) < total_frames:
            frame_states.extend([False] * (total_frames - len(frame_states)))
        elif len(frame_states) > total_frames:
            frame_states = frame_states[:total_frames]
        return frame_states

# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_flashing():
    subj, run, ses, day, image_name = prompt_info()
    init_serial()
    init_log(subj, ses, day, run)
    jitter_rng = random.Random(JITTER_SEED)  # dedicated RNG so jitter doesn't disturb other random use

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
    img_stim = visual.ImageStim(win, image=None, units='norm', size=(2.0, 2.0), pos=(0, 0))

    # ── Post-Flashing Visual stimuli ──────────────────────────────────────────
    fixation_stim = visual.TextStim(
        win,
        text="+",
        height=0.12,
        color='white',
        pos=(0, 0),
        units='norm',
    )
    close_eyes_stim = visual.TextStim(
        win,
        text="Close Your Eyes",
        height=0.09,
        color='white',
        bold=True,
        pos=(0, 0),
        units='norm',
    )

    # ── Beep Sound ─────────────────────────────────────────────────────────────
    beep_snd = None
    try:
        # pyrefly: ignore [missing-import]
        from psychopy import sound, prefs
        prefs.hardware['audioLib'] = ['ptb', 'sounddevice', 'pygame']
        prefs.hardware['audioDevice'] = ['Headphones (HBTS004)', 'default']
        beep_snd = sound.Sound(value='C', octave=6, secs=0.6, volume=1.0)
    except Exception as exc:
        print(f'[AUDIO WARNING] Could not initialize beep sound: {exc}')

    def play_beep():
        if beep_snd is not None:
            try:
                beep_snd.stop()
                beep_snd.play()
            except Exception as e:
                print(f'[AUDIO WARNING] Error playing beep: {e}')

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

    # Pre-build continuous frame schedules across all frequency blocks
    print("[SCHEDULE] Pre-building continuous frame schedules for all frequency blocks...")
    all_frames = []
    for blk_idx, (freq, dur) in enumerate(FREQ_SEQUENCE, start=1):
        block_states = build_block_frames(
            freq, dur, refresh_hz, JITTER_ENABLED, JITTER_FRACTION, JITTER_MODE, jitter_rng
        )
        for blk_frame_idx, is_on in enumerate(block_states):
            all_frames.append({
                'block_num': blk_idx,
                'frequency_hz': freq,
                'block_frame': blk_frame_idx,
                'is_on': is_on,
            })

    total_duration_s = sum(dur for _, dur in FREQ_SEQUENCE)
    total_frames = len(all_frames)
    print(f"[SCHEDULE] Prepared {total_frames} continuous frames ({total_duration_s:.1f}s total) across {len(FREQ_SEQUENCE)} blocks.")

    # ── 1. Welcome Screen ─────────────────────────────────────────────────
    msg.text = "Frequency Flashing\n\nExperiment Starts Soon."
    t_start = _clk.getTime()
    log(subj, ses, day, run, event_label='experiment_start',
        onset=t_start, duration=WELCOME_DUR)
    wait_stims_timed(win, [msg], WELCOME_DUR, subj, ses, day, run)

    # ── 2. Spacebar Prompt ────────────────────────────────────────────────
    msg.text = "To start continuous flashing,\n\nPress the space bar."
    log(subj, ses, day, run, event_label='waiting_for_space', onset=_clk.getTime())
    wait_space(win, [msg], subj, ses, day, run)
    log(subj, ses, day, run, event_label='space_pressed', onset=_clk.getTime())

    # Brief blank settle before flashing begins
    win.flip()
    core.wait(0.5)

    # ── 3. Continuous Flashing Loop (No breaks, 10s periodic triggers) ────
    TWO_PI = 2.0 * math.pi
    frame_period = 1.0 / refresh_hz
    next_trigger_time = 0.0

    global_clk = core.Clock()
    t_flashing_start = _clk.getTime()
    current_block = None

    print(f"[EXPERIMENT] Flashing started! Continuous display running through {FREQ_SEQUENCE[-1][0]} Hz...")
    log(subj, ses, day, run, event_label='continuous_flashing_start', onset=t_flashing_start)

    for global_idx, frame_info in enumerate(all_frames):
        # 1. Check for ESC key abort
        if event.getKeys(['escape']):
            print("[ABORT] ESC pressed by user. Terminating flashing...")
            send_ttl()
            t_abort = _clk.getTime()
            log(subj, ses, day, run, event_label='abort', onset=t_abort)
            save_stimulus_log(subj, ses, day, run)
            close_log()
            close_serial()
            win.close()
            core.quit()
            return

        blk_num = frame_info['block_num']
        freq = frame_info['frequency_hz']
        blk_frame = frame_info['block_frame']
        is_on = frame_info['is_on']

        # Log transition between frequency blocks seamlessly
        if blk_num != current_block:
            current_block = blk_num
            print(f"[TRANSITION] Seamless switch -> Block {blk_num}: {freq} Hz (frame {global_idx})")
            log(subj, ses, day, run, block_num=blk_num, frequency_hz=freq, image_name=image_name,
                event_label='frequency_block_onset', onset=_clk.getTime())

        # 2. Draw stimulus if state is ON
        if is_on:
            for s in stims_to_flash:
                s.draw()

        # 3. Schedule TTL Trigger every 10 seconds (including t = 0.0s)
        t_flip_target = global_idx * frame_period
        is_trigger_frame = 0
        if t_flip_target >= next_trigger_time - (frame_period * 0.5):
            is_trigger_frame = 1
            win.callOnFlip(send_ttl)
            next_trigger_time += TRIGGER_INTERVAL_S
            log(subj, ses, day, run, block_num=blk_num, frequency_hz=freq, image_name=image_name,
                event_label='trigger_sent', onset=_clk.getTime())

        # 4. Pace to real-time vsync deadline
        target_t = t_flashing_start + (global_idx + 1) * frame_period
        while _clk.getTime() < target_t:
            pass

        # 5. Flip display buffer (hardware VSync)
        win.flip()

        # 6. Record frame stimulus state and trigger status for phase locking
        g_time = global_clk.getTime()
        time_s = (blk_frame + 1) * frame_period
        phase_rad = (TWO_PI * freq * time_s) % TWO_PI
        phase_deg = round(math.degrees(phase_rad), 2)

        _stimulus_log.append({
            'time_s':         round(time_s, 6),
            'global_time_s':  round(g_time, 6),
            'frame':          global_idx,
            'block_frame':    blk_frame,
            'block_num':      blk_num,
            'frequency_hz':   freq,
            'stimulus_state': 'ON' if is_on else 'OFF',
            'phase_rad':      round(phase_rad, 4),
            'phase_deg':      phase_deg,
            'trigger_sent':   is_trigger_frame,
            'cycle':          blk_num,
            'breath_phase':   f"{freq}Hz",
        })

    # Log end of continuous flashing
    t_flashing_end = _clk.getTime()
    log(subj, ses, day, run, event_label='continuous_flashing_end', onset=t_flashing_end)
    print(f"[EXPERIMENT] Continuous flashing completed successfully ({len(_stimulus_log)} frames).")

    # ── 4. POST-FLASHING REST PHASE (Fixation & Eyes-Closed) ─────────────
    # 1. Trigger when the continuous flashing finishes
    print('[TTL] Flashing complete. Sending trigger.')
    send_ttl()
    log(subj, ses, day, run, event_label='flashing_complete_trigger', onset=_clk.getTime())

    # 2. Fixation Screen (30s total, "Close Your Eyes" displayed in final second)
    print(f'[POST-FLASH] Fixation screen for {POST_FLASH_FIXATION_DUR}s (showing "Close Your Eyes" at final sec)...')
    log(subj, ses, day, run, event_label='fixation_start', onset=_clk.getTime(), duration=POST_FLASH_FIXATION_DUR)
    fix_clk = core.Clock()
    while fix_clk.getTime() < POST_FLASH_FIXATION_DUR:
        if event.getKeys(['escape']):
            if beep_snd is not None:
                beep_snd.stop()
            abort(win, subj, ses, day, run)
            return

        t = fix_clk.getTime()
        if t >= (POST_FLASH_FIXATION_DUR - 1.0):
            close_eyes_stim.draw()
        else:
            fixation_stim.draw()

        win.flip()

    # 3. Followed by Beep and transition to 1-minute Blank Screen
    print('[AUDIO] Playing "Close Your Eyes" beep...')
    play_beep()

    # Clear screen to black/background
    win.color = 'black'
    win.flip()

    # Trigger sent when blank screen appears
    print(f'[TTL] Blank screen starting ({EYES_CLOSED_DUR}s eyes-closed). Sending trigger.')
    send_ttl()
    log(subj, ses, day, run, event_label='eyes_closed_start', onset=_clk.getTime(), duration=EYES_CLOSED_DUR)

    blank_clk = core.Clock()
    while blank_clk.getTime() < EYES_CLOSED_DUR:
        if event.getKeys(['escape']):
            if beep_snd is not None:
                beep_snd.stop()
            abort(win, subj, ses, day, run)
            return

        win.flip()

    # 4. End of 1 minute: Play beep sound, send trigger, and finish
    print('[AUDIO] 1-minute eyes-closed complete. Playing final beep...')
    play_beep()

    print('[TTL] Experiment finished. Sending trigger.')
    send_ttl()
    log(subj, ses, day, run, event_label='eyes_closed_end', onset=_clk.getTime())

    # Small delay to let the final beep finish playing
    core.wait(1.0)

    # ── 5. Goodbye Screen ─────────────────────────────────────────────────
    msg.text = "End of the session.\nThank you."
    t_end = _clk.getTime()
    log(subj, ses, day, run, event_label='experiment_end',
        onset=t_end, duration=GOODBYE_DUR)
    wait_stims_timed(win, [msg], GOODBYE_DUR, subj, ses, day, run)

    # ── 6. Save Logs & Cleanup ────────────────────────────────────────────
    save_stimulus_log(subj, ses, day, run)
    close_log()
    close_serial()
    win.close()
    core.quit()

if __name__ == '__main__':
    run_flashing()