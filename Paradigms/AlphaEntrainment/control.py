#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, csv, time, random, math
from datetime import datetime

# pyrefly: ignore [missing-import]
from psychopy import visual, core, event, gui

try:
    import serial, serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── HARDWARE & CONFIG ─────────────────────────────────────────────────────────
SERIAL_PORT  = 'COM5'
BAUD_RATE    = 115200

FULLSCREEN   = True
FALLBACK_SCREEN_SIZE = [1920, 1200]
OUTPUT_DIR   = 'data'

# Flashing stimulus colour (HEX or named). '#5b8cff' = blue, 'white' = pure white.
FLASH_COLOR  = '#5b8cff'

# ── EPOCH-BASED RANDOM FREQUENCY CONFIG ──────────────────────────────────────
TOTAL_FLASH_DURATION_S = 300.0   # 5 minutes of continuous flashing
EPOCH_DUR_MIN_S        = 10.0    # minimum epoch duration (seconds)
EPOCH_DUR_MAX_S        = 20.0    # maximum epoch duration (seconds)
FREQ_MIN_HZ            = 8.0     # minimum flash frequency (Hz, float)
FREQ_MAX_HZ            = 13.0    # maximum flash frequency (Hz, float)
FREQ_STEP_HZ           = 0.2     # step size for frequency pool (e.g. 0.5 → 8.0, 8.5, …, 13.0)
                                  # Smaller step = more unique frequencies available per run.
                                  # Pool size = (FREQ_MAX_HZ - FREQ_MIN_HZ) / FREQ_STEP_HZ + 1
                                  # At 0.5 Hz step: 11 unique freqs; at 0.25: 21; at 0.1: 51
EPOCH_SEED             = None    # int for a reproducible epoch schedule; None = fresh random

# ── INTER-FLASH JITTER CONFIG ─────────────────────────────────────────────────
JITTER_ENABLED   = True
JITTER_FRACTION  = 0.2      # max deviation fraction from the nominal inter-flash interval
JITTER_MODE      = 'random'  # 'random' or 'alternating'
JITTER_SEED      = None     # int for reproducible jitter; None = fresh random

WELCOME_DUR  = 1.0   # seconds for welcome screen
GOODBYE_DUR  = 1.0   # seconds for goodbye screen

# ── POST-FLASHING REST CONFIGURATION ─────────────────────────────────────────
POST_FLASH_FIXATION_DUR = 30.0  # eyes-open fixation cross duration (s)
EYES_CLOSED_DUR         = 60.0  # eyes-closed blank-screen duration (s)

NOMINAL_REFRESH_HZ = 120.0
MIN_PLAUSIBLE_HZ   = 30.0
MAX_PLAUSIBLE_HZ   = 300.0


# ── DISPLAY RESOLUTION ────────────────────────────────────────────────────────
def get_native_resolution(fallback=FALLBACK_SCREEN_SIZE):
    """
    Query the true native pixel resolution on Windows, bypassing DPI scaling
    so the PsychoPy window fills the full physical panel.
    """
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        width  = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        if width > 0 and height > 0:
            return [int(width), int(height)]
    except Exception as e:
        print(f'[DISPLAY WARNING] Could not query native resolution ({e}); '
              f'using fallback {fallback}.')
    return list(fallback)


# ── RUNTIME PROMPT ────────────────────────────────────────────────────────────
def prompt_info():
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


# ── LOGGING (single stimulus_log.csv, PhaseLocking.py compatible) ─────────────
_stimulus_log = []
_session_ts   = None
_clk          = None

_STIM_FIELDS = [
    'time_s', 'global_time_s', 'frame', 'block_frame',
    'block_num', 'frequency_hz', 'stimulus_state',
    'phase_rad', 'phase_deg', 'trigger_sent',
    'cycle', 'breath_phase',
]

def init_log():
    global _stimulus_log, _session_ts, _clk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _session_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    _stimulus_log = []
    _clk          = core.Clock()
    print(f'[LOG] Stimulus log initialised (session: {_session_ts})')

def save_stimulus_log(subj, ses, day, run):
    global _stimulus_log, _session_ts
    if not _stimulus_log:
        print('[LOG] No stimulus frames to save.')
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = _session_ts or datetime.now().strftime('%Y%m%d_%H%M%S')

    # Primary path used by PhaseLocking.py (overwritten each run)
    csv_primary = os.path.join(OUTPUT_DIR, 'stimulus_log.csv')
    # Timestamped archive to preserve data across runs
    csv_archive = os.path.join(
        OUTPUT_DIR,
        f'stimulus_log_control_{subj}_ses{ses}_day{day}_run{run}_{ts}.csv'
    )

    for path in [csv_primary, csv_archive]:
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=_STIM_FIELDS)
                writer.writeheader()
                writer.writerows(_stimulus_log)
            print(f'[LOG] Saved: {path}  ({len(_stimulus_log)} rows)')
        except Exception as e:
            print(f'[LOG ERROR] Failed to save {path}: {e}')


# ── HELPERS ───────────────────────────────────────────────────────────────────
def abort(win, subj, ses, day, run):
    send_ttl()
    save_stimulus_log(subj, ses, day, run)
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
        if event.getKeys(keyList=['space']):
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
    Measure the monitor's actual refresh rate. Rejects any value outside a
    plausible physical range (driver vsync broken) and falls back to the
    confirmed panel spec.
    """
    print('[DISPLAY] Measuring monitor refresh rate (please wait)...')
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                       nWarmUpFrames=15, threshold=1)
    if measured is None or not (MIN_PLAUSIBLE_HZ <= measured <= MAX_PLAUSIBLE_HZ):
        print(f'[DISPLAY WARNING] Measured value ({measured}) outside plausible range. '
              f'Falling back to configured panel spec: {fallback_hz} Hz.')
        return float(fallback_hz)
    print(f'[DISPLAY] Measured refresh rate: {measured:.3f} Hz')
    return float(measured)


# ── EPOCH SCHEDULE GENERATION ─────────────────────────────────────────────────
def generate_epoch_sequence(total_dur, dur_min, dur_max,
                            freq_min, freq_max, freq_step, rng):
    n_steps = int(round((freq_max - freq_min) / freq_step))
    pool    = [round(freq_min + i * freq_step, 10) for i in range(n_steps + 1)]
    rng.shuffle(pool)
    pool_iter = iter(pool)

    sequence = []
    elapsed  = 0.0
    while elapsed < total_dur:
        remaining = total_dur - elapsed
        if remaining <= 0:
            break

        # Draw next unique frequency from the pre-shuffled pool
        try:
            freq = next(pool_iter)
        except StopIteration:
            raise RuntimeError(
                f'[EPOCH] Frequency pool exhausted after {len(sequence)} epochs '
                f'({elapsed:.1f}s / {total_dur:.1f}s covered). '
                f'Decrease FREQ_STEP_HZ to expand the pool, or reduce '
                f'TOTAL_FLASH_DURATION_S / increase epoch durations.'
            )

        dur = rng.uniform(dur_min, dur_max)
        dur = min(dur, remaining)   # clamp last epoch
        sequence.append((freq, round(dur, 4)))
        elapsed += dur
    return sequence


# ── JITTER FRAME BUILDING ─────────────────────────────────────────────────────
def build_jittered_cycle_frames(n_flashes, nominal_period_s, refresh_hz,
                                 jitter_fraction, mode, rng):
    """
    Compute per-cycle frame counts so that flash SPACING varies while the
    AVERAGE rate over the epoch stays locked to the nominal frequency.
    """
    max_delta_s = nominal_period_s * min(jitter_fraction, 0.45)
    periods_s   = []
    i = 0
    while i < n_flashes:
        if i + 1 < n_flashes:
            delta = (max_delta_s if mode == 'alternating'
                     else rng.uniform(-max_delta_s, max_delta_s))
            periods_s.append(nominal_period_s + delta)
            periods_s.append(nominal_period_s - delta)
            i += 2
        else:
            periods_s.append(nominal_period_s)
            i += 1

    frame_counts = []
    cum_time_s   = 0.0
    prev_boundary = 0
    for p in periods_s:
        cum_time_s += p
        boundary = int(round(cum_time_s * refresh_hz))
        frame_counts.append(max(2, boundary - prev_boundary))
        prev_boundary = boundary
    return frame_counts, periods_s


def build_epoch_frames(freq, dur, refresh_hz, jitter_enabled,
                        jitter_fraction, jitter_mode, rng):
    """
    Build the list of boolean (True=ON, False=OFF) states for every frame in
    one epoch of duration `dur` at frequency `freq`.
    Guarantees exact total frame count = round(dur * refresh_hz).
    """
    nominal_period_s = 1.0 / float(freq)
    total_frames     = int(round(dur * refresh_hz))

    if not jitter_enabled:
        frames_per_cycle    = refresh_hz / float(freq)
        half_cycle_frames   = frames_per_cycle / 2.0
        return [((f % frames_per_cycle) < half_cycle_frames) for f in range(total_frames)]

    n_flashes = int(round(dur / nominal_period_s))
    cycle_frames, periods_s = build_jittered_cycle_frames(
        n_flashes, nominal_period_s, refresh_hz, jitter_fraction, jitter_mode, rng
    )
    print(f'[JITTER] {freq} Hz epoch ({dur:.2f}s): {n_flashes} flashes, '
          f'intervals {min(periods_s)*1000:.1f}–{max(periods_s)*1000:.1f} ms '
          f'(nominal {nominal_period_s*1000:.1f} ms)')

    frame_states = []
    for n_cycle_frames in cycle_frames:
        half = n_cycle_frames / 2.0
        frame_states.extend([i < half for i in range(n_cycle_frames)])

    # Guard rounding discrepancies
    if len(frame_states) < total_frames:
        frame_states.extend([False] * (total_frames - len(frame_states)))
    elif len(frame_states) > total_frames:
        frame_states = frame_states[:total_frames]
    return frame_states


# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_flashing():
    subj, run, ses, day, image_name = prompt_info()
    init_serial()
    init_log()

    # Dedicated RNGs — epoch schedule and jitter are independent
    epoch_rng  = random.Random(EPOCH_SEED)
    jitter_rng = random.Random(JITTER_SEED)

    # ── Window setup ──────────────────────────────────────────────────────────
    screen_size = get_native_resolution()
    print(f'[DISPLAY] Detected native resolution: {screen_size[0]}x{screen_size[1]}')
    win = visual.Window(screen_size, fullscr=FULLSCREEN,
                        color='black', units='height', allowGUI=False,
                        waitBlanking=True, useFBO=True, winType='pyglet')
    win.mouseVisible = False
    print(f'[DISPLAY] Window buffer resolution: {win.size[0]}x{win.size[1]}')

    refresh_hz = measure_refresh_rate(win)

    # ── Visual stimuli ────────────────────────────────────────────────────────
    msg = visual.TextStim(win, text='', height=0.05, color='white',
                          alignText='center', pos=(0, 0))
    img_stim = visual.ImageStim(win, image=None, units='norm',
                                size=(2.0, 2.0), pos=(0, 0))
    fixation_stim = visual.TextStim(win, text='+', height=0.12,
                                    color='white', pos=(0, 0), units='norm')
    close_eyes_stim = visual.TextStim(win, text='Close Your Eyes', height=0.09,
                                      color='white', bold=True,
                                      pos=(0, 0), units='norm')

    # ── Beep sound ────────────────────────────────────────────────────────────
    beep_snd = None
    try:
        # pyrefly: ignore [missing-import]
        from psychopy import sound, prefs
        prefs.hardware['audioLib']    = ['ptb', 'sounddevice', 'pygame']
        prefs.hardware['audioDevice'] = ['Headphones (HBTS004)', 'default']
        beep_snd = sound.Sound(value='C', octave=6, secs=0.6, volume=1.0)
    except Exception as exc:
        print(f'[AUDIO WARNING] Could not initialise beep: {exc}')

    def play_beep():
        if beep_snd is not None:
            try:
                beep_snd.stop()
                beep_snd.play()
            except Exception as e:
                print(f'[AUDIO WARNING] Error playing beep: {e}')

    # ── Resolve stimulus ──────────────────────────────────────────────────────
    img_path = image_name
    if not os.path.isabs(img_path) and not os.path.exists(img_path):
        for candidate_dir in ['', 'data', os.path.join('data', 'images')]:
            candidate = os.path.join(candidate_dir, image_name) if candidate_dir else image_name
            if os.path.exists(candidate):
                img_path = candidate
                break

    if 'blank-white-screen.png' in image_name:
        print(f'[STIMULUS] Using full-screen rectangle ({FLASH_COLOR}).')
        rect_stim = visual.Rect(win, units='norm', width=2.0, height=2.0,
                                pos=(0, 0), fillColor=FLASH_COLOR, lineColor=FLASH_COLOR)
        stims_to_flash = [rect_stim]
    elif os.path.exists(img_path):
        print(f'[STIMULUS] Loaded image: {img_path}')
        img_stim.setImage(img_path)
        stims_to_flash = [img_stim]
    else:
        print(f'[STIMULUS WARNING] "{image_name}" not found. Using rectangle ({FLASH_COLOR}).')
        rect_stim = visual.Rect(win, units='norm', width=2.0, height=2.0,
                                pos=(0, 0), fillColor=FLASH_COLOR, lineColor=FLASH_COLOR)
        stims_to_flash = [rect_stim]

    # ── Generate epoch schedule ───────────────────────────────────────────────
    epoch_sequence = generate_epoch_sequence(
        TOTAL_FLASH_DURATION_S,
        EPOCH_DUR_MIN_S, EPOCH_DUR_MAX_S,
        FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_STEP_HZ,
        epoch_rng,
    )
    print(f'[SCHEDULE] Generated {len(epoch_sequence)} epochs '
          f'({sum(d for _, d in epoch_sequence):.2f}s total):')
    for i, (f, d) in enumerate(epoch_sequence, 1):
        print(f'  Epoch {i:>2}: {f:.2f} Hz  {d:.2f}s')

    # ── Pre-build per-epoch frame lists ───────────────────────────────────────
    print('[SCHEDULE] Pre-building frame schedules for all epochs...')
    all_frames = []   # list of dicts; one entry per display frame
    for blk_idx, (freq, dur) in enumerate(epoch_sequence, start=1):
        epoch_states = build_epoch_frames(
            freq, dur, refresh_hz,
            JITTER_ENABLED, JITTER_FRACTION, JITTER_MODE, jitter_rng,
        )
        n_epoch_frames = len(epoch_states)
        for blk_frame_idx, is_on in enumerate(epoch_states):
            all_frames.append({
                'block_num':    blk_idx,
                'frequency_hz': freq,
                'duration_s':   dur,
                'block_frame':  blk_frame_idx,
                'n_epoch_frames': n_epoch_frames,
                'is_on':        is_on,
            })

    total_frames = len(all_frames)
    print(f'[SCHEDULE] {total_frames} frames prepared across {len(epoch_sequence)} epochs.')

    # ── 1. Welcome Screen ─────────────────────────────────────────────────────
    msg.text = 'Frequency Flashing\n\nExperiment Starts Soon.'
    wait_stims_timed(win, [msg], WELCOME_DUR, subj, ses, day, run)

    # ── 2. Spacebar Prompt ────────────────────────────────────────────────────
    msg.text = 'To start continuous flashing,\n\nPress the space bar.'
    wait_space(win, [msg], subj, ses, day, run)

    # Brief blank settle before flashing
    win.flip()
    core.wait(0.5)

    # ── 3. Continuous Epoch Flashing Loop ─────────────────────────────────────
    TWO_PI      = 2.0 * math.pi
    frame_period = 1.0 / refresh_hz

    global_clk       = core.Clock()
    t_flashing_start = _clk.getTime()
    current_block    = None
    prev_block       = None

    print(f'[EXPERIMENT] Continuous epoch flashing started '
          f'({len(epoch_sequence)} epochs, {TOTAL_FLASH_DURATION_S:.0f}s total).')

    for global_idx, frame_info in enumerate(all_frames):
        # ── ESC abort ────────────────────────────────────────────────────────
        if event.getKeys(['escape']):
            print('[ABORT] ESC pressed.')
            send_ttl()
            save_stimulus_log(subj, ses, day, run)
            close_serial()
            win.close()
            core.quit()
            return

        blk_num         = frame_info['block_num']
        freq            = frame_info['frequency_hz']
        blk_frame       = frame_info['block_frame']
        n_epoch_frames  = frame_info['n_epoch_frames']
        is_on           = frame_info['is_on']
        is_trigger_frame = 0

        # ── Epoch-boundary trigger: ONSET ─────────────────────────────────
        # Fire on the very first frame of each new epoch
        if blk_num != current_block:
            # Send offset trigger for the epoch that just ended
            if current_block is not None:
                # The previous epoch's offset trigger was scheduled on its last frame
                # (handled below in the offset check). Nothing extra needed here.
                pass
            prev_block    = current_block
            current_block = blk_num
            is_trigger_frame = 1      # mark as onset trigger frame
            win.callOnFlip(send_ttl)
            print(f'[EPOCH ONSET] Epoch {blk_num}: {freq} Hz  '
                  f'{frame_info["duration_s"]:.2f}s  (frame {global_idx})')

        # ── Epoch-boundary trigger: OFFSET ────────────────────────────────
        # Fire on the very last frame of each epoch (and it is NOT also an onset frame)
        elif blk_frame == n_epoch_frames - 1:
            is_trigger_frame = 1      # mark as offset trigger frame
            win.callOnFlip(send_ttl)
            print(f'[EPOCH OFFSET] Epoch {blk_num}: {freq} Hz  (frame {global_idx})')

        # ── Draw stimulus ─────────────────────────────────────────────────
        if is_on:
            for s in stims_to_flash:
                s.draw()

        # ── Pace to vsync deadline ────────────────────────────────────────
        target_t = t_flashing_start + (global_idx + 1) * frame_period
        while _clk.getTime() < target_t:
            pass

        # ── Flip ──────────────────────────────────────────────────────────
        win.flip()

        # ── Record stimulus frame ─────────────────────────────────────────
        g_time    = global_clk.getTime()
        time_s    = (blk_frame + 1) * frame_period
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
            'breath_phase':   f'{freq}Hz',
        })

    print(f'[EXPERIMENT] Continuous flashing complete ({len(_stimulus_log)} frames logged).')

    # ── 4. POST-FLASHING REST PHASE ───────────────────────────────────────────

    # 4a. Fixation cross (30 s; "Close Your Eyes" in final 1 s)
    print(f'[POST-FLASH] Fixation screen for {POST_FLASH_FIXATION_DUR}s...')
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

    # 4b. Beep + trigger → eyes-closed blank screen starts
    print('[AUDIO] Playing "Close Your Eyes" beep...')
    play_beep()

    win.color = 'black'
    win.flip()

    print(f'[TTL] Eyes-closed screen starting ({EYES_CLOSED_DUR}s). Sending trigger.')
    send_ttl()

    # 4c. Eyes-closed blank screen (60 s)
    blank_clk = core.Clock()
    while blank_clk.getTime() < EYES_CLOSED_DUR:
        if event.getKeys(['escape']):
            if beep_snd is not None:
                beep_snd.stop()
            abort(win, subj, ses, day, run)
            return
        win.flip()

    # 4d. Final beep + trigger → experiment ends
    print('[AUDIO] 1-minute eyes-closed complete. Playing final beep...')
    play_beep()

    print('[TTL] Experiment finished. Sending trigger.')
    send_ttl()

    core.wait(1.0)  # allow beep to finish

    # ── 5. Goodbye Screen ─────────────────────────────────────────────────────
    msg.text = 'End of the session.\nThank you.'
    wait_stims_timed(win, [msg], GOODBYE_DUR, subj, ses, day, run)

    # ── 6. Save log & clean up ────────────────────────────────────────────────
    save_stimulus_log(subj, ses, day, run)
    close_serial()
    win.close()
    core.quit()


if __name__ == '__main__':
    run_flashing()
