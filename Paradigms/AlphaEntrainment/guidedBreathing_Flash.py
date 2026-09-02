#!/usr/bin/env python
import os
import sys
import time

try:
    import serial, serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── HARDWARE & CONFIG ─────────────────────────────────────────────────────────
SERIAL_PORT   = 'COM5'
BAUD_RATE     = 115200

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

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
INHALE_DUR        = 4.0       # seconds for breathe-IN  phase
HOLD_DUR          = 0.0       # seconds for hold-breath phase (set 0 to skip)
EXHALE_DUR        = 6.0       # seconds for breathe-OUT phase
NUM_CYCLES        = 27      # number of complete breath cycles (0 = infinite)
FULLSCREEN        = True

# ── POST-BREATHING REST & EYES-CLOSED CONFIGURATION ───────────────────────────
POST_BREATH_FIXATION_DUR = 15.0  # seconds for fixation screen (at 15th sec: "Close Your Eyes")
EYES_CLOSED_DUR          = 60.0  # seconds for blank screen eyes-closed rest (1 minute)

# ── FLASH FREQUENCY DEFAULT (Hz) ─────────────────────────────────────────────
DEFAULT_FLASH_HZ  = 8.97    # default flicker rate if not modified in dialog

# ── FLASH COLOURS ─────────────────────────────────────────────────────────────
FLASH_COLOR_INHALE  = '#5b8cff'   # cool blue   -- breath in
FLASH_COLOR_HOLD    = '#5b8cff'   # warm gold   -- hold breath
FLASH_COLOR_EXHALE  = '#5b8cff'   # soft violet -- breath out
BG_COLOR            = '#0d0d1a'   # background / OFF colour

# ── SCREEN REFRESH (fallback if measurement fails) ───────────────────────────
NOMINAL_REFRESH_HZ  = 120.0
MIN_PLAUSIBLE_HZ    = 30.0
MAX_PLAUSIBLE_HZ    = 300.0


OM_AUDIO   = r'C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\Audio\Om.wav'    
MAA_AUDIO  = r'C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\Audio\MAA.wav'
# ── INSTRUCTION SCREEN COLOURS ────────────────────────────────────────────────
INSTRUCTION_COLOR = '#d0e8ff'
TEXT_COLOR        = 'white'


# ==============================================================================
# HELPERS
# ==============================================================================

def prompt_frequency(default_hz=DEFAULT_FLASH_HZ):
    """
    Prompt the user for the flash frequency in Hz via a GUI dialog box.
    """
    # pyrefly: ignore [missing-import]
    from psychopy import gui, core
    dlg = gui.Dlg(title='Guided Breathing Flash')
    dlg.addField('Flash Frequency (Hz):', str(default_hz))
    data = dlg.show()
    if not dlg.OK:
        core.quit()
        sys.exit(0)

    freq_str = str(data[0]).strip()
    try:
        freq = float(freq_str)
        if freq <= 0:
            raise ValueError('Frequency must be greater than 0 Hz')
        return freq
    except ValueError as e:
        print(f'[INPUT WARNING] Invalid frequency "{freq_str}" ({e}); using fallback {default_hz} Hz.')
        return float(default_hz)


def measure_refresh_rate(win, fallback_hz=NOMINAL_REFRESH_HZ):
    """
    Measure the monitor actual refresh rate. Rejects implausible readings
    (e.g. from a GPU driver that does not honour vsync) and falls back to the
    confirmed panel spec instead.
    """
    # pyrefly: ignore [missing-import]
    from psychopy import core as _core
    print('[DISPLAY] Measuring actual monitor refresh rate (please wait)...')
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                      nWarmUpFrames=15, threshold=1)
    if measured is None or not (MIN_PLAUSIBLE_HZ <= measured <= MAX_PLAUSIBLE_HZ):
        print(f'[DISPLAY WARNING] Measured value ({measured}) implausible; '
              f'using fallback {fallback_hz} Hz.')
        return float(fallback_hz)
    print(f'[DISPLAY] Measured refresh rate: {measured:.3f} Hz')
    return float(measured)


def flash_phase(win, rect, flash_hz, duration_s, refresh_hz, clk, event,
                snd=None, audio_ok=False, on_first_flip=None):
    """
    on_first_flip: optional callable scheduled via win.callOnFlip so it fires
    at the exact VSync of the very first rendered frame -- i.e. the trigger
    and the first screen flash are hardware-synchronised to the same refresh.
    """
    frames_per_cycle  = refresh_hz / flash_hz
    half_cycle_frames = frames_per_cycle / 2.0
    total_frames      = int(round(duration_s * refresh_hz))
    frame_period      = 1.0 / refresh_hz

    if audio_ok and snd is not None:
        snd.stop()
        snd.play()

    t_start = clk.getTime()

    for frame_n in range(total_frames):
        if event.getKeys(['escape']):
            if audio_ok and snd is not None:
                snd.stop()
            return False

        phase = frame_n % frames_per_cycle
        if phase < half_cycle_frames:
            rect.draw()          # ON frame  -- colour visible
        # OFF frame  -- window cleared to BG_COLOR (set on win creation)

        # Schedule trigger to fire at the exact VSync of the first frame only
        if frame_n == 0 and on_first_flip is not None:
            win.callOnFlip(on_first_flip)

        # Pace to real-time deadline (guard against non-vsync drivers)
        target_t = t_start + (frame_n + 1) * frame_period
        while clk.getTime() < target_t:
            pass
        win.flip()

    if audio_ok and snd is not None:
        snd.stop()

    return True


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run_breathing_flash():
    # Late imports
    try:
        # pyrefly: ignore [missing-import]
        from psychopy import visual, core, event, sound, prefs, gui
        prefs.hardware['audioLib'] = ['ptb', 'sounddevice', 'pygame']
        prefs.hardware['audioDevice'] = ['Headphones (HBTS004)', 'default']
    except ImportError:
        print('[ERROR] PsychoPy is required. Install via:  pip install psychopy')
        sys.exit(1)

    # Prompt for flash frequency before initializing full screen window
    flash_hz = prompt_frequency(DEFAULT_FLASH_HZ)
    print(f'[CONFIG] Flashing frequency set to {flash_hz:.2f} Hz')

    init_serial()

    # ── Audio ──────────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    om_path  = OM_AUDIO  if os.path.isabs(OM_AUDIO)  else os.path.join(script_dir, OM_AUDIO)
    maa_path = MAA_AUDIO if os.path.isabs(MAA_AUDIO) else os.path.join(script_dir, MAA_AUDIO)

    audio_ok = False
    snd_om = snd_maa = None
    for label, path in [('OM', om_path), ('MAA', maa_path)]:
        if not os.path.exists(path):
            print(f'[AUDIO WARNING] {label} file not found: {path}')

    if os.path.exists(om_path) and os.path.exists(maa_path):
        try:
            snd_om  = sound.Sound(om_path,  secs=INHALE_DUR, stereo=True)
            snd_maa = sound.Sound(maa_path, secs=EXHALE_DUR, stereo=True)
            audio_ok = True
            print('[AUDIO] Sound objects loaded successfully.')
        except Exception as exc:
            print(f'[AUDIO WARNING] Could not load audio: {exc}')
    else:
        print('[AUDIO] Running without audio (one or both files missing).')

    # ── Beep Sound ─────────────────────────────────────────────────────────────
    beep_snd = None
    try:
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

    # ── Window ─────────────────────────────────────────────────────────────────
    win = visual.Window(
        fullscr=FULLSCREEN,
        color=BG_COLOR,
        units='norm',
        allowGUI=False,
        winType='pyglet',
        waitBlanking=True,
        useFBO=True,
    )
    win.mouseVisible = False
    clk = core.Clock()

    # Measure true refresh rate before anything is shown to the participant
    refresh_hz = measure_refresh_rate(win)

    # ── Full-screen flash rectangle ────────────────────────────────────────────
    # Colour will be updated per phase; starts as inhale colour
    rect = visual.Rect(
        win,
        units='norm',
        width=2.0, height=2.0,
        pos=(0, 0),
        fillColor=FLASH_COLOR_INHALE,
        lineColor=FLASH_COLOR_INHALE,
    )

    # ── Post-Breathing Visual stimuli ──────────────────────────────────────────
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

    # ── INSTRUCTION SCREEN stimuli ─────────────────────────────────────────────
    hold_line = (
        f"  HOLD BREATH ( {HOLD_DUR:.0f} s )  -->  screen holds colour\n"
        "  No audio during hold.\n\n"
    ) if HOLD_DUR > 0 else ''

    instruction_text = (
  
        "The screen will flash a colour to guide each\n"
        "breathing phase. No other visuals are shown.\n\n"
        f"  BREATHE IN  ( {INHALE_DUR:.0f} s )  -->  BLUE flash  /  'OM'\n\n"
        + hold_line +
        f"  BREATHE OUT ( {EXHALE_DUR:.0f} s )  -->  VIOLET flash  /  'MAA'\n\n"
        f"  Flash frequency: {flash_hz:.2f} Hz\n\n"
        f"  {NUM_CYCLES} breath cycle(s) total.\n\n"

        "Press  SPACE  to begin   |   ESC to quit"
    )

    instr_stim = visual.TextStim(
        win,
        text=instruction_text,
        height=0.07,           # norm units
        color=INSTRUCTION_COLOR,
        pos=(0, 0),
        wrapWidth=1.8,
        alignText='center',
        units='norm',
    )

    header_rect = visual.Rect(
        win,
        width=1.6, height=0.17,
        fillColor='#1a1a3a',
        lineColor='#3355bb',
        lineWidth=2,
        pos=(0, 0.88),
        units='norm',
    )
    header_text = visual.TextStim(
        win,
        text='Alpha Entrainment  |  Breathing Flash Module',
        height=0.06,
        bold=True,
        color='#88aaff',
        pos=(0, 0.88),
        units='norm',
    )

    # ── Show Instructions ──────────────────────────────────────────────────────
    event.clearEvents()
    waiting = True
    while waiting:
        if event.getKeys(['escape']):
            close_serial()
            win.close()
            core.quit()
        header_rect.draw()
        header_text.draw()
        instr_stim.draw()
        win.flip()
        if event.getKeys(['space']):
            waiting = False

    # Brief blank before starting
    win.color = BG_COLOR
    win.flip()
    core.wait(0.4)

    # ==========================================================================
    # BREATHING LOOP
    # ==========================================================================
    cycle       = 0
    total_cycles = NUM_CYCLES if NUM_CYCLES > 0 else float('inf')

    while cycle < total_cycles:
        cycle += 1
        print(f'[CYCLE {cycle}] Starting cycle {cycle} of {NUM_CYCLES}')

        # ── INHALE PHASE ───────────────────────────────────────────────────────
        # Trigger is sent via callOnFlip so it fires at the exact VSync moment
        # the first flicker frame appears on screen (hardware-synchronised).
        print(f'[INHALE] {INHALE_DUR}s at {flash_hz:.2f} Hz')
        rect.fillColor  = FLASH_COLOR_INHALE
        rect.lineColor  = FLASH_COLOR_INHALE
        ok = flash_phase(win, rect, flash_hz, INHALE_DUR,
                         refresh_hz, clk, event,
                         snd=snd_om, audio_ok=audio_ok,
                         on_first_flip=send_ttl)
        if not ok:
            if audio_ok:
                snd_om.stop()
                snd_maa.stop()
            close_serial()
            win.close()
            core.quit()

        # ── HOLD BREATH PHASE ──────────────────────────────────────────────────
        if HOLD_DUR > 0:
            print(f'[HOLD]   {HOLD_DUR}s at {flash_hz:.2f} Hz')
            rect.fillColor = FLASH_COLOR_HOLD
            rect.lineColor = FLASH_COLOR_HOLD
            ok = flash_phase(win, rect, flash_hz, HOLD_DUR,
                             refresh_hz, clk, event,
                             snd=None, audio_ok=False)  # no audio during hold
            if not ok:
                if audio_ok:
                    snd_om.stop()
                    snd_maa.stop()
                close_serial()
                win.close()
                core.quit()

        # ── EXHALE PHASE ───────────────────────────────────────────────────────
        print(f'[EXHALE] {EXHALE_DUR}s at {flash_hz:.2f} Hz')
        rect.fillColor  = FLASH_COLOR_EXHALE
        rect.lineColor  = FLASH_COLOR_EXHALE
        ok = flash_phase(win, rect, flash_hz, EXHALE_DUR,
                         refresh_hz, clk, event,
                         snd=snd_maa, audio_ok=audio_ok)
        if not ok:
            if audio_ok:
                snd_om.stop()
                snd_maa.stop()
            close_serial()
            win.close()
            core.quit()

    # ── POST-BREATHING PHASE ───────────────────────────────────────────────────
    # 1. Trigger when the last breathing cycle finishes
    print('[TTL] Breathing cycles complete. Sending trigger.')
    send_ttl()

    # 2. Fixation Screen (15s total, "Close Your Eyes" displayed at 15th second)
    print(f'[POST-BREATH] Fixation screen for {POST_BREATH_FIXATION_DUR}s (showing "Close Your Eyes" at 15th sec)...')
    fix_clk = core.Clock()
    while fix_clk.getTime() < POST_BREATH_FIXATION_DUR:
        if event.getKeys(['escape']):
            if audio_ok:
                snd_om.stop()
                snd_maa.stop()
            close_serial()
            win.close()
            core.quit()

        t = fix_clk.getTime()
        if t >= (POST_BREATH_FIXATION_DUR - 1.0):
            close_eyes_stim.draw()
        else:
            fixation_stim.draw()

        win.flip()

    # 3. Followed by Beep and transition to 1-minute Blank Screen
    print('[AUDIO] Playing "Close Your Eyes" beep...')
    play_beep()

    # Clear screen to black/background
    win.color = BG_COLOR
    win.flip()

    # Trigger sent when blank screen appears
    print(f'[TTL] Blank screen starting ({EYES_CLOSED_DUR}s eyes-closed). Sending trigger.')
    send_ttl()

    blank_clk = core.Clock()
    while blank_clk.getTime() < EYES_CLOSED_DUR:
        if event.getKeys(['escape']):
            if beep_snd is not None:
                beep_snd.stop()
            close_serial()
            win.close()
            core.quit()

        win.flip()

    # 4. End of 1 minute: Play beep sound, send trigger, and finish
    print('[AUDIO] 1-minute eyes-closed complete. Playing final beep...')
    play_beep()

    print('[TTL] Experiment finished. Sending trigger.')
    send_ttl()

    # Small delay to let the final beep finish playing
    core.wait(1.0)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    close_serial()
    win.close()
    core.quit()


if __name__ == '__main__':
    run_breathing_flash()
