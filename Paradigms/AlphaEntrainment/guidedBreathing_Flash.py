#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Guided Breathing + Full-Screen Flash Paradigm
===============================================
Same breathing phases as guided_breathing.py, but instead of an animated
circle the entire screen flashes a solid color in a square-wave pattern.

Phases (each with a distinct flash color):
  - Breathe IN   ( INHALE_DUR s )  ->  FLASH_COLOR_INHALE  flashes  /  'OM'  plays
  - Hold Breath  ( HOLD_DUR   s )  ->  FLASH_COLOR_HOLD    flashes  (no audio)
  - Breathe OUT  ( EXHALE_DUR s )  ->  FLASH_COLOR_EXHALE  flashes  /  'MAA' plays

Nothing is shown on-screen during the exercise except the flashing rectangle.
A full instruction page is displayed before the exercise begins.

Features
--------
1. Full-screen instruction page (SPACE to start, ESC to quit).
2. Frame-locked square-wave (ON/OFF) colour flashing -- same engine as white_flash.py.
3. Separate configurable flash colours and Hz for each phase.
4. External audio files (OM for inhale, MAA for exhale).
5. Set HOLD_DUR = 0 to skip the hold phase entirely.
6. Press ESC at any time to exit cleanly.
"""

import os
import sys

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
INHALE_DUR        = 4.0       # seconds for breathe-IN  phase
HOLD_DUR          = 0.0       # seconds for hold-breath phase (set 0 to skip)
EXHALE_DUR        = 6.0       # seconds for breathe-OUT phase
NUM_CYCLES        = 5         # number of complete breath cycles (0 = infinite)
FULLSCREEN        = True

# ── FLASH FREQUENCIES (Hz) ───────────────────────────────────────────────────
FLASH_HZ_INHALE   = 10.0      # flicker rate during inhale
FLASH_HZ_HOLD     = 10.0      # flicker rate during hold
FLASH_HZ_EXHALE   = 10.0      # flicker rate during exhale

# ── FLASH COLOURS ─────────────────────────────────────────────────────────────
FLASH_COLOR_INHALE  = '#5b8cff'   # cool blue   -- breath in
FLASH_COLOR_HOLD    = '#5b8cff'   # warm gold   -- hold breath
FLASH_COLOR_EXHALE  = '#5b8cff'   # soft violet -- breath out
BG_COLOR            = '#0d0d1a'   # background / OFF colour

# ── SCREEN REFRESH (fallback if measurement fails) ───────────────────────────
NOMINAL_REFRESH_HZ  = 120.0
MIN_PLAUSIBLE_HZ    = 30.0
MAX_PLAUSIBLE_HZ    = 300.0

# ── AUDIO FILES ───────────────────────────────────────────────────────────────
# Place your audio files in the same directory as this script (or provide
# absolute paths).  Supported formats: WAV, MP3, OGG, FLAC.
OM_AUDIO   = r'C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\Audio\Om.wav'    # played during the INHALE phase
MAA_AUDIO  = r'C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\Audio\MAA.wav'
# ── INSTRUCTION SCREEN COLOURS ────────────────────────────────────────────────
INSTRUCTION_COLOR = '#d0e8ff'
TEXT_COLOR        = 'white'


# ==============================================================================
# HELPERS
# ==============================================================================

def measure_refresh_rate(win, fallback_hz=NOMINAL_REFRESH_HZ):
    """
    Measure the monitor actual refresh rate. Rejects implausible readings
    (e.g. from a GPU driver that does not honour vsync) and falls back to the
    confirmed panel spec instead.
    """
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
                snd=None, audio_ok=False):
    """
    Run a frame-locked square-wave flash for `duration_s` seconds.
    The rectangle `rect` is drawn every ON frame; the window is cleared on
    OFF frames.  Audio `snd` is started at the beginning if provided.

    Returns True normally, False if ESC was pressed (caller should quit).
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
        from psychopy import visual, core, event, sound, prefs
        prefs.hardware['audioLib'] = ['ptb', 'sounddevice', 'pygame']
        prefs.hardware['audioDevice'] = ['Speakers (Realtek(R) Audio)', 'default']
    except ImportError:
        print('[ERROR] PsychoPy is required. Install via:  pip install psychopy')
        sys.exit(1)

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
        f"  Flash frequency: {FLASH_HZ_INHALE:.0f} Hz (inhale)  |  "
        f"{FLASH_HZ_HOLD:.0f} Hz (hold)  |  "
        f"{FLASH_HZ_EXHALE:.0f} Hz (exhale)\n\n"
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
        print(f'[INHALE] {INHALE_DUR}s at {FLASH_HZ_INHALE} Hz')
        rect.fillColor  = FLASH_COLOR_INHALE
        rect.lineColor  = FLASH_COLOR_INHALE
        ok = flash_phase(win, rect, FLASH_HZ_INHALE, INHALE_DUR,
                         refresh_hz, clk, event,
                         snd=snd_om, audio_ok=audio_ok)
        if not ok:
            if audio_ok:
                snd_om.stop()
                snd_maa.stop()
            win.close()
            core.quit()

        # ── HOLD BREATH PHASE ──────────────────────────────────────────────────
        if HOLD_DUR > 0:
            print(f'[HOLD]   {HOLD_DUR}s at {FLASH_HZ_HOLD} Hz')
            rect.fillColor = FLASH_COLOR_HOLD
            rect.lineColor = FLASH_COLOR_HOLD
            ok = flash_phase(win, rect, FLASH_HZ_HOLD, HOLD_DUR,
                             refresh_hz, clk, event,
                             snd=None, audio_ok=False)  # no audio during hold
            if not ok:
                if audio_ok:
                    snd_om.stop()
                    snd_maa.stop()
                win.close()
                core.quit()

        # ── EXHALE PHASE ───────────────────────────────────────────────────────
        print(f'[EXHALE] {EXHALE_DUR}s at {FLASH_HZ_EXHALE} Hz')
        rect.fillColor  = FLASH_COLOR_EXHALE
        rect.lineColor  = FLASH_COLOR_EXHALE
        ok = flash_phase(win, rect, FLASH_HZ_EXHALE, EXHALE_DUR,
                         refresh_hz, clk, event,
                         snd=snd_maa, audio_ok=audio_ok)
        if not ok:
            if audio_ok:
                snd_om.stop()
                snd_maa.stop()
            win.close()
            core.quit()

    # ── END SCREEN ─────────────────────────────────────────────────────────────
    win.color = BG_COLOR
    end_stim = visual.TextStim(
        win,
        text=(
            "Session Complete\n\n"
            "Well done.\n\n"
            "Take a moment to rest\n"
            "before continuing.\n\n"
            "(This window will close in 5 seconds)"
        ),
        height=0.08,
        color=TEXT_COLOR,
        pos=(0, 0),
        alignText='center',
        units='norm',
    )

    t_close = clk.getTime() + 5.0
    while clk.getTime() < t_close:
        if event.getKeys(['escape', 'space']):
            break
        end_stim.draw()
        win.flip()

    # ── Cleanup ────────────────────────────────────────────────────────────────
    win.close()
    core.quit()


if __name__ == '__main__':
    run_breathing_flash()
