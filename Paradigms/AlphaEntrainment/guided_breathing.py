#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Guided Breathing Paradigm
==========================
Guides the participant through a breathing exercise:
  - Breathe IN   for INHALE_DUR seconds  -> "OM"  mantra tone plays
  - Hold Breath  for HOLD_DUR   seconds  -> circle holds at max size
  - Breathe OUT  for EXHALE_DUR seconds  -> "MAA" mantra tone plays

Features
--------
1. Full-screen instruction page before the exercise begins.
2. Animated expanding/contracting circle as a visual breathing guide.
3. External audio files for OM (inhale) and MAA (exhale) mantras.
4. On-screen countdown timer for each phase.
5. Configurable number of breathing cycles and all phase durations.
6. Set HOLD_DUR = 0 to skip the hold phase entirely.
7. Press ESC at any time to exit cleanly.
"""

import os
import sys
import math

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
INHALE_DUR        = 3.0       # seconds for breathe-IN  phase
HOLD_DUR          = 2.0       # seconds for hold-breath phase (set 0 to skip)
EXHALE_DUR        = 9.0       # seconds for breathe-OUT phase
NUM_CYCLES        = 5         # number of complete breath cycles (0 = infinite)
FULLSCREEN        = True

# ── AUDIO FILES ───────────────────────────────────────────────────────────────
# Place your audio files in the same directory as this script (or provide
# absolute paths).  Supported formats: WAV, MP3, OGG, FLAC.
OM_AUDIO   = r'C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\Audio\Om.wav'    # played during the INHALE phase
MAA_AUDIO  = r'C:\Users\saiik\OneDrive\Documents\GitHub\AIMS_Research\Paradigms\AlphaEntrainment\Audio\MAA.wav'   # played during the EXHALE phase

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
BG_COLOR          = '#0d0d1a'   # deep navy background
CIRCLE_INHALE     = '#5b8cff'   # cool blue   -- breath in
CIRCLE_HOLD       = '#f0c040'   # warm gold   -- hold breath
CIRCLE_EXHALE     = '#a05ddc'   # soft violet -- breath out
TEXT_COLOR        = 'white'
TIMER_COLOR       = '#c8d8ff'
INSTRUCTION_COLOR = '#d0e8ff'

# ── CIRCLE SIZE ───────────────────────────────────────────────────────────────
CIRCLE_MIN_RADIUS = 0.10       # height-units, smallest (fully exhaled)
CIRCLE_MAX_RADIUS = 0.28       # height-units, largest  (fully inhaled)


# ==============================================================================
# PSYCHOPY VISUAL EXPERIMENT
# ==============================================================================

def run_breathing():
    # Late imports (psychopy)
    try:
        from psychopy import visual, core, event, sound, prefs
        prefs.hardware['audioLib'] = ['pygame', 'sounddevice', 'ptb']
    except ImportError:
        print('[ERROR] PsychoPy is required. Install via:  pip install psychopy')
        sys.exit(1)

    # Resolve audio file paths (support relative paths next to this script)
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

    # ── Window ────────────────────────────────────────────────────────────────
    win = visual.Window(
        fullscr=FULLSCREEN,
        color=BG_COLOR,
        units='height',
        allowGUI=False,
        winType='pyglet',
        waitBlanking=True,
    )
    win.mouseVisible = False
    clk = core.Clock()

    # ── Visual Stimuli ────────────────────────────────────────────────────────
    # Breathing circle -- GratingStim(mask='circle') renders via OpenGL texture
    # and never calls GLU tessellation, avoiding ctypes callback crashes.
    circle = visual.GratingStim(
        win,
        mask='circle',
        sf=0,
        color=CIRCLE_INHALE,
        size=CIRCLE_MIN_RADIUS * 2,
        pos=(0, 0.06),
    )

    # Glow halo -- larger disk at low opacity drawn behind the main circle
    glow = visual.GratingStim(
        win,
        mask='circle',
        sf=0,
        color=CIRCLE_INHALE,
        size=(CIRCLE_MIN_RADIUS + 0.012) * 2,
        opacity=0.35,
        pos=(0, 0.06),
    )

    phase_label = visual.TextStim(
        win,
        text='',
        height=0.055,
        bold=True,
        color=TEXT_COLOR,
        pos=(0, -0.23),
    )

    timer_label = visual.TextStim(
        win,
        text='',
        height=0.038,
        color=TIMER_COLOR,
        pos=(0, -0.32),
    )

    mantra_label = visual.TextStim(
        win,
        text='',
        height=0.042,
        italic=True,
        color='#ffd580',
        pos=(0, 0.06),
    )

    cycle_label = visual.TextStim(
        win,
        text='',
        height=0.028,
        color='#8899bb',
        pos=(0, 0.44),
    )

    # ── INSTRUCTION SCREEN ────────────────────────────────────────────────────
    hold_line = (
        f"  HOLD BREATH ( {HOLD_DUR:.0f} s )  -->  circle holds\n"
        "  The circle stays full while you hold.\n\n"
    ) if HOLD_DUR > 0 else ''

    instruction_text = (
        "This exercise guides you through slow,\n"
        "mindful breathing with mantra sound.\n\n"
        f"  BREATHE IN  ( {INHALE_DUR:.0f} s ) :  'OM'\n"
        "  The circle expands as you inhale.\n\n"
        + hold_line +
        f"  BREATHE OUT ( {EXHALE_DUR:.0f} s ) : 'MAA'\n"
        "  The circle contracts as you exhale.\n\n"
        f"Complete {NUM_CYCLES} breath cycle(s).\n\n"
        "Press  SPACE  to begin   |   ESC to quit"
    )

    instr_stim = visual.TextStim(
        win,
        text=instruction_text,
        height=0.033,
        color=INSTRUCTION_COLOR,
        pos=(0, 0),
        wrapWidth=1.2,
        alignText='center',
    )

    # Decorative header bar
    header_rect = visual.Rect(
        win,
        width=0.85,
        height=0.09,
        fillColor='#1a1a3a',
        lineColor='#3355bb',
        lineWidth=2,
        pos=(0, 0.44),
    )
    header_text = visual.TextStim(
        win,
        text='Breathing Module',
        height=0.030,
        bold=True,
        color='#88aaff',
        pos=(0, 0.44),
    )

    # ── Show Instructions ─────────────────────────────────────────────────────
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

    # Brief black flash before start
    win.color = BG_COLOR
    win.flip()
    core.wait(0.4)

    # ==========================================================================
    # BREATHING LOOP
    # ==========================================================================
    cycle = 0
    total_cycles = NUM_CYCLES if NUM_CYCLES > 0 else float('inf')

    while cycle < total_cycles:
        cycle += 1
        cycle_str = f'Cycle {cycle} / {NUM_CYCLES}' if NUM_CYCLES > 0 else f'Cycle {cycle}'

        # ── INHALE PHASE ──────────────────────────────────────────────────────
        phase_label.text  = '- Breathe IN -'
        mantra_label.text = 'OM'
        circle.color  = CIRCLE_INHALE
        glow.color    = CIRCLE_INHALE
        cycle_label.text  = cycle_str

        if audio_ok:
            snd_om.stop()
            snd_om.play()

        t_phase = clk.getTime()
        t_end   = t_phase + INHALE_DUR

        while clk.getTime() < t_end:
            if event.getKeys(['escape']):
                if audio_ok:
                    snd_om.stop()
                    snd_maa.stop()
                win.close()
                core.quit()

            elapsed  = clk.getTime() - t_phase
            progress = min(elapsed / INHALE_DUR, 1.0)

            # Smooth easing: ease-in-out cubic
            eased  = progress * progress * (3 - 2 * progress)
            radius = CIRCLE_MIN_RADIUS + eased * (CIRCLE_MAX_RADIUS - CIRCLE_MIN_RADIUS)

            # Subtle glow pulse
            glow_opacity = 0.20 + 0.25 * math.sin(math.pi * progress)

            circle.size = radius * 2
            glow.size   = (radius + 0.018) * 2
            glow.opacity  = glow_opacity

            remaining = max(0.0, t_end - clk.getTime())
            timer_label.text = f'{remaining:.1f} s'

            cycle_label.draw()
            glow.draw()
            circle.draw()
            mantra_label.draw()
            phase_label.draw()
            timer_label.draw()
            win.flip()

        if audio_ok:
            snd_om.stop()

        # ── HOLD BREATH PHASE ─────────────────────────────────────────────────
        if HOLD_DUR > 0:
            phase_label.text  = '- Hold Breath -'
            mantra_label.text = ''
            circle.color  = CIRCLE_HOLD
            glow.color    = CIRCLE_HOLD
            # Circle stays at max size during hold
            circle.size = CIRCLE_MAX_RADIUS * 2
            glow.size   = (CIRCLE_MAX_RADIUS + 0.018) * 2

            t_phase = clk.getTime()
            t_end   = t_phase + HOLD_DUR

            while clk.getTime() < t_end:
                if event.getKeys(['escape']):
                    if audio_ok:
                        snd_om.stop()
                        snd_maa.stop()
                    win.close()
                    core.quit()

                elapsed  = clk.getTime() - t_phase
                progress = min(elapsed / HOLD_DUR, 1.0)

                # Slow gentle pulsing glow during hold
                glow_opacity = 0.25 + 0.20 * math.sin(2 * math.pi * progress * 1.5)

                glow.opacity  = glow_opacity

                remaining = max(0.0, t_end - clk.getTime())
                timer_label.text = f'{remaining:.1f} s'

                cycle_label.draw()
                glow.draw()
                circle.draw()
                phase_label.draw()
                timer_label.draw()
                win.flip()


        phase_label.text  = '- Breathe OUT -'
        mantra_label.text = 'MAA'
        circle.color  = CIRCLE_EXHALE
        glow.color    = CIRCLE_EXHALE

        if audio_ok:
            snd_maa.stop()
            snd_maa.play()

        t_phase = clk.getTime()
        t_end   = t_phase + EXHALE_DUR

        while clk.getTime() < t_end:
            if event.getKeys(['escape']):
                if audio_ok:
                    snd_om.stop()
                    snd_maa.stop()
                win.close()
                core.quit()

            elapsed  = clk.getTime() - t_phase
            progress = min(elapsed / EXHALE_DUR, 1.0)

            # Ease-in-out: circle contracts smoothly
            eased  = progress * progress * (3 - 2 * progress)
            radius = CIRCLE_MAX_RADIUS - eased * (CIRCLE_MAX_RADIUS - CIRCLE_MIN_RADIUS)

            glow_opacity = 0.30 * (1.0 - progress) + 0.08

            circle.size = radius * 2
            glow.size   = (radius + 0.018) * 2
            glow.opacity  = glow_opacity

            remaining = max(0.0, t_end - clk.getTime())
            timer_label.text = f'{remaining:.1f} s'

            cycle_label.draw()
            glow.draw()
            circle.draw()
            mantra_label.draw()
            phase_label.draw()
            timer_label.draw()
            win.flip()

        if audio_ok:
            snd_maa.stop()

    # ── END SCREEN ────────────────────────────────────────────────────────────
    end_stim = visual.TextStim(
        win,
        text=(
            "Session Complete\n\n"
            "Well done.\n\n"
            "Take a moment to rest\n"
            "before continuing.\n\n"
            "(This window will close in 5 seconds)"
        ),
        height=0.040,
        color=TEXT_COLOR,
        pos=(0, 0),
        alignText='center',
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
    run_breathing()
