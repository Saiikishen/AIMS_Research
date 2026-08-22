#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time
# pyrefly: ignore [missing-import]
from psychopy import prefs
prefs.hardware['audioDevice'] = ['Headphones (HBTS004)', 'default']
# pyrefly: ignore [missing-import]
from psychopy import visual, core, event, sound

try:
    import serial, serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── HARDWARE & CONFIG ─────────────────────────────────────────────────────────
SERIAL_PORT   = 'COM5'
BAUD_RATE     = 115200
FULLSCREEN    = True
FALLBACK_SCREEN_SIZE = [1920, 1200]

# ── DISPLAY RESOLUTION ────────────────────────────────────────────────────────
def get_native_resolution(fallback=FALLBACK_SCREEN_SIZE):
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        if width > 0 and height > 0:
            return [int(width), int(height)]
    except Exception as e:
        print(f'[DISPLAY WARNING] Could not query native resolution ({e}); '
              f'using fallback {fallback}.')
    return list(fallback)

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

# ── HELPERS ───────────────────────────────────────────────────────────────────
def check_esc(win):
    if event.getKeys(['escape']):
        close_serial()
        win.close()
        core.quit()

# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────
def run_base_alpha():
    init_serial()

    screen_size = get_native_resolution()
    win = visual.Window(screen_size, fullscr=FULLSCREEN,
                        color='black', units='height', allowGUI=False,
                        waitBlanking=True)
    win.mouseVisible = False

    # Visual stimuli
    msg_open = visual.TextStim(win, text="Keep your eyes open till you hear a sound", height=0.05, color='white')
    fixation = visual.TextStim(win, text="+", height=0.08, color='white')
    msg_countdown = visual.TextStim(win, text="", height=0.08, color='white')

    # Audio setup using PsychoPy's sound engine
    beep = sound.Sound(value='C', octave=6, secs=0.6, volume=1.0)
    try:
        open_eyes_audio = sound.Sound('open_eyes.wav')
    except Exception:
        open_eyes_audio = sound.Sound(value='G', octave=5, secs=0.6, volume=1.0)

    def play_close_beep():
        beep.play()

    def play_open_audio():
        if not os.path.exists('open_eyes.wav'):
            print("[AUDIO WARNING] 'open_eyes.wav' not found. Using fallback tone.")
        open_eyes_audio.play()

    # ── 1. Show "Keep your eyes open till you hear a sound" for 5 sec ──
    msg_open.draw()
    win.flip()
    
    # TRIGGER 1: At the moment "Keep your eyes open..." is shown
    send_ttl() 
    
    clk = core.Clock()
    while clk.getTime() < 5.0:
        check_esc(win)
        msg_open.draw()
        win.flip()

    # ── 2. Black screen with fixation point for 65 sec, countdown starts at 62nd sec ──
    clk.reset()
    while clk.getTime() < 65.0:
        check_esc(win)
        
        t = clk.getTime()
        if 62.0 <= t < 63.0:
            msg_countdown.text = "Close your eyes in 3"
            msg_countdown.draw()
        elif 63.0 <= t < 64.0:
            msg_countdown.text = "Close your eyes in 2"
            msg_countdown.draw()
        elif 64.0 <= t < 65.0:
            msg_countdown.text = "Close your eyes in 1"
            msg_countdown.draw()
        else:
            fixation.draw()
            
        win.flip()

    # ── 3. At 65th sec, send beep and trigger ──
    win.flip() # Clear screen to black while eyes are closed
    play_close_beep()
    
    # TRIGGER 2: When the beep sound to close eyes is produced
    send_ttl() 

    # ── 4. Wait 65 seconds of eyes closed ──
    clk.reset()
    while clk.getTime() < 65.0:
        check_esc(win)
        win.flip()

    # ── 5. Play "Open your eyes" audio and send trigger ──
    play_open_audio()
    
    # TRIGGER 3: At the moment "Open your eyes" is played
    send_ttl() 

    # Small delay to let audio finish
    core.wait(2.0)

    # Cleanup
    close_serial()
    win.close()
    core.quit()

if __name__ == '__main__':
    run_base_alpha()
