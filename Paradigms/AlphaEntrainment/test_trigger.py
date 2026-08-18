"""
test_trigger.py
----------------
Simple test script for the ESP32 TTL trigger device.

Sends the trigger byte (0x01) over serial to the ESP32, which should
fire a TTL pulse on GPIO4 (D4) each time it's received.

Usage:
    python test_trigger.py

Adjust PORT below if your ESP32 doesn't enumerate as COM5
(check Device Manager > Ports (COM & LPT) after plugging it in).
"""

import serial
import time

PORT = "COM5"
BAUD_RATE = 115200
TRIGGER_BYTE = b'\x01'

def main():
    print(f"Connecting to {PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open {PORT}: {e}")
        return

    # Give the ESP32 a moment to reset/reboot after the port opens
    # (opening a serial connection can reset some boards)
    time.sleep(2)
    print("Connected.")

    try:
        num_triggers = int(input("How many test triggers to send? [default 5]: ") or 5)
        interval_s = float(input("Interval between triggers in seconds? [default 1.0]: ") or 1.0)
    except ValueError:
        print("Invalid input, using defaults (5 triggers, 1.0s interval).")
        num_triggers = 5
        interval_s = 1.0

    print(f"\nSending {num_triggers} trigger(s), {interval_s}s apart...\n")

    for i in range(1, num_triggers + 1):
        ser.write(TRIGGER_BYTE)
        print(f"[{i}/{num_triggers}] Sent trigger byte 0x01")
        time.sleep(interval_s)

    ser.close()
    print("\nDone. Serial port closed.")

if __name__ == "__main__":
    main()