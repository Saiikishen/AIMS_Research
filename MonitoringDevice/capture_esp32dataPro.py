import csv
import struct
import socket
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER = "10.108.129.32"
PORT = 1883
TOPIC = "sensor/data1"

HEADER_FORMAT = "<IIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
SAMPLE_FORMAT = "<hh"
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)

filename = f"EEG_ECG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csvfile = open(filename, "w", newline="", buffering=1)
writer = csv.writer(csvfile)
writer.writerow(["Epoch_Number", "ESP_Epoch_Start_ms", "Sample_Index", "EEG_ADC", "ECG_ADC", "PC_Time"])


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"on_connect called, reason_code={reason_code}")
    if reason_code == 0:
        client.subscribe(TOPIC)
        print(f"Subscribed to {TOPIC}")
    else:
        print(f"Connection refused by broker: {reason_code}")


def on_message(client, userdata, msg):
    payload = msg.payload
    if len(payload) < HEADER_SIZE:
        print(f"Discarded: packet too small ({len(payload)} bytes)")
        return
    epoch_number, epoch_start_ms, first_sample_index, sample_count = struct.unpack_from(HEADER_FORMAT, payload, 0)
    expected_size = HEADER_SIZE + (sample_count * SAMPLE_SIZE)
    if len(payload) != expected_size:
        print(f"Discarded invalid packet: {len(payload)} vs expected {expected_size}")
        return
    pc_time = datetime.now().isoformat(timespec="microseconds")
    for i in range(sample_count):
        offset = HEADER_SIZE + (i * SAMPLE_SIZE)
        eeg, ecg = struct.unpack_from(SAMPLE_FORMAT, payload, offset)
        writer.writerow([epoch_number, epoch_start_ms, first_sample_index + i, eeg, ecg, pc_time])
    csvfile.flush()
    print(f"Epoch {epoch_number} | {sample_count} samples received")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    print(f"Disconnected, reason_code={reason_code}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

try:
    print(f"Connecting to {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=60)
    print("Connect call succeeded, entering loop_forever()")
    client.loop_forever()

except (socket.timeout, ConnectionRefusedError, OSError) as e:
    print(f"Could NOT connect to broker {BROKER}:{PORT} -> {e}")
    print("Check: 1) broker IP correct, 2) Mosquitto/broker service running, 3) firewall allows port 1883, 4) same network as ESP32")

except KeyboardInterrupt:
    print("\nStopped by user.")

except Exception as e:
    import traceback
    print(f"Unexpected error: {e}")
    traceback.print_exc()

finally:
    csvfile.close()
    print(f"CSV saved: {filename}")
    input("Press Enter to close...")  # Keeps window open on Windows so you can read the error