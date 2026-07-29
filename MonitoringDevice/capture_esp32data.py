import csv
from datetime import datetime

import paho.mqtt.client as mqtt

BROKER = "10.108.129.32"      # Replace with your PC IP
PORT = 1883
TOPIC = "sensor/data1"

filename = f"EMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

csvfile = open(filename, "w", newline="")
writer = csv.writer(csvfile)

writer.writerow([
    "ESP_Time_ms",
    "EMG1",
    "EMG2",
    "PC_Time"
])


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connected")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):

    try:
        payload = msg.payload.decode().strip()

        t, ch1, ch2 = payload.split(",")

        writer.writerow([
            int(t),
            int(ch1),
            int(ch2),
            datetime.now().strftime("%H:%M:%S.%f")
        ])

        csvfile.flush()

        print(payload)

    except Exception as e:
        print(e)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)

client.loop_forever()