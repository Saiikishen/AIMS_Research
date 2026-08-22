

import os
import sys
import random
import time
from datetime import datetime

import tkinter as tk
from tkinter import simpledialog, messagebox

import csv

# ---------------------------------------------------------------------------
# Configurable test parameters
# ---------------------------------------------------------------------------
TEST_DURATION = 40.0        # seconds, total test length
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
OBJECT_RADIUS = 25
FALL_DURATION = 3.0         # seconds for an object to fall top -> bottom
MIN_SPAWN_GAP = 0.5         # min seconds between object spawns
MAX_SPAWN_GAP = 0.9        # max seconds between object spawns
RED_PROBABILITY = 0.3       # chance any given spawned object is red (target)
FRAME_INTERVAL_MS = 50      # ~33 FPS movement update
HIT_TOLERANCE = 1.3         # multiplier on radius for a forgiving tap hitbox
GRACE_PERIOD_S = 0.10       # seconds after object leaves screen where a tap still counts
CSV_EVENTS_FILENAME = "falling_object_tap_test_events.csv"
CSV_SUMMARY_FILENAME = "falling_object_tap_test_summary.csv"

FALL_SPEED = CANVAS_HEIGHT / FALL_DURATION  # pixels per second


class FallingObjectTapTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Falling Object Tap Test")
        self.root.geometry(f"{CANVAS_WIDTH + 40}x{CANVAS_HEIGHT + 140}")
        self.root.resizable(False, False)
        self.root.withdraw()

        self.patient_id = self.ask_patient_id()
        if not self.patient_id:
            self.root.destroy()
            sys.exit(0)

        self.root.deiconify()

        # Test state
        self.active_objects = {}   # canvas_item_id -> {"color", "spawn_time"}
        self.recently_expired = {}  # grace buffer: {"color", "spawn_time", "expired_time", "cx", "cy"}
        self.events = []           # list of per-object outcome dicts
        self.total_red = 0
        self.total_black = 0
        self.hits = 0
        self.false_alarms = 0
        self.empty_taps = 0
        self.test_start_time = None
        self.next_spawn_time = 0.0
        self.running = False
        self.csv_path = None

        self.build_ui()
        self.root.bind("<space>", self.on_space_pressed)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def ask_patient_id(self):
        return simpledialog.askstring("Patient ID", "Enter Patient ID:", parent=self.root)

    def build_ui(self):
        tk.Label(self.root, text=f"Patient ID: {self.patient_id}", font=("Arial", 12)).pack(pady=6)

        self.instruction_label = tk.Label(
            self.root,
            text="TAP the RED circles only - ignore the black ones",
            font=("Arial", 12, "bold"),
            fg="darkred",
        )
        self.instruction_label.pack(pady=4)

        self.status_label = tk.Label(self.root, text="Press SPACE to start", font=("Arial", 11))
        self.status_label.pack(pady=2)

        self.canvas = tk.Canvas(
            self.root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", highlightthickness=1,
            highlightbackground="gray",
        )
        self.canvas.pack(pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_click)

    # ------------------------------------------------------------------
    # Test flow
    # ------------------------------------------------------------------
    def on_space_pressed(self, event):
        if not self.running and self.test_start_time is None:
            self.root.unbind("<space>")
            self.start_test()

    def start_test(self):
        self.running = True
        self.test_start_time = time.time()
        self.next_spawn_time = 0.0
        self.status_label.config(text=f"Time remaining: {int(TEST_DURATION)}s | Hits: 0 | False alarms: 0")
        self.update_loop()

    def update_loop(self):
        if not self.running:
            return

        elapsed = time.time() - self.test_start_time
        if elapsed >= TEST_DURATION:
            self.end_test()
            return

        # Spawn a new object if it's time
        if elapsed >= self.next_spawn_time:
            self.spawn_object()
            self.next_spawn_time = elapsed + random.uniform(MIN_SPAWN_GAP, MAX_SPAWN_GAP)

        # Move every active object down; remove + score any that fell off-screen
        dy = FALL_SPEED * (FRAME_INTERVAL_MS / 1000.0)
        current_time = time.time()
        for item_id in list(self.active_objects.keys()):
            self.canvas.move(item_id, 0, dy)
            coords = self.canvas.coords(item_id)
            if not coords:
                continue
            x0, y0, x1, y1 = coords
            if y0 > CANVAS_HEIGHT:
                obj = self.active_objects.pop(item_id)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                self.canvas.delete(item_id)
                grace_key = id(obj)
                self.recently_expired[grace_key] = {
                    "color": obj["color"],
                    "spawn_time": obj["spawn_time"],
                    "expired_time": current_time,
                    "cx": cx,
                    "cy": cy,
                }

        # Clean up grace period buffer
        for grace_key in list(self.recently_expired.keys()):
            entry = self.recently_expired[grace_key]
            if current_time - entry["expired_time"] >= GRACE_PERIOD_S:
                self.recently_expired.pop(grace_key)
                if entry["color"] == "red":
                    self.log_event(entry, status="MISSED", reaction_time=None)

        self.status_label.config(
            text=f"Time remaining: {int(TEST_DURATION - elapsed)}s | "
            f"Hits: {self.hits} | False alarms: {self.false_alarms}"
        )

        self.root.after(FRAME_INTERVAL_MS, self.update_loop)

    def spawn_object(self):
        x = random.randint(OBJECT_RADIUS, CANVAS_WIDTH - OBJECT_RADIUS)
        color = "red" if random.random() < RED_PROBABILITY else "black"
        item_id = self.canvas.create_oval(
            x - OBJECT_RADIUS, -2 * OBJECT_RADIUS, x + OBJECT_RADIUS, 0,
            fill=color, outline="",
        )
        self.active_objects[item_id] = {"color": color, "spawn_time": time.time()}
        if color == "red":
            self.total_red += 1
        else:
            self.total_black += 1

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def on_canvas_click(self, event):
        if not self.running:
            self.empty_taps += 1
            return

        hit_tolerance_px = OBJECT_RADIUS * HIT_TOLERANCE
        best_source = None   # "active" or "grace"
        best_key = None
        best_dist = float('inf')

        # 1) Check all currently visible (active) objects
        for item_id, obj_data in self.active_objects.items():
            coords = self.canvas.coords(item_id)
            if not coords:
                continue
            x0, y0, x1, y1 = coords
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            dist = ((event.x - cx) ** 2 + (event.y - cy) ** 2) ** 0.5
            if dist <= hit_tolerance_px and dist < best_dist:
                best_dist = dist
                best_key = item_id
                best_source = "active"

        # 2) Check recently expired objects (grace period buffer)
        for grace_key, entry in self.recently_expired.items():
            cx, cy = entry["cx"], entry["cy"]
            dist = ((event.x - cx) ** 2 + (event.y - cy) ** 2) ** 0.5
            if dist <= hit_tolerance_px and dist < best_dist:
                best_dist = dist
                best_key = grace_key
                best_source = "grace"

        if best_source is None:
            self.empty_taps += 1
            return

        # Retrieve and remove the object from whichever pool it was in
        if best_source == "active":
            obj = self.active_objects.pop(best_key)
            self.canvas.delete(best_key)
        else:
            obj = self.recently_expired.pop(best_key)

        if obj["color"] == "red":
            reaction_time = round(time.time() - obj["spawn_time"], 4)
            self.log_event(obj, status="HIT", reaction_time=reaction_time)
            self.hits += 1
        else:
            self.log_event(obj, status="FALSE ALARM", reaction_time=None)
            self.false_alarms += 1

    def log_event(self, obj, status, reaction_time):
        self.events.append(
            {
                "event_num": len(self.events) + 1,
                "color": obj["color"],
                "status": status,
                "reaction_time": reaction_time,
            }
        )

    # ------------------------------------------------------------------
    # Wrap-up
    # ------------------------------------------------------------------
    def end_test(self):
        self.running = False

        # Resolve any objects still on screen when time ran out
        for item_id, obj in list(self.active_objects.items()):
            self.canvas.delete(item_id)
            if obj["color"] == "red":
                self.log_event(obj, status="MISSED", reaction_time=None)
        self.active_objects.clear()

        # Resolve any objects in the grace period buffer
        for grace_key, entry in list(self.recently_expired.items()):
            if entry["color"] == "red":
                self.log_event(entry, status="MISSED", reaction_time=None)
        self.recently_expired.clear()

        missed_red = sum(1 for e in self.events if e["status"] == "MISSED")
        correct_rejections = self.total_black - self.false_alarms

        hit_rate = (self.hits / self.total_red * 100) if self.total_red else 0.0
        false_alarm_rate = (self.false_alarms / self.total_black * 100) if self.total_black else 0.0
        denom = self.total_red + self.total_black
        overall_accuracy = ((self.hits + correct_rejections) / denom * 100) if denom else 0.0

        latencies = [e["reaction_time"] for e in self.events if e["status"] == "HIT"]
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        self.save_to_csv(
            hit_rate=hit_rate,
            false_alarm_rate=false_alarm_rate,
            overall_accuracy=overall_accuracy,
            avg_latency=avg_latency,
            missed_red=missed_red,
            correct_rejections=correct_rejections,
        )

        summary_lines = [
            f"Patient ID: {self.patient_id}",
            f"Red targets: {self.total_red}  |  Black distractors: {self.total_black}",
            f"Hits: {self.hits}  |  Missed: {missed_red}  |  False alarms: {self.false_alarms}",
            f"Hit rate: {hit_rate:.1f}%",
            f"Overall accuracy: {overall_accuracy:.1f}%",
        ]
        if avg_latency is not None:
            summary_lines.append(f"Average latency: {avg_latency:.3f} s")
        else:
            summary_lines.append("Average latency: N/A (no hits)")
        summary_lines.append(f"\nSaved to:\n{self.csv_path}")

        self.status_label.config(text="Test Complete!")
        messagebox.showinfo("Test Complete", "\n".join(summary_lines))
        self.root.after(200, self.root.destroy)

    def save_to_csv(self, hit_rate, false_alarm_rate, overall_accuracy, avg_latency,
                       missed_red, correct_rejections):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        events_filepath = os.path.join(script_dir, CSV_EVENTS_FILENAME)
        summary_filepath = os.path.join(script_dir, CSV_SUMMARY_FILENAME)
        self.csv_path = f"{CSV_EVENTS_FILENAME} and {CSV_SUMMARY_FILENAME}"

        events_headers = ["Patient ID", "Test Timestamp", "Event #", "Object Color", "Status", "Reaction Time (s)"]
        summary_headers = [
            "Patient ID", "Test Timestamp", "Total Red", "Total Black", "Hits", "Missed Red",
            "False Alarms", "Correct Rejections", "Hit Rate (%)", "Overall Accuracy (%)",
            "False Alarm Rate (%)", "Avg Latency (s)",
        ]

        test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save Events
        events_exists = os.path.exists(events_filepath)
        with open(events_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not events_exists:
                writer.writerow(events_headers)
            for e in self.events:
                writer.writerow([
                    self.patient_id,
                    test_timestamp,
                    e["event_num"],
                    e["color"],
                    e["status"],
                    e["reaction_time"] if e["reaction_time"] is not None else "N/A",
                ])

        # Save Summary
        summary_exists = os.path.exists(summary_filepath)
        with open(summary_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not summary_exists:
                writer.writerow(summary_headers)
            writer.writerow([
                self.patient_id,
                test_timestamp,
                self.total_red,
                self.total_black,
                self.hits,
                missed_red,
                self.false_alarms,
                correct_rejections,
                round(hit_rate, 2),
                round(overall_accuracy, 2),
                round(false_alarm_rate, 2),
                round(avg_latency, 4) if avg_latency is not None else "N/A",
            ])


if __name__ == "__main__":
    root = tk.Tk()
    app = FallingObjectTapTest(root)
    root.mainloop()