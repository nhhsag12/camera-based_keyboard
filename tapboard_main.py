import statistics
import time
import cv2
import json
import tkinter as tk
from tkinter import scrolledtext
import threading
from pynput.keyboard import Controller, Key
from typing import Set
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
import sys # Import sys

# Assuming camera_manager, hand_tracker, keyboard_manager, and visualization_utils are available
from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils

FINGERTIPS = [
    "index",
    "middle",
    "ring",
    "pinky",
    "thumb"
]

class TapboardX:
    """Main application class for the virtual keyboard interface."""

    # Constants
    KEY_MAP = {
        "BACKSPACE": Key.backspace,
        "ENTER": Key.enter,
        "SPACE": Key.space,
        "SHIFT": Key.shift,
        "CTRL": Key.ctrl,
        "ALT": Key.alt,
        "WIN": Key.cmd,
        "ESC": Key.esc,
        "DEL": Key.delete,
        "UP": Key.up,
        "DOWN": Key.down,
        "LEFT": Key.left,
        "RIGHT": Key.right,
        "TAB": Key.tab,
        "CAPS": Key.caps_lock,
    }

    def __init__(self, name="hoang", frame_per_second=30):
        # Configuration
        self.annotation_filename = 'assets/keyboard_annotations.json'
        self.thresholds_filename = 'assets/key_thresholds.json'
        self.analysis_filename = f"assets/data/analysis_data_{name}_{time.strftime('%Y%m%d_%H%M%S')}_{frame_per_second}.json"
        self.log_filename = f"assets/data/log_{name}_{time.strftime('%Y%m%d_%H%M%S')}_{frame_per_second}.log" # Log file
        self.touched_frames_count = 0
        self.touched_frames_count_history = []
        self.points_per_key = 4
        self.release_threshold = 0.290
        self.frame_queue_size = 2
        self.queue_frame = [(0.0, False, 0.0, None) for i in range(self.frame_queue_size)] # (depth, is_active_finger, time, pressed_key)
        self.fingers_depth_frames_history = {"Right": {k: deepcopy(self.queue_frame) for k in FINGERTIPS},
                                        "Left": {k: deepcopy(self.queue_frame) for k in FINGERTIPS}}
        self.num_active_finger_frames_history = {
            "Right": {k: 0 for k in FINGERTIPS},
            "Left": {k: 0 for k in FINGERTIPS}
        }
        self.release_velocity_history = []
        self.press_velocity_history = []
        self.release_velocity_threshold = -0.05
        self.press_velocity_threshold = 0.1
        self.start_press_time = 0.0
        self.active_time_limit = 0.45


        # State variables
        self.key_depth_thresholds = {}
        self.active_finger = None
        self.active_key = None
        self.active_hand = None
        self.last_pressed_keys = set()
        self.start_active_time = 0


        # Components
        self.keyboard = Controller()
        self.camera_manager = CameraManager(fps=frame_per_second)
        self.hand_tracker = HandTracker()
        self.keyboard_manager = KeyboardManager(
            annotation_filename=self.annotation_filename,
            points_per_key=self.points_per_key
        )

        # UI thread
        self.ui_thread = None

        # Log file setup
        self.log_file = None
        # self._open_log_file()


    def _open_log_file(self):
        """Opens the log file and redirects stdout."""
        try:
            self.log_file = open(self.log_filename, 'a')
            sys.stdout = self.log_file
            print(f"Logging started. Output redirected to '{self.log_filename}'.")
        except Exception as e:
            print(f"Error opening log file: {e}")

    def _close_log_file(self):
        """Closes the log file and restores stdout."""
        if self.log_file:
            print(f"Logging ended. Output restored to console.")
            self.log_file.close()
            sys.stdout = sys.__stdout__ # Restore stdout

    def load_key_thresholds(self) -> bool:
        """Load key depth thresholds from file."""
        try:
            with open(self.thresholds_filename, 'r') as f:
                data = json.load(f)
                self.key_depth_thresholds = {key: tuple(value) for key, value in data.items()}
            print(f"Successfully loaded key thresholds from '{self.thresholds_filename}'.")
            return True
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return False

    def is_finger_pressing_key(self, key_data: dict, finger_depth: float) -> bool:
        """Check if finger is pressing a key based on depth threshold."""
        key_name = key_data.get("key")
        if not key_name:
            return False

        threshold = self.key_depth_thresholds.get(key_name)
        if not threshold:
            return False

        min_depth, max_depth = threshold
        return min_depth <= finger_depth

    def has_active_finger(self) -> bool:
        """Check if there's an active finger currently pressing a key."""
        return (self.active_hand is not None and
                self.active_finger is not None and
                self.active_key is not None)

    def reset_active_finger(self):
        """Reset the active finger state."""
        self.active_finger = None
        self.active_key = None
        self.active_hand = None

    def set_active_finger(self, hand_idx: int, finger_name: str, key_name: str):
        """Set the active finger state."""
        self.active_hand = hand_idx
        self.active_finger = finger_name
        self.active_key = key_name

    def _calculate_velocity(self, previous_finger_frame_state, current_finger_frame_state):
        distance = current_finger_frame_state[0] - previous_finger_frame_state[0]
        time_diff = current_finger_frame_state[2] - previous_finger_frame_state[2]
        # print(f"Distance: {distance}") # These will also go to log if uncommented
        print(f"Time diff: {time_diff}") # These will also go to log if uncommented
        if time_diff == 0:
            print(previous_finger_frame_state)
            print(current_finger_frame_state)
        return distance/0.033

    def _plot_release_velocity_history(self):
        data_to_plot = self.release_velocity_history

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generate the violin plot
        violin_parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True, showextrema=True)

        # Customize the lines (means, medians, extrema)
        for part in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if part in violin_parts:
                violin_parts[part].set_edgecolor("black")
                violin_parts[part].set_linewidth(1.5)

        # Set labels and title
        ax.set_title("Violin Plot of Release Velocity Distribution", fontsize=16)
        ax.set_ylabel("Velocity (m/s)", fontsize=12)

        ax.set_xticks([1])
        ax.set_xticklabels(["Velocity Distribution"], fontsize=10)

        # Add a grid for better readability
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        # Adjust the layout to prevent labels from overlapping
        plt.tight_layout()

        # Log statistical information
        print("-" * 10)
        print("Release Velocity Statistics:")
        print(f"The minimum release velocity is: {min(self.release_velocity_history)}")
        print(f"The maximum release velocity is: {max(self.release_velocity_history)}")
        print(f"The average release velocity is: {sum(self.release_velocity_history)/len(self.release_velocity_history)}")
        print(f"The median release velocity is: {statistics.median(self.release_velocity_history)}")
        print(f"The standard deviation of release velocity is: {statistics.stdev(self.release_velocity_history)}")
        print(f"The variance of release velocity is: {statistics.variance(self.release_velocity_history)}")
        print("-" * 10)

        # Show the plot
        plt.show()


    def _plot_press_velocity_history(self):
        data_to_plot = self.press_velocity_history

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generate the violin plot
        violin_parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True, showextrema=True)

        pc = violin_parts['bodies'][0]
        pc.set_facecolor("salmon")

        # Customize the lines (means, medians, extrema)
        for part in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if part in violin_parts:
                violin_parts[part].set_edgecolor("black")
                violin_parts[part].set_linewidth(1.5)

        # Set labels and title
        ax.set_title("Violin Plot of Press Velocity Distribution", fontsize=16)
        ax.set_ylabel("Velocity (m/s)", fontsize=12)

        ax.set_xticks([1])
        ax.set_xticklabels(["Velocity Distribution"], fontsize=10)

        # Add a grid for better readability
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        # Adjust layout to prevent labels from overlapping
        plt.tight_layout()

        # Log statistical information
        print("-" * 10)
        print("Press Velocity Statistics:")
        print(f"The minimum press velocity is: {min(self.press_velocity_history)}")
        print(f"The maximum press velocity is: {max(self.press_velocity_history)}")
        print(f"The average press velocity is: {sum(self.press_velocity_history)/len(self.press_velocity_history)}")
        print(f"The median press velocity is: {statistics.median(self.press_velocity_history)}")
        print(f"The standard deviation of press velocity is: {statistics.stdev(self.press_velocity_history)}")
        print(f"The variance of press velocity is: {statistics.variance(self.press_velocity_history)}")
        print("-" * 10)

        # Show the plot
        plt.show()

    def _plot_touched_frames_distribution(self):
        frequency_dict = Counter(self.touched_frames_count_history)
        number_of_touched_key = list(frequency_dict.values()) # Convert to list for plotting
        number_of_frames = list(frequency_dict.keys()) # Convert to list for plotting

        plt.bar(number_of_frames, number_of_touched_key)
        plt.xlabel("Number of frames in touched state")
        plt.ylabel("Number of touched keys")
        plt.title("The distribution of touched frames per key")
        plt.show()

        with open(self.analysis_filename, 'w') as f:
            json.dump(frequency_dict, f, indent=4)
            print(f"Saved analysis data to '{self.analysis_filename}'.")

        # Log statistical information for touched frames
        print("-" * 10)
        print("Touched Frames Distribution Statistics:")
        if self.touched_frames_count_history:
            print(f"Total touched key instances: {len(self.touched_frames_count_history)}")
            print(f"Min frames in touched state: {min(self.touched_frames_count_history)}")
            print(f"Max frames in touched state: {max(self.touched_frames_count_history)}")
            print(f"Average frames in touched state: {sum(self.touched_frames_count_history)/len(self.touched_frames_count_history)}")
            print(f"Median frames in touched state: {statistics.median(self.touched_frames_count_history)}")
            print(f"Standard deviation of frames in touched state: {statistics.stdev(self.touched_frames_count_history)}")
            print(f"Variance of frames in touched state: {statistics.variance(self.touched_frames_count_history)}")
        else:
            print("No touched frame data to analyze.")
        print("-" * 10)
        pass


    def process_finger_tips(self, hand_landmarks, hand_idx: int, color_image,
                            aligned_depth_frame, depth_frame_dims, hand_label) -> Set[str]:
        """Process finger tips for a single hand and return pressed keys."""
        current_pressed_keys = set()

        finger_tips = {
            'thumb': self.hand_tracker.get_thumb_finger_tip(hand_landmarks, color_image.shape),
            'index': self.hand_tracker.get_index_finger_tip(hand_landmarks, color_image.shape),
            'middle': self.hand_tracker.get_middle_finger_tip(hand_landmarks, color_image.shape),
            'ring': self.hand_tracker.get_ring_finger_tip(hand_landmarks, color_image.shape),
            'pinky': self.hand_tracker.get_pinky_finger_tip(hand_landmarks, color_image.shape),
        }

        for finger_name, (tip_coords, finger_depth) in finger_tips.items():
            if not tip_coords:
                continue
            px, py = tip_coords
            depth_frame_width, depth_frame_height = depth_frame_dims
            clamped_px = max(0, min(px, depth_frame_width - 1))
            clamped_py = max(0, min(py, depth_frame_height - 1))
            depth_m = aligned_depth_frame.get_distance(clamped_px, clamped_py)

            # Update the depth history of each fingertip in each frame
            pop_finger = self.fingers_depth_frames_history[hand_label][finger_name].pop(0)
            if pop_finger[1]: # If this finger is the active finger
                self.num_active_finger_frames_history[hand_label][finger_name] = max(0, self.num_active_finger_frames_history[hand_label][finger_name]-1)
                if self.num_active_finger_frames_history[hand_label][finger_name] == 0:
                    touched_finger_frame_state = pop_finger
                    velo_history = []
                    for finger_frame_in_state in self.fingers_depth_frames_history[hand_label][finger_name]:
                        if not finger_frame_in_state[1]: # The finger in released state
                            velocity = self._calculate_velocity(touched_finger_frame_state, finger_frame_in_state)
                            velo_history.append(velocity)
                    min_velocity = min(velo_history)
                    if min_velocity < self.release_velocity_threshold:
                        current_pressed_keys.add(pop_finger[3])
                    self.release_velocity_history.append(min_velocity)
                    print(f"Minimum Release Velocity for {finger_name} ({hand_label}): {min_velocity}") # This will be logged


            viz_utils.draw_finger_tip_info  (color_image, px, py, depth_m)

            # Check if this finger is currently active and should be released
            if (self.has_active_finger() and
                    hand_idx == self.active_hand and
                    finger_name == self.active_finger):
                current_active_time = time.time() - self.start_press_time
                if current_active_time > self.active_time_limit:
                    self.reset_active_finger()
                if depth_m < self.release_threshold:
                    self.fingers_depth_frames_history[hand_label][finger_name].append((depth_m, False, time.time(), None))
                    print(f"Key {self.active_key} released by {finger_name} ({hand_label})!") # This will be logged
                    self.reset_active_finger()
                else:
                    pressed_key = self.fingers_depth_frames_history[hand_label][finger_name][-1][3]
                    self.fingers_depth_frames_history[hand_label][finger_name].append((depth_m, True, time.time(), pressed_key))
                    self.num_active_finger_frames_history[hand_label][finger_name] = min(self.frame_queue_size, self.num_active_finger_frames_history[hand_label][finger_name]+1)
                continue

            # Skip if there's already an active finger
            if self.has_active_finger():
                self.fingers_depth_frames_history[hand_label][finger_name].append((depth_m, False, time.time(), None))
                continue

            # Check for new key presses
            flag = False            # Flag to check if the key is pressed
            finger_frame_in_state = (depth_m, False, time.time())
            finger_point = (px, py)
            for key_data in self.keyboard_manager.get_annotated_keys():
                if self.keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    if self.is_finger_pressing_key(key_data, depth_m):
                        key_name = key_data['key']
                        # self.set_active_finger(hand_idx, finger_name, key_name)
                        # # current_pressed_keys.add(key_name)
                        # finger_frame_in_state = (depth_m, True, time.time(), key_name)
                        # flag = True
                        # print(f"Key {key_name} pressed by {finger_name} ({hand_label})!") # This will be logged

                        press_velo_history = []
                        for finger_frame_state in self.fingers_depth_frames_history[hand_label][finger_name]:
                            if not finger_frame_state[1]:
                                velocity = self._calculate_velocity(finger_frame_state, finger_frame_in_state)
                                press_velo_history.append(velocity)
                        if press_velo_history: # Ensure press_velo_history is not empty
                            max_velocity = max(press_velo_history)
                            print(f"Maximum Press Velocity for {finger_name} ({hand_label}): {max_velocity}") # This will be logged
                            self.press_velocity_history.append(max_velocity)
                            if max_velocity > self.press_velocity_threshold:
                                self.set_active_finger(hand_idx, finger_name, key_name)
                                self.start_press_time = time.time()
                                # current_pressed_keys.add(key_name)
                                finger_frame_in_state = (depth_m, True, time.time(), key_name)
                                flag = True
                                print(f"Key {key_name} pressed by {finger_name} ({hand_label})!") # This will be logged
                                self.start_press_time = time.time()
                        break
            self.fingers_depth_frames_history[hand_label][finger_name].append(finger_frame_in_state)
            if flag:
                self.num_active_finger_frames_history[hand_label][finger_name] = min(self.frame_queue_size, self.num_active_finger_frames_history[hand_label][finger_name]+1)

        return current_pressed_keys

    def simulate_key_presses(self, current_pressed_keys: Set[str]):
        """Simulate key presses and releases using pynput."""
        newly_pressed = current_pressed_keys - self.last_pressed_keys
        newly_released = self.last_pressed_keys - current_pressed_keys

        for key_str in newly_pressed:
            self._press_key(key_str)

        for key_str in newly_released:
            self._release_key(key_str)

        self.last_pressed_keys = current_pressed_keys

    def _press_key(self, key_str: str):
        """Press a single key."""
        try:
            if key_str in self.KEY_MAP:
                self.keyboard.press(self.KEY_MAP[key_str])
            elif len(key_str) == 1:
                self.keyboard.press(key_str.lower())
        except Exception as e:
            print(f"Could not press key '{key_str}': {e}") # This will be logged

    def _release_key(self, key_str: str):
        """Release a single key."""
        try:
            if key_str in self.KEY_MAP:
                self.keyboard.release(self.KEY_MAP[key_str])
            elif len(key_str) == 1:
                self.keyboard.release(key_str.lower())
        except Exception as e:
            print(f"Could not release key '{key_str}': {e}") # This will be logged

    def cleanup(self):
        """Clean up resources and release any pressed keys."""
        print("Application stopping...") # This will be logged

        # Release all pressed keys
        for key_str in self.last_pressed_keys:
            self._release_key(key_str)

        self.camera_manager.stop_stream()
        self.hand_tracker.close()
        cv2.destroyAllWindows()
        print("Application stopped.") # This will be logged
        self._close_log_file() # Close the log file when cleaning up

    def start_ui_thread(self):
        """Start the UI in a separate thread."""
        self.ui_thread = threading.Thread(target=self._run_ui, daemon=True)
        self.ui_thread.start()

    def _run_ui(self):
        """Run the tkinter UI in a separate thread."""
        try:
            root = tk.Tk()
            root.title("Virtual Keyboard Output")
            root.geometry("600x400")

            main_frame = tk.Frame(root, padx=10, pady=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            label = tk.Label(main_frame, text="Click inside this box to start typing with the virtual keyboard.")
            label.pack(pady=(0, 5))

            text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=20)
            text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            text_area.focus()

            root.mainloop()
        except Exception as e:
            print(f"Error in UI thread: {e}") # This will be logged

    def run(self):
        """Main application loop."""
        # Initialize
        if not self.load_key_thresholds():
            self._close_log_file() # Ensure log file is closed on early exit
            return

        # Start UI
        self.start_ui_thread()

        try:
            if not self.camera_manager.start_stream():
                print("Failed to start camera stream. Exiting.") # This will be logged
                self._close_log_file() # Ensure log file is closed on early exit
                return

            while True:
                color_image, aligned_depth_frame, depth_frame_dims = self.camera_manager.get_frames()
                if color_image is None or aligned_depth_frame is None:
                    continue

                current_pressed_keys = set()
                results = self.hand_tracker.process_frame(color_image)
                if results.multi_hand_landmarks:
                    # hand_label=results.multi_handedness[0].classification[0].label # This line might cause an error if no hands are detected
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        self.hand_tracker.draw_landmarks(color_image, hand_landmarks)
                        hand_label = results.multi_handedness[hand_idx].classification[0].label
                        hand_pressed_keys = self.process_finger_tips(
                            hand_landmarks, hand_idx, color_image,
                            aligned_depth_frame, depth_frame_dims,
                            hand_label,
                        )
                        current_pressed_keys.update(hand_pressed_keys)
                if self.has_active_finger():
                    self.touched_frames_count += 1
                else:
                    if self.touched_frames_count > 0:
                        self.touched_frames_count_history.append(self.touched_frames_count)
                    self.touched_frames_count = 0
                # Simulate key presses
                self.simulate_key_presses(current_pressed_keys)

                # Visualization
                viz_utils.draw_keycap_annotations(
                    color_image,
                    self.keyboard_manager.get_annotated_keys(),
                    self.active_key,
                    self.points_per_key
                )
                cv2.imshow('Virtual Keyboard Interface', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # self._plot_touched_frames_distribution()
            # if self.release_velocity_history: # Only plot if data exists
            #     self._plot_release_velocity_history()
            # else:
            #     print("No release velocity data to plot.")
            # if self.press_velocity_history: # Only plot if data exists
            #     self._plot_press_velocity_history()
            # else:
            #     print("No press velocity data to plot.")
            self.cleanup()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="hoang")
    parser.add_argument("--frame_per_second", type=int, default=30)
    args = parser.parse_args()
    app = TapboardX(name=args.name, frame_per_second=args.frame_per_second)
    app.run()