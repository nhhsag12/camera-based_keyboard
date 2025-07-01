import cv2
import json
from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils

def run_keyboard_interface():
    """
    Initializes and runs the main loop for the virtual keyboard interface,
    handling camera input, hand tracking, key press detection based on per-key
    depth thresholds, and visualization.
    """

    # --- Configuration ---
    ANNOTATION_FILENAME = 'assets/keyboard_annotations.json'
    THRESHOLDS_FILENAME = 'assets/key_thresholds.json'  # Path to your new JSON file
    POINTS_PER_KEY = 4
    # A general depth threshold to check if a finger is near the keyboard plane.
    # This acts as a preliminary filter before checking per-key thresholds.
    GENERAL_DEPTH_THRESHOLD_M = 0.21

    # --- Per-Key Depth Threshold Configuration ---
    # This dictionary will be loaded from the JSON file.
    KEY_DEPTH_THRESHOLDS = {}

    def load_key_thresholds_from_file(filename: str) -> bool:
        """
        Loads the per-key depth thresholds from a specified JSON file.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        nonlocal KEY_DEPTH_THRESHOLDS
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Convert lists from JSON back to tuples
                KEY_DEPTH_THRESHOLDS = {key: tuple(value) for key, value in data.items()}
            print(f"Successfully loaded key thresholds from '{filename}'.")
            return True
        except FileNotFoundError:
            print(f"Error: The threshold file '{filename}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{filename}'. Check for syntax errors.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while loading thresholds: {e}")
            return False


    def is_finger_pressing_key(key_data: dict, finger_depth: float) -> bool:
        """
        Checks if a finger's depth is within the specific threshold for a given key.

        Args:
            key_data (dict): Dictionary containing key information, including the key name.
            finger_depth (float): The current depth of the finger in meters.

        Returns:
            bool: True if the finger is considered to be pressing the key, False otherwise.
        """
        key_name = key_data.get("key")
        if not key_name:
            return False

        # Get the specific threshold for this key.
        threshold = KEY_DEPTH_THRESHOLDS.get(key_name)
        if not threshold:
            # This key might not be in our configuration; ignore it.
            return False

        min_depth, max_depth = threshold
        return min_depth <= finger_depth < max_depth

    # --- Initialize ---
    # Load thresholds from file. If it fails, exit the application.
    if not load_key_thresholds_from_file(THRESHOLDS_FILENAME):
        return

    camera_manager = CameraManager()
    hand_tracker = HandTracker()
    keyboard_manager = KeyboardManager(annotation_filename=ANNOTATION_FILENAME, points_per_key=POINTS_PER_KEY)

    # --- Application State ---
    typed_text = ""
    last_detected_key = None # Avoids rapid-fire typing from a single press

    try:
        if not camera_manager.start_stream():
            print("Failed to start camera stream. Exiting.")
            return

        while True:
            color_image, aligned_depth_frame, depth_frame_dims = camera_manager.get_frames()

            if color_image is None or aligned_depth_frame is None:
                continue

            current_frame_detected_key = None # Reset for each new frame
            results = hand_tracker.process_frame(color_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_tracker.draw_landmarks(color_image, hand_landmarks)

                    # --- Get all finger tip data ---
                    finger_tips = {
                        'thumb': hand_tracker.get_thumb_finger_tip(hand_landmarks, color_image.shape),
                        'index': hand_tracker.get_index_finger_tip(hand_landmarks, color_image.shape),
                        'middle': hand_tracker.get_middle_finger_tip(hand_landmarks, color_image.shape),
                        'ring': hand_tracker.get_ring_finger_tip(hand_landmarks, color_image.shape),
                        'pinky': hand_tracker.get_pinky_finger_tip(hand_landmarks, color_image.shape),
                    }

                    # --- Process each finger for a key press ---
                    for finger_name, (tip_coords, _) in finger_tips.items():
                        # Ensure we have a key detection to break out of this inner loop
                        if current_frame_detected_key:
                            break

                        px, py = tip_coords

                        # Get actual width and height of the depth frame for clamping
                        depth_frame_width = depth_frame_dims[0]  # Width
                        depth_frame_height = depth_frame_dims[1]  # Height

                        # Clamp coordinates to be within the depth frame dimensions
                        # clamped_px should be within [0, depth_frame_width - 1]
                        clamped_px = max(0, min(px, depth_frame_width - 1))
                        # clamped_py should be within [0, depth_frame_height - 1]
                        clamped_py = max(0, min(py, depth_frame_height - 1))

                        # Now pass the correctly clamped x (clamped_px) and y (clamped_py)
                        depth_m = aligned_depth_frame.get_distance(clamped_px, clamped_py)

                        # Draw info for the current finger
                        viz_utils.draw_finger_tip_info(color_image, px, py, depth_m)

                        # --- Key Press Logic ---
                        # 1. First, check if the finger is close enough to the keyboard plane in general.
                        if depth_m >= GENERAL_DEPTH_THRESHOLD_M:
                            finger_point = (px, py)
                            # 2. Then, iterate through keys to see if the finger is inside a keycap
                            #    and meets that specific key's depth threshold.
                            for key_data in keyboard_manager.get_annotated_keys():
                                if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                                    if is_finger_pressing_key(key_data, depth_m):
                                        current_frame_detected_key = key_data['key']
                                        # Key found, no need to check other keys for this finger
                                        break

            # --- Update Typed Text ---
            if current_frame_detected_key and current_frame_detected_key != last_detected_key:
                if current_frame_detected_key == "ENTER":
                    typed_text += "\n"
                elif current_frame_detected_key == "BACKSPACE":
                    typed_text = typed_text[:-1]
                elif current_frame_detected_key == "SPACE":
                    typed_text += " "
                elif current_frame_detected_key not in ["CTRL", "ALT", "WIN", "SHIFT", "ESC", "DEL"]:
                    typed_text += current_frame_detected_key
            last_detected_key = current_frame_detected_key

            # --- Visualization ---
            viz_utils.draw_keycap_annotations(color_image, keyboard_manager.get_annotated_keys(), current_frame_detected_key, POINTS_PER_KEY)
            viz_utils.display_text_overlays(color_image, current_frame_detected_key, typed_text)

            cv2.imshow('Virtual Keyboard Interface', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- Clean Up ---
        camera_manager.stop_stream()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print("Application stopped.")

if __name__ == "__main__":
    run_keyboard_interface()
