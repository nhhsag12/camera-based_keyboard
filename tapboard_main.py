import cv2
from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils
import time

def run_keyboard_interface():
    # --- Configuration ---
    ANNOTATION_FILENAME = 'assets/keyboard_annotations.json'
    POINTS_PER_KEY = 4
    KEYBOARD_ROW_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    KEYBOARD_ROW_2 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
    KEYBOARD_ROW_3 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
    KEYBOARD_ROW_4 = ['SHIFT', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'BACKSPACE']
    KEYBOARD_ROW_5 = ["CTRL", "ALT", "WIN", "ESC", "DEL", "SPACE", "ENTER"]
    DEPTH_THRESHOLD_ROW_1 = (0.211, 0.230)
    DEPTH_THRESHOLD_ROW_2 = (0.211, 0.229)
    DEPTH_THRESHOLD_ROW_3 = (0.211, 0.230)
    DEPTH_THRESHOLD_ROW_4 = (0.211, 0.230)
    DEPTH_THRESHOLD_ROW_5 = (0.211, 0.230)

    # --- Velocity-based Detection Parameters ---
    # Downward velocity threshold to consider a finger 'Touching' (entering the plane)
    TOUCH_VELOCITY_THRESHOLD = -0.01 # m/s. Small negative to detect entering. Can be 0 or slightly negative.
    # Upward velocity threshold to consider a 'Tap' (release that triggers event)
    TAP_VELOCITY_THRESHOLD = -0.1    # m/s. Significantly positive for a quick lift. Tune this!
    # Minimum depth to consider interaction
    MIN_INTERACTION_DEPTH = 0.20
    # Maximum depth to consider interaction (prevents ghost touches when hand is too far)
    MAX_INTERACTION_DEPTH = 0.25

    # --- Finger Tracking State Variables ---
    previous_depth_at_index_finger_m = None
    last_frame_time = time.time()

    # --- Key State Management ---
    # `key_touched_states`: True if finger is currently "on" the key (similar to your old key_press_states)
    key_touched_states = {key: False for row in [KEYBOARD_ROW_1, KEYBOARD_ROW_2, KEYBOARD_ROW_3, KEYBOARD_ROW_4, KEYBOARD_ROW_5] for key in row}
    # `key_was_touched_this_frame`: A temporary flag for each key to manage transitions
    # This will hold the key that was just "tapped" in the current frame.
    detected_key_event = None
    # This will track the *currently touched* key for display.
    current_displayed_key = None

    # --- Initialize Managers ---
    camera_manager = CameraManager()
    hand_tracker = HandTracker()
    keyboard_manager = KeyboardManager(annotation_filename=ANNOTATION_FILENAME, points_per_key=POINTS_PER_KEY)

    # --- Global variables for application state ---
    typed_text = ""

    try:
        if not camera_manager.start_stream():
            print("Failed to start camera stream. Exiting.")
            return

        while True:
            current_frame_time = time.time()
            delta_time = current_frame_time - last_frame_time
            last_frame_time = current_frame_time

            color_image, aligned_depth_frame, depth_frame_dims = camera_manager.get_frames()

            if color_image is None:
                continue

            detected_key_event = None
            current_displayed_key = None # Reset for each frame
            is_touching_keyboard = False # Flag for overall keyboard touch state
            previous_key_touch_state = key_touched_states.copy()

            results = hand_tracker.process_frame(color_image)

            index_finger_pixel_x, index_finger_pixel_y = None, None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_tracker.draw_landmarks(color_image, hand_landmarks)

                    (index_finger_pixel_x, index_finger_pixel_y), _ = hand_tracker.get_index_finger_tip(hand_landmarks, color_image.shape)

                    clamped_index_finger_pixel_x = max(0, min(index_finger_pixel_x, depth_frame_dims[0] - 1))
                    clamped_index_finger_pixel_y = max(0, min(index_finger_pixel_y, depth_frame_dims[1] - 1))

                    depth_at_index_finger_m = aligned_depth_frame.get_distance(clamped_index_finger_pixel_x, clamped_index_finger_pixel_y)

                    viz_utils.draw_finger_tip_info(color_image, index_finger_pixel_x, index_finger_pixel_y, depth_at_index_finger_m)

                    if previous_depth_at_index_finger_m is not None and delta_time > 0:
                        depth_change = depth_at_index_finger_m - previous_depth_at_index_finger_m
                        depth_velocity = depth_change / delta_time # m/s

                        # Check if finger is within interaction range
                        if MIN_INTERACTION_DEPTH <= depth_at_index_finger_m <= MAX_INTERACTION_DEPTH:
                            finger_point = (index_finger_pixel_x, index_finger_pixel_y)

                            for key_data in keyboard_manager.get_annotated_keys():
                                key_name = key_data['key']

                                # Check if finger is over the keycap AND within the specific row depth
                                is_over_key_and_in_depth = False
                                if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                                    if key_name in KEYBOARD_ROW_1 and DEPTH_THRESHOLD_ROW_1[0] <= depth_at_index_finger_m < DEPTH_THRESHOLD_ROW_1[1]:
                                        is_over_key_and_in_depth = True
                                    elif key_name in KEYBOARD_ROW_2 and DEPTH_THRESHOLD_ROW_2[0] <= depth_at_index_finger_m < DEPTH_THRESHOLD_ROW_2[1]:
                                        is_over_key_and_in_depth = True
                                    elif key_name in KEYBOARD_ROW_3 and DEPTH_THRESHOLD_ROW_3[0] <= depth_at_index_finger_m < DEPTH_THRESHOLD_ROW_3[1]:
                                        is_over_key_and_in_depth = True
                                    elif key_name in KEYBOARD_ROW_4 and DEPTH_THRESHOLD_ROW_4[0] <= depth_at_index_finger_m < DEPTH_THRESHOLD_ROW_4[1]:
                                        is_over_key_and_in_depth = True
                                    elif key_name in KEYBOARD_ROW_5 and DEPTH_THRESHOLD_ROW_5[0] <= depth_at_index_finger_m < DEPTH_THRESHOLD_ROW_5[1]:
                                        is_over_key_and_in_depth = True

                                # --- State Machine Logic (Released -> Touched -> Tapped/Released) ---

                                # 1. Transition to 'Touched' State
                                # Finger is over key, within depth, and moving slightly downward/stable, AND was not touched.
                                if is_over_key_and_in_depth and not key_touched_states[key_name]:
                                    key_touched_states[key_name] = True # Mark as touched
                                    print(f"Key {key_name} - Touched (Velocity: {depth_velocity:.3f})")
                                else:
                                    if not is_over_key_and_in_depth:
                                    # print(previous_key_touch_state[key_name])
                                        key_touched_states[key_name] = False
                                    #     key_touched_states = dict.fromkeys(key_touched_states, False)

                                # 2. Transition from "Touched" to "Released" (and trigger 'Tap' if conditions met)
                                if previous_key_touch_state[key_name] and not key_touched_states[key_name]:
                                    print(f"Key {key_name} - Released (Velocity: {depth_velocity:.3f})")
                                    if depth_velocity < TAP_VELOCITY_THRESHOLD:
                                        print(f"Key {key_name} - Tapped/Released (Velocity: {depth_velocity:.3f})")
                                        detected_key_event = key_name


                                # 2. Transition from 'Touched' to 'Released' (and trigger 'Tap' if conditions met)
                                # Key was touched AND (finger moved off OR finger moved upward rapidly)
                                # elif key_touched_states[key_name]:
                                #     # If finger lifts off the key with sufficient upward velocity (a 'Tap')
                                #     if depth_velocity > TAP_VELOCITY_THRESHOLD:
                                #         print(f"Key {key_name} - Tapped/Released (Velocity: {depth_velocity:.3f})")
                                #         detected_key_event = key_name # This is the key event!
                                #         key_touched_states[key_name] = False # Mark as released
                                #         break # A key was tapped, no need to check others
                                #     # If finger moves off the key without a clear tap (drag-off)
                                #     elif not keyboard_manager.is_point_in_keycap(finger_point, key_data):
                                #         key_touched_states[key_name] = False # Mark as released
                                #         print(f"Key {key_name} - Drag-off Released")


                            # Determine current_displayed_key and is_touching_keyboard based on ANY touched key
                            # This loop is separated to ensure current_displayed_key reflects currently active key
                            any_key_is_touched = False
                            for key_data in keyboard_manager.get_annotated_keys():
                                key_name = key_data['key']
                                if key_touched_states[key_name]:
                                    current_displayed_key = key_name # Highlight the key that is logically "down"
                                    any_key_is_touched = True
                                    break # Only highlight one key if multiple fingers (though current only uses index)
                            is_touching_keyboard = any_key_is_touched

                        else: # Finger is outside the interaction depth range
                            # Reset all key states if finger is too far or too close
                            for key in key_touched_states:
                                key_touched_states[key] = False
                            is_touching_keyboard = False # No finger in interaction range, so not touching

                    previous_depth_at_index_finger_m = depth_at_index_finger_m

            else: # No hand detected, reset all key states
                for key in key_touched_states:
                    key_touched_states[key] = False
                previous_depth_at_index_finger_m = None # Reset previous depth too
                is_touching_keyboard = False

            # Update typed text based on the detected_key_event (which now comes from release/tap)
            if detected_key_event: # A new key tap event occurred
                if detected_key_event == "ENTER":
                    typed_text += "\n"
                elif detected_key_event == "BACKSPACE":
                    typed_text = typed_text[:-1]
                elif detected_key_event == "SPACE":
                    typed_text += " "
                # Special keys that don't add to typed_text
                elif detected_key_event in ["SHIFT", "CTRL", "ALT", "WIN", "ESC", "DEL"]:
                    pass
                else: # Regular character keys
                    typed_text += detected_key_event

            # Draw keycap annotations and highlight the currently held/touched key for visualization
            viz_utils.draw_keycap_annotations(color_image, keyboard_manager.get_annotated_keys(), current_displayed_key, POINTS_PER_KEY)

            # Display typed text and current key
            viz_utils.display_text_overlays(color_image, current_displayed_key, typed_text)

            # --- Display "Fingertip Touching Keyboard" Status ---
            status_text = "Fingertip Touching Keyboard: YES" if is_touching_keyboard else "Fingertip Touching Keyboard: NO"
            status_color = (0, 255, 0) if is_touching_keyboard else (0, 0, 255) # Green for YES, Red for NO
            text_x = 10
            text_y = color_image.shape[0] - 20
            cv2.putText(color_image, status_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

            # --- Display "Fingertip Location" ---
            fingertip_location_text = f"Fingertip Location: ({index_finger_pixel_x}, {index_finger_pixel_y})"
            fingertip_location_color = (0, 255, 0)
            text_x = 10
            text_y = color_image.shape[0] - 40
            cv2.putText(color_image, fingertip_location_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

            cv2.imshow('Virtual Keyboard Interface', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera_manager.stop_stream()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print("Application stopped.")

if __name__ == "__main__":
    run_keyboard_interface()