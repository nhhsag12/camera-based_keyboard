import cv2
from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils



def run_keyboard_interface():
    # --- Configuration ---
    ANNOTATION_FILENAME = 'src/keyboard_annotations.json'
    POINTS_PER_KEY = 4
    DEPTH_THRESHOLD_M = 0.285  # Minimum depth for a valid keypress detection
    KEYBOARD_ROW_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    KEYBOARD_ROW_2 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
    KEYBOARD_ROW_3 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
    KEYBOARD_ROW_4 = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    KEYBOARD_ROW_5 = ["DEL", "SPACE", "ENTER"]
    DEPTH_THRESHOLD_ROW_1 = (0.285, 0.304)
    DEPTH_THRESHOLD_ROW_2 = (0.295, 0.311)
    DEPTH_THRESHOLD_ROW_3 = (0.305, 0.318)
    DEPTH_THRESHOLD_ROW_4 = (0.310, 0.326)
    DEPTH_THRESHOLD_ROW_5 = (0.325, 0.335)

    def check_touch_down(key_data: dict, keyboard_row: list, depth_finger, depth_threshold: tuple) -> bool:
        if key_data["key"] in keyboard_row and depth_finger >= depth_threshold[0] and depth_finger < depth_threshold[1]:
            return True
        return False


    # --- Initialize Managers ---
    camera_manager = CameraManager()
    hand_tracker = HandTracker()
    keyboard_manager = KeyboardManager(annotation_filename=ANNOTATION_FILENAME, points_per_key=POINTS_PER_KEY)

    # --- Global variables for application state ---
    typed_text = ""
    last_detected_key = None # To avoid repeated additions when holding a key

    try:
        if not camera_manager.start_stream():
            print("Failed to start camera stream. Exiting.")
            return

        while True:
            color_image, aligned_depth_frame, depth_frame_dims = camera_manager.get_frames()

            if color_image is None:
                continue

            current_frame_detected_key = None # Reset for each frame

            # Process hand landmarks
            results = hand_tracker.process_frame(color_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_tracker.draw_landmarks(color_image, hand_landmarks)

                    # Get index finger tip coordinates and depth
                    (thumb_finger_pixel_x, thumb_finger_pixel_y), _ = hand_tracker.get_thumb_finger_tip(hand_landmarks, color_image.shape)
                    (index_finger_pixel_x, index_finger_pixel_y), _ = hand_tracker.get_index_finger_tip(hand_landmarks, color_image.shape)
                    (middle_finger_pixel_x, middle_finger_pixel_y), _ = hand_tracker.get_middle_finger_tip(hand_landmarks, color_image.shape)
                    (ring_finger_pixel_x, ring_finger_pixel_y), _ = hand_tracker.get_ring_finger_tip(hand_landmarks, color_image.shape)
                    (pinky_finger_pixel_x, pinky_finger_pixel_y), _ = hand_tracker.get_pinky_finger_tip(hand_landmarks, color_image.shape)

                    # Clamp coordinates for depth lookup
                    clamped_thumb_finger_pixel_x = max(0, min(thumb_finger_pixel_x, depth_frame_dims[0] - 1))
                    clamped_thumb_finger_pixel_y = max(0, min(thumb_finger_pixel_y, depth_frame_dims[1] - 1))
                    clamped_index_finger_pixel_x = max(0, min(index_finger_pixel_x, depth_frame_dims[0] - 1))
                    clamped_index_finger_pixel_y = max(0, min(index_finger_pixel_y, depth_frame_dims[1] - 1))
                    clamped_middle_finger_pixel_x = max(0, min(middle_finger_pixel_x, depth_frame_dims[0] - 1))
                    clamped_middle_finger_pixel_y = max(0, min(middle_finger_pixel_y, depth_frame_dims[1] - 1))
                    clamped_ring_finger_pixel_x = max(0, min(ring_finger_pixel_x, depth_frame_dims[0] - 1))
                    clamped_ring_finger_pixel_y = max(0, min(ring_finger_pixel_y, depth_frame_dims[1] - 1))
                    clamped_pinky_finger_pixel_x = max(0, min(pinky_finger_pixel_x, depth_frame_dims[0] - 1))
                    clamped_pinky_finger_pixel_y = max(0, min(pinky_finger_pixel_y, depth_frame_dims[1] - 1))

                    depth_at_index_finger_m = aligned_depth_frame.get_distance(clamped_index_finger_pixel_x, clamped_index_finger_pixel_y)
                    depth_at_thumb_finger_m = aligned_depth_frame.get_distance(clamped_thumb_finger_pixel_x, clamped_thumb_finger_pixel_y)
                    depth_at_middle_finger_m = aligned_depth_frame.get_distance(clamped_middle_finger_pixel_x, clamped_middle_finger_pixel_y)
                    depth_at_ring_finger_m = aligned_depth_frame.get_distance(clamped_ring_finger_pixel_x, clamped_ring_finger_pixel_y)
                    depth_at_pinky_finger_m = aligned_depth_frame.get_distance(clamped_pinky_finger_pixel_x, clamped_pinky_finger_pixel_y)


                    # Draw finger tip info
                    viz_utils.draw_finger_tip_info(color_image, index_finger_pixel_x, index_finger_pixel_y, depth_at_index_finger_m)
                    viz_utils.draw_finger_tip_info(color_image, thumb_finger_pixel_x, thumb_finger_pixel_y, depth_at_thumb_finger_m)
                    viz_utils.draw_finger_tip_info(color_image, middle_finger_pixel_x, middle_finger_pixel_y, depth_at_middle_finger_m)
                    viz_utils.draw_finger_tip_info(color_image, ring_finger_pixel_x, ring_finger_pixel_y, depth_at_ring_finger_m)
                    viz_utils.draw_finger_tip_info(color_image, pinky_finger_pixel_x, pinky_finger_pixel_y, depth_at_pinky_finger_m)

                    # Check for key press if finger is within depth threshold
                    if depth_at_index_finger_m >= DEPTH_THRESHOLD_M:
                        finger_point = (index_finger_pixel_x, index_finger_pixel_y)
                        for key_data in keyboard_manager.get_annotated_keys():
                            if check_touch_down(key_data, KEYBOARD_ROW_1, depth_at_index_finger_m, DEPTH_THRESHOLD_ROW_1) or check_touch_down(key_data, KEYBOARD_ROW_2, depth_at_index_finger_m, DEPTH_THRESHOLD_ROW_2) or check_touch_down(key_data, KEYBOARD_ROW_3, depth_at_index_finger_m, DEPTH_THRESHOLD_ROW_3) or check_touch_down(key_data, KEYBOARD_ROW_4, depth_at_index_finger_m , DEPTH_THRESHOLD_ROW_4)  or check_touch_down(key_data, KEYBOARD_ROW_5, depth_at_index_finger_m, DEPTH_THRESHOLD_ROW_5):
                                if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                                    current_frame_detected_key = key_data['key']
                                    break # Found a key, no need to check others

                    # if depth_at_thumb_finger_m >= DEPTH_THRESHOLD_M:
                    #     finger_point = (thumb_finger_pixel_x, thumb_finger_pixel_y)
                    #     for key_data in keyboard_manager.get_annotated_keys():
                    #         if check_touch_down(key_data, KEYBOARD_ROW_1, depth_at_thumb_finger_m, DEPTH_THRESHOLD_ROW_1) or check_touch_down(key_data, KEYBOARD_ROW_2, depth_at_thumb_finger_m, DEPTH_THRESHOLD_ROW_2) or check_touch_down(key_data, KEYBOARD_ROW_3, depth_at_thumb_finger_m, DEPTH_THRESHOLD_ROW_3) or check_touch_down(key_data, KEYBOARD_ROW_4, depth_at_thumb_finger_m , DEPTH_THRESHOLD_ROW_4)  or check_touch_down(key_data, KEYBOARD_ROW_5, depth_at_thumb_finger_m, DEPTH_THRESHOLD_ROW_5):
                    #             if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    #                 current_frame_detected_key = key_data['key']
                    #                 break
                    #
                    # if depth_at_middle_finger_m >= DEPTH_THRESHOLD_M:
                    #     finger_point = (middle_finger_pixel_x, middle_finger_pixel_y)
                    #     for key_data in keyboard_manager.get_annotated_keys():
                    #         if check_touch_down(key_data, KEYBOARD_ROW_1, depth_at_middle_finger_m, DEPTH_THRESHOLD_ROW_1) or check_touch_down(key_data, KEYBOARD_ROW_2, depth_at_middle_finger_m, DEPTH_THRESHOLD_ROW_2) or check_touch_down(key_data, KEYBOARD_ROW_3, depth_at_middle_finger_m, DEPTH_THRESHOLD_ROW_3) or check_touch_down(key_data, KEYBOARD_ROW_4, depth_at_middle_finger_m , DEPTH_THRESHOLD_ROW_4)  or check_touch_down(key_data, KEYBOARD_ROW_5, depth_at_middle_finger_m, DEPTH_THRESHOLD_ROW_5):
                    #             if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    #                 current_frame_detected_key = key_data['key']
                    #                 break
                    #
                    # if depth_at_ring_finger_m >= DEPTH_THRESHOLD_M:
                    #     finger_point = (ring_finger_pixel_x, ring_finger_pixel_y)
                    #     for key_data in keyboard_manager.get_annotated_keys():
                    #         if check_touch_down(key_data, KEYBOARD_ROW_1, depth_at_ring_finger_m, DEPTH_THRESHOLD_ROW_1) or check_touch_down(key_data, KEYBOARD_ROW_2, depth_at_ring_finger_m, DEPTH_THRESHOLD_ROW_2) or check_touch_down(key_data, KEYBOARD_ROW_3, depth_at_ring_finger_m, DEPTH_THRESHOLD_ROW_3) or check_touch_down(key_data, KEYBOARD_ROW_4, depth_at_ring_finger_m , DEPTH_THRESHOLD_ROW_4)  or check_touch_down(key_data, KEYBOARD_ROW_5, depth_at_ring_finger_m, DEPTH_THRESHOLD_ROW_5):
                    #             if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    #                 current_frame_detected_key = key_data['key']
                    #                 break
                    #
                    # if depth_at_pinky_finger_m >= DEPTH_THRESHOLD_M:
                    #     finger_point = (pinky_finger_pixel_x, pinky_finger_pixel_y)
                    #     for key_data in keyboard_manager.get_annotated_keys():
                    #         if check_touch_down(key_data, KEYBOARD_ROW_1, depth_at_pinky_finger_m, DEPTH_THRESHOLD_ROW_1) or check_touch_down(key_data, KEYBOARD_ROW_2, depth_at_pinky_finger_m, DEPTH_THRESHOLD_ROW_2) or check_touch_down(key_data, KEYBOARD_ROW_3, depth_at_pinky_finger_m, DEPTH_THRESHOLD_ROW_3) or check_touch_down(key_data, KEYBOARD_ROW_4, depth_at_pinky_finger_m , DEPTH_THRESHOLD_ROW_4)  or check_touch_down(key_data, KEYBOARD_ROW_5, depth_at_pinky_finger_m, DEPTH_THRESHOLD_ROW_5):
                    #             if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                    #                 current_frame_detected_key = key_data['key']
                    #                 break


            # Update typed text based on detection
            if current_frame_detected_key and current_frame_detected_key != last_detected_key:
                if current_frame_detected_key == "ENTER":
                    typed_text += "\n"
                elif current_frame_detected_key == "DEL":
                    typed_text = typed_text[:-1]
                elif current_frame_detected_key == "SPACE":
                    typed_text += " "
                else:
                    typed_text += current_frame_detected_key
            last_detected_key = current_frame_detected_key

            # Draw keycap annotations and highlight detected key
            viz_utils.draw_keycap_annotations(color_image, keyboard_manager.get_annotated_keys(), current_frame_detected_key, POINTS_PER_KEY)

            # Display typed text and current key
            viz_utils.display_text_overlays(color_image, current_frame_detected_key, typed_text)

            # Display the result
            cv2.imshow('Virtual Keyboard Interface', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up resources
        camera_manager.stop_stream()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print("Application stopped.")

if __name__ == "__main__":
    run_keyboard_interface()