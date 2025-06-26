import cv2
import numpy as np

def draw_finger_tip_info(image, finger_pixel_x, finger_pixel_y, depth_at_finger_m):
    cv2.circle(image, (finger_pixel_x, finger_pixel_y), 5, (0, 255, 255), -1)  # Yellow circle
    cv2.putText(image, f"Depth: {depth_at_finger_m:.4f}m",
                (finger_pixel_x + 10, finger_pixel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def draw_keycap_annotations(image, annotated_keys, current_frame_detected_key, points_per_key):
    for key_data in annotated_keys:
        key_points_list = key_data['points']
        key_value = key_data['key']

        if len(key_points_list) == points_per_key:
            pts = np.array([[p['x'], p['y']] for p in key_points_list], np.int32)
            pts = pts.reshape((-1, 1, 2))

            if key_value == current_frame_detected_key:
                polygon_color = (255, 255, 0)  # Cyan for detected key
                # text_color = (255, 255, 0)
                thickness = 2
            else:
                polygon_color = (0, 0, 255)  # Red for undetected keys
                # text_color = (0, 0, 255)
                thickness = 1

            cv2.polylines(image, [pts], True, polygon_color, thickness)

            for p in key_points_list:
                cv2.circle(image, (p['x'], p['y']), 1, polygon_color, -1)

            if key_points_list:
                first_p = key_points_list[0]
                # cv2.putText(image, key_value, (first_p['x'] + 10, first_p['y'] + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

def display_text_overlays(image, current_key, typed_text):
    current_key_display_text = f"Current: {current_key if current_key else 'None'}"
    cv2.putText(image, current_key_display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    typed_text_display = f"Typed: {typed_text}"
    cv2.putText(image, typed_text_display, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)