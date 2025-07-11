import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
from camera_manager import CameraManager # Import CameraManager

# --- Configuration for RealSense Camera ---
# Define camera resolution (consistent for both color and depth)
# These values will be passed to CameraManager

# Initialize CameraManager
camera_manager = CameraManager()

# --- Initialize MediaPipe Hands ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Depth Tracking Variables ---
# min/max depths for the currently active tracking period
current_period_min_depth = float('inf')
current_period_max_depth = float('-inf')

# min/max depths to display (could be current period or last finished period)
display_min_depth = None
display_max_depth = None

current_finger_depth = 0.0 # To display current depth

tracking_active = False # True when recording, False otherwise
tracking_start_time = 0.0 # Timestamp when tracking started

# --- Main Program ---
try:
    # Start streaming using CameraManager
    if not camera_manager.start_stream():
        print("Failed to start camera stream. Exiting.")
        exit()

    # The depth scale is now managed by CameraManager if needed elsewhere
    # depth_scale = camera_manager.depth_scale # You can access it like this if required

    print("\n--- Finger Depth Tracking Program (Manual Control) ---")
    print("Move your index finger in front of the camera.")
    print("Press 'SPACE' to START tracking the min/max depth of your finger.")
    print("Press 'SPACE' again to STOP tracking and display the results.")
    print("Press 'r' to RESET the displayed min/max depths.")
    print("Press 'q' to QUIT the program.")

    while True:
        # Get frames from CameraManager
        color_image, aligned_depth_frame, depth_frame_dims = camera_manager.get_frames()

        if color_image is None or aligned_depth_frame is None:
            continue

        # Get actual width and height of the depth frame for clamping from the returned tuple
        depth_frame_width = depth_frame_dims[0]
        depth_frame_height = depth_frame_dims[1]

        # Reset current finger depth for this frame
        current_finger_depth = 0.0
        finger_detected_this_frame = False

        # Process the color image with MediaPipe Hands
        # The CameraManager already sets color_image.flags.writeable = True
        RGB_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        # Draw hand landmarks and get finger depth
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmark for the tip of the index finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]

                # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                h, w, _ = color_image.shape
                finger_pixel_x, finger_pixel_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Clamp coordinates to ensure they are within the depth frame's bounds
                # Use depth_frame_width for X and depth_frame_height for Y
                clamped_finger_pixel_x = max(0, min(finger_pixel_x, depth_frame_width - 1))
                clamped_finger_pixel_y = max(0, min(finger_pixel_y, depth_frame_height - 1))

                # Get depth value at the landmark's pixel location
                depth_at_finger_m = aligned_depth_frame.get_distance(clamped_finger_pixel_x, clamped_finger_pixel_y)

                # Only consider valid depths (non-zero and within a reasonable range, e.g., < 5m)
                if depth_at_finger_m > 0 and depth_at_finger_m < 5.0:
                    current_finger_depth = depth_at_finger_m
                    finger_detected_this_frame = True

                    # Draw a circle at the landmark and display the depth
                    cv2.circle(color_image, (finger_pixel_x, finger_pixel_y), 5, (0, 255, 255), -1) # Yellow circle
                    cv2.putText(color_image, f"Current Depth: {current_finger_depth:.3f}m",
                                (finger_pixel_x + 10, finger_pixel_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # --- Depth Tracking Logic (only if active) ---
                    if tracking_active:
                        current_period_min_depth = min(current_period_min_depth, current_finger_depth)
                        current_period_max_depth = max(current_period_max_depth, current_finger_depth)
                        display_min_depth = current_period_min_depth # Update display values live
                        display_max_depth = current_period_max_depth

                break # Only track one hand for simplicity

        # --- Display Tracking Information ---
        status_text = ""
        if tracking_active:
            elapsed_time = time.time() - tracking_start_time
            status_text = f"Tracking... Elapsed: {elapsed_time:.1f}s"
            status_color = (0, 255, 0) # Green for active
        else:
            status_text = "Tracking PAUSED. Press 'SPACE' to START."
            status_color = (0, 165, 255) # Orange for paused

        cv2.putText(color_image, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        cv2.putText(color_image, f"Current: {current_finger_depth:.3f}m" if finger_detected_this_frame else "Current: N/A",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if display_min_depth is not None and display_min_depth != float('inf'):
            cv2.putText(color_image, f"Min: {display_min_depth:.3f}m",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(color_image, "Min: --.--m",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

        if display_max_depth is not None and display_max_depth != float('-inf'):
            cv2.putText(color_image, f"Max: {display_max_depth:.3f}m",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(color_image, "Max: --.--m",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

        # Display instructions at the bottom
        cv2.putText(color_image, "SPACE: Start/Stop Tracking | R: Reset | Q: Quit",
                    (10, color_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        # Display the color frame
        cv2.imshow('Finger Depth Tracking (Manual)', color_image)

        # --- Key Press Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): # Spacebar pressed
            if not tracking_active:
                # Start tracking
                tracking_active = True
                tracking_start_time = time.time()
                current_period_min_depth = float('inf') # Reset for new period
                current_period_max_depth = float('-inf')
                print("Tracking STARTED.")
            else:
                # Stop tracking
                tracking_active = False
                print("Tracking STOPPED.")
                if display_min_depth != float('inf') and display_max_depth != float('-inf'):
                    print(f"Results: Min Depth: {display_min_depth:.3f}m, Max Depth: {display_max_depth:.3f}m")
                else:
                    print("No valid finger depths recorded during the period.")
        elif key == ord('r'): # Reset all
            tracking_active = False
            current_period_min_depth = float('inf')
            current_period_max_depth = float('-inf')
            display_min_depth = None # Reset display values
            display_max_depth = None
            current_finger_depth = 0.0
            tracking_start_time = 0.0
            print("Tracking data RESET.")

finally:
    # Stop streaming and clean up resources using CameraManager
    print("\nStopping stream and cleaning up...")
    camera_manager.stop_stream()
    cv2.destroyAllWindows()
    hands.close()