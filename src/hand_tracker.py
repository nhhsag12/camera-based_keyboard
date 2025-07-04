import time

import mediapipe as mp
import cv2

from src.one_euro_filter import OneEuroFilter


class HandTracker:
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3,
                 min_cutoff=4.5, beta=1.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.filters = {}
        self.min_cutoff = min_cutoff
        self.beta = beta

        # Define the indices of the fingertip landmarks
        self.fingertip_indices = [
            self.mp_hands.HandLandmark.THUMB_TIP.value,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
            self.mp_hands.HandLandmark.RING_FINGER_TIP.value,
            self.mp_hands.HandLandmark.PINKY_TIP.value
        ]

    def process_frame(self, image):
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        RGB_image.flags.writeable = False
        results = self.hands.process(RGB_image)
        RGB_image.flags.writeable = True

        if results.multi_hand_landmarks:
            current_time = time.time()
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx not in self.filters:
                    # Initialize filters for this hand if it's new
                    self.filters[hand_idx] = {}
                    # Only initialize filters for fingertip landmarks and their x, y components
                    for i in self.fingertip_indices:
                        self.filters[hand_idx][i] = {
                            'x': OneEuroFilter(t0=current_time, x0=hand_landmarks.landmark[i].x,
                                               min_cutoff=self.min_cutoff, beta=self.beta),
                            'y': OneEuroFilter(t0=current_time, x0=hand_landmarks.landmark[i].y,
                                               min_cutoff=self.min_cutoff, beta=self.beta)
                            # 'z' is intentionally omitted here
                        }

                # Apply filter only to specified fingertip landmarks' x and y
                for i, landmark in enumerate(hand_landmarks.landmark):
                    if i in self.fingertip_indices:
                        # Apply filter for X coordinate
                        filtered_x = self.filters[hand_idx][i]['x'](current_time, landmark.x)
                        landmark.x = filtered_x

                        # Apply filter for Y coordinate
                        filtered_y = self.filters[hand_idx][i]['y'](current_time, landmark.y)
                        landmark.y = filtered_y

                        # Z coordinate is NOT filtered here
                    # Other non-fingertip landmarks are also NOT filtered

        return results

    # The get_*_finger_tip methods will now return filtered x,y and unfiltered z
    def get_index_finger_tip(self, hand_landmarks, image_shape):
        h, w, _ = image_shape
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
        finger_pixel_x = int(index_finger_tip.x * w)
        finger_pixel_y = int(index_finger_tip.y * h)
        return (finger_pixel_x, finger_pixel_y), index_finger_tip.z

    def get_thumb_finger_tip(self, hand_landmarks, image_shape):
        h, w, _ = image_shape
        thumb_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP.value]
        finger_pixel_x = int(thumb_finger_tip.x * w)
        finger_pixel_y = int(thumb_finger_tip.y * h)
        return (finger_pixel_x, finger_pixel_y), thumb_finger_tip.z

    def get_middle_finger_tip(self, hand_landmarks, image_shape):
        h, w, _ = image_shape
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value]
        finger_pixel_x = int(middle_finger_tip.x * w)
        finger_pixel_y = int(middle_finger_tip.y * h)
        return (finger_pixel_x, finger_pixel_y), middle_finger_tip.z

    def get_ring_finger_tip(self, hand_landmarks, image_shape):
        h, w, _ = image_shape
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP.value]
        finger_pixel_x = int(ring_finger_tip.x * w)
        finger_pixel_y = int(ring_finger_tip.y * h)
        return (finger_pixel_x, finger_pixel_y), ring_finger_tip.z

    def get_pinky_finger_tip(self, hand_landmarks, image_shape):
        h, w, _ = image_shape
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP.value]
        finger_pixel_x = int(pinky_finger_tip.x * w)
        finger_pixel_y = int(pinky_finger_tip.y * h)
        return (finger_pixel_x, finger_pixel_y), pinky_finger_tip.z

    def draw_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def close(self):
        self.hands.close()
