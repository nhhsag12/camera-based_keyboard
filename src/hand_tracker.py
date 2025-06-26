import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, image):
        # Convert the BGR image to RGB for MediaPipe.
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        RGB_image.flags.writeable = False
        results = self.hands.process(RGB_image)
        # Mark the image as writeable again, if it was set to False.
        RGB_image.flags.writeable = True
        return results

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

    # def get_pinky_finger_tip(self, hand_landmarks, image_shape):
    #     h, w, _ = image_shape
    #     pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP.value]

    # def get_finger_distance(self, finger_pixel_1, finger_pixel_2):
    #     return ((finger_pixel_1[0] - finger_pixel_2[0]) ** 2 + (finger_pixel_1[1] - finger_pixel_2[1]) ** 2) ** 0.5

    def draw_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def close(self):
        self.hands.close()