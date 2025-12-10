import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(
            static_image_mode=False,        #if this was true, it would be used for image processing not video
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw=mp.solutions.drawing_utils

    def detect(self, frame):
        #returns pixel coords of hand landmarks
        h,w,_=frame.shape
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result=self.hands.process(rgb)

        hands_dict={"left": None, "right": None}

        if not result.multi_hand_landmarks:
            return hands_dict
        
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label=handedness.classification[0].label.lower()  #tells us either left or right

            points=[]
            for lm in hand_landmarks.landmark:
                x,y = int(lm.x * w), int(lm.y *h)
                points.append((x,y))

            hands_dict[label]=points

        return hands_dict
