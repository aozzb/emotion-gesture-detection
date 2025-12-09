# Responsible for:
# - Initializing MediaPipe Face Mesh
# - Detecting facial landmarks for one face
# - Computing geometric metrics such as:
#       * mouth open ratio
#       * smile curvature
#       * eye aspect ratio (eye openness)
# - Classifying simple expressions based on threshold rules:
#       * "smile"
#       * "mouth_open"
#       * "eyes_closed"
#       * "neutral"
# - Returning a label describing the detected face expression

import cv2
import mediapipe as mp
import numpy as np

class FaceExpression:
    def __init__(self):
        self.mp_face_mesh=mp.solutions.face_mesh
        self.face_mesh=self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,                  #better accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        #landmarks used for tracking
        self.MOUTH_LEFT=61
        self.MOUTH_RIGHT=291
        self.UPPER_LIP=13
        self.LOWER_LIP=14

        self.LEFT_EYE_TOP=159
        self.LEFT_EYE_BOTTOM=145
        self.RIGHT_EYE_TOP=386
        self.RIGHT_EYE_BOTTOM=374

        # smoothing buffers
        self.ear_history = []
        self.mouth_open_history = []
        self.smile_history = []


    def _get_landmarks(self, frame):
        #return normalised landmarks if face is found or else return none
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result=self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None
        
        return result.multi_face_landmarks[0]   #we want first face detected

    def _distance(self, p1,p2):
        return np.linalg.norm(np.array(p1)-np.array(p2))
    
    def _smooth(self, history, new_value, window=5):
        history.append(new_value)
        if len(history) > window:
            history.pop(0)
        return sum(history) / len(history)


    def _get_points(self, frame, landmarks):
        #returns the facial landmarks in pixel coordinates (x,y)
        h,w, _=frame.shape

        def lm(idx):
            pt=landmarks.landmark[idx]
            return (int(pt.x *w), int(pt.y *h))
        
        points={
            "mouth_left": lm(self.MOUTH_LEFT),
            "mouth_right": lm(self.MOUTH_RIGHT),
            "upper_lip": lm(self.UPPER_LIP),
            "lower_lip": lm(self.LOWER_LIP),
            "left_eye_top": lm(self.LEFT_EYE_TOP),
            "left_eye_bottom": lm(self.LEFT_EYE_BOTTOM),
            "right_eye_top": lm(self.RIGHT_EYE_TOP),
            "right_eye_bottom": lm(self.RIGHT_EYE_BOTTOM),
        }

        return points
    
    def _mouth_open_ratio(self,pts):
        #how open the mouth is
        mouth_width=self._distance(pts["mouth_left"], pts["mouth_right"])
        lip_height=self._distance(pts["upper_lip"], pts["lower_lip"])

        if mouth_width==0:
            return 0
        
        return lip_height/mouth_width
    
    def _smile_ratio(self, pts):
        #how wide the mouth is
        mouth_width=self._distance(pts["mouth_left"], pts["mouth_right"])
        lip_height=self._distance(pts["upper_lip"], pts["lower_lip"])

        if lip_height==0:
            return 0
        
        return mouth_width/lip_height
    
    def _ear(self, pts):
        #how open the eyes are
        left_eye=self._distance(pts["left_eye_top"], pts["left_eye_bottom"])
        right_eye=self._distance(pts["right_eye_top"], pts["right_eye_bottom"])

        return (left_eye+right_eye)/2
    
    def _smile_curvature(self, pts):
        #Measures how much the mouth corners are raised.
        left = pts["mouth_left"]
        right = pts["mouth_right"]
        upper = pts["upper_lip"]
        lower = pts["lower_lip"]
        # lip center = midpoint of upper & lower lip
        lip_center_y = (upper[1] + lower[1]) / 2
        corner_avg_y = (left[1] + right[1]) / 2

        # Positive value = corners are higher than lip center (smile)
        return lip_center_y - corner_avg_y

    
    def get_expression(self, frame):
        #returns one of the following: eyes_closed, mouth_open, smile, neutral

        landmarks=self._get_landmarks(frame)
        if landmarks is None:
            return "neutral"
        
        pts=self._get_points(frame, landmarks)

        # raw metrics
        mouth_open_ratio = self._mouth_open_ratio(pts)
        smile_ratio = self._smile_ratio(pts)
        ear = self._ear(pts)

        # smoothed metrics
        mouth_open_ratio = self._smooth(self.mouth_open_history, mouth_open_ratio)
        smile_ratio = self._smooth(self.smile_history, smile_ratio)
        ear = self._smooth(self.ear_history, ear)

        if ear<5:
            return "eyes_closed"
        elif mouth_open_ratio>0.55:
            return "mouth_open"
        elif smile_ratio > 2.8 and self._smile_curvature(pts) > 5:
            return "smile"
        else:
            return "neutral"
        
    def _get_all_landmarks_pixel(self, frame, landmarks):
        """Return all 468 face mesh landmarks in pixel coordinates."""
        h, w, _ = frame.shape
        pts = []
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
        return pts
    
    
