import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self, min_conf=0.5):               #min_conf sets a minimum confidence level required to be sure about an emotion               
        self.mp_face=mp.solutions.face_detection
        self.detector=self.mp_face.FaceDetection(
            model_selection=0,                      #optimised for webcam/close range faces
            min_detection_confidence=min_conf
        )

    def detect(self, frame):                  #returns (x,y,w,h) for detected face and "None" if no face found
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #mediapipe expects RGB while opencv gives BGR, hence convert
        result=self.detector.process(rgb)

        if not result.detections:
            return None
        
        h, w, _=frame.shape                         #extract height and width
        detection=result.detections[0]               #take only first face detected
        box=detection.location_data.relative_bounding_box

        x=int(box.xmin * w)                         #converting all to integers as my cropping func will need pixel coordinates
        y=int(box.ymin * h)
        w_box=int(box.width *w)
        h_box=int(box.height * h)

        return (x,y,w_box,h_box)
    


