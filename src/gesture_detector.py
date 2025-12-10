import math

class GestureDetector:
    def __init__(self):
        pass

    def dist(self, p1, p2):           #returns dist bw two points
        if p1 is None or p2 is None:
            return 99999
        return math.dist(p1, p2)
    
    def get_bbox(self, points):
        xs=[p[0] for p in points]
        ys=[p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))  #x1,x2,y1,y2
    
    def get_gesture(self, face_kp, face_bbox, hands):
        if face_kp is None or face_bbox is None:
            return None
        
        mouth_pts=[
            face_kp["mouth_left"],
            face_kp["mouth_right"],
            face_kp["upper_lip"],
            face_kp["lower_lip"],
        ]

        eye_pts=[
            face_kp["left_eye_top"], face_kp["left_eye_bottom"],
            face_kp["right_eye_top"], face_kp["right_eye_bottom"]
        ]

        mouth_x1 = min([p[0] for p in mouth_pts])
        mouth_y1 = min([p[1] for p in mouth_pts])
        mouth_x2 = max([p[0] for p in mouth_pts])
        mouth_y2 = max([p[1] for p in mouth_pts])

        eye_x1 = min([p[0] for p in eye_pts])
        eye_y1 = min([p[1] for p in eye_pts])
        eye_x2 = max([p[0] for p in eye_pts])
        eye_y2 = max([p[1] for p in eye_pts])

        face_w = face_bbox[2]

        touch_thresh=face_w*0.15

        for hand_label in ["left", "right"]:
            hand=hands.get(hand_label)
            if hand is None:
                continue

            index_tip=hand[8]

            chin_point=face_kp["lower_lip"]
            if self.dist(index_tip, chin_point) < touch_thresh:     #simple threshold logic that compares dist bw index finger tip and chin (approximated by lower lip)
                return "chin_touch"
            
            hx1, hy1, hx2, hy2=self.get_bbox(hand)
            if not (hx2 < mouth_x1 or hx1 > mouth_x2 or hy2 < mouth_y1 or hy1 > mouth_y2):      #simple overlap logic
                return "hand_over_mouth"
            
            if not (hx2 <eye_x1 or hx1>eye_x2 or hy2<eye_y1 or hy1>eye_y2):     #same overlap logic
                return "hand_over_eyes"
            
        return None

