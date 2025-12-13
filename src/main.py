import cv2
import time

from face_detector import FaceDetector
from face_expression import FaceExpression
from hand_detector import HandDetector
from gesture_detector import GestureDetector
from mapper import EmoteMapper


def main():
    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()
    face_expression = FaceExpression()
    hand_detector = HandDetector()
    gesture_detector = GestureDetector()
    emote_mapper=EmoteMapper()

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        #face detection
        face_bbox = face_detector.detect(frame)

        expression = None
        face_kp = None

        if face_bbox is not None:
            # Draw face bbox for debugging
            x, y, w, h = face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

            #face expression
            expression, face_kp = face_expression.detect(frame)

            # Draw essential face keypoints
            if face_kp is not None:
                for key, (px, py) in face_kp.items():
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

        #hand detection
        hands = hand_detector.detect(frame)

        # Draw hand landmarks (optional debugging)
        if hands["left"]:
            for (x, y) in hands["left"]:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        if hands["right"]:
            for (x, y) in hands["right"]:
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        #gesture detection
        gesture = None
        if face_kp is not None and face_bbox is not None:
            gesture = gesture_detector.get_gesture(face_kp, face_bbox, hands)

        #
        emote_key = emote_mapper.resolve_emote(expression, gesture)
        emote_mapper.render(emote_key)


        #calculate fps
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if expression:
            cv2.putText(frame, f"Expression: {expression}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        cv2.imshow("Emotion + Gesture Detection (B2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
