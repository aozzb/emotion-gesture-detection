# Responsible for:
# - the real-time loop
# - calling all modules
# - displaying output


import cv2
from face_detector import FaceDetector
from face_expression import FaceExpression
import time

def main():
    detector= FaceDetector()
    expr=FaceExpression()

    cap=cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    prev_time = 0
    
    while True:
        ret, frame=cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        box=detector.get_face_box(frame)

        if box is not None:
            x,y,w,h=box

            cv2.rectangle(
                frame,
                (x,y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

        expression=expr.get_expression(frame)

        print("Expression:", expression)

        cv2.putText(
            frame,
            expression,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Webcam - Expression Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

