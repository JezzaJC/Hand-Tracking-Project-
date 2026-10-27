import cv2
import numpy as np
import math

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Use cosine rule to find angle
            a = math.dist(end, start)
            b = math.dist(far, start)
            c = math.dist(end, far)
            angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))  # in radians

            if angle <= math.pi / 2:  # angle less than 90 degree, count as finger
                finger_count += 1
                cv2.circle(drawing, far, 5, [0, 0, 255], -1)

        return finger_count
    return 0

def run_hand_tracking():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 2000:
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Count fingers
                fingers = count_fingers(max_contour, frame)
                cv2.putText(frame, f"Fingers: {fingers}", (x, y-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking with Finger Count (x to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_tracking()
