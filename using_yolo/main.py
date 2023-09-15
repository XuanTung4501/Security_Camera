from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
import math
from sort import *
import time
from Utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', default="../videos/people.mp4")
args = parser.parse_args()

pTime = 0
cap = cv2.VideoCapture(args.video_path)  # For Video

model = YOLO("../models/best.pt")

classNames = ['person']


# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
detectionArea = [120, 161, 735, 500]  # xmin, ymin, xmax, ymax
alpha = 0.4
totalCount = []

while True:
    success, img = cap.read()
    img1 = img.copy()
    img = opaqueRect(img, detectionArea, alpha)
    results = model(img)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf >= 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if detectionArea[0] < cx < detectionArea[2] and detectionArea[1] < cy < detectionArea[3]:
            cvzone.putTextRect(img, 'intruder', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
            crop_img = img1[y1:y1 + h, x1:x1 + w]

            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.imwrite(f"./detected_images/intruder{str(int(id))}.png", crop_img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    showDateTime(img, (800, 45))
    cv2.putText(img, f'FPS: {str(int(fps))}', (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.putText(img, f'Press Q to Quit', (400, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
