import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', default="../videos/video1.mp4")
args = parser.parse_args()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(args.video_path)

while cap.isOpened():
    # Reading the video stream
    ret, image = cap.read()
    if ret:
        regions, _ = hog.detectMultiScale(image,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)

        for x, y, w, h in regions:
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 1)

        cv2.imshow("Image", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()