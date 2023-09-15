import cv2
from datetime import datetime
import numpy as np

def opaqueRect(img, rect, alpha, color=(0,0,255)):
    overlay = img.copy()
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    new_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return new_img

def showDateTime(img, coor, fontsize=1, thickness=2, color=(255,0,255)):
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(img, dt, coor, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness)


