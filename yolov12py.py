# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:40:30 2025

@author: dominika
"""


import cv2
from ultralytics import YOLO

model = YOLO('yolov12s.pt')

video_path = "myvideo.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv12 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
