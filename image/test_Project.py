import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture('tests.mp4')


ROI_x1, ROI_y1 = 50, 100 # มุมบนซ้าย
ROI_x2, ROI_y2 = 700, 700  # มุมล่างขวา


frame_count = []
object_count = []
current_frame = 0

plt.ion()  
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  
ax.set_xlim(0, 50)  
ax.set_ylim(0, 10) 
ax.set_xlabel("Frame")
ax.set_ylabel("Objects Detected")
ax.set_title("Real-time Object Detection Graph")
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 940, 680)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    roi_frame = frame[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
    results = model(roi_frame) 
    detected_frame = results[0].plot(labels=False)
    frame[ROI_y1:ROI_y2, ROI_x1:ROI_x2] = detected_frame
    cv2.rectangle(frame, (ROI_x1, ROI_y1), (ROI_x2, ROI_y2), (255, 0, 0), 3)
    num_objects = len(results[0].boxes)  

    frame_count.append(current_frame)
    object_count.append(num_objects)
    current_frame += 1

    if len(frame_count) > 50:
        frame_count.pop(0)
        object_count.pop(0)

    line.set_xdata(frame_count)
    line.set_ydata(object_count)
    ax.set_xlim(max(0, current_frame - 50), current_frame)
    ax.set_ylim(0, max(object_count) + 2)
    plt.draw()
    plt.pause(0.01)  
    
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
