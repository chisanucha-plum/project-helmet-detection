import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')  
cap = cv2.VideoCapture('image\samui.mp4')

roi_points =   [(251, 832), (74, 547), (50, 465), (720, 267), (1528, 492), (1527, 833)]
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


class_names = ['helmet off', 'helmet on']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, iou=0.5)  

    num_objects = 0
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  
            x1, y1, x2, y2 = map(int, box[:4])  
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  

            inside = cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False)
            if inside >= 0:  
                num_objects += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                class_name = class_names[int(cls)] 
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    
    if len(roi_points) > 2:
        cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=2)

    frame_count.append(current_frame)
    object_count.append(num_objects)
    current_frame += 1

    if len(frame_count) > 50:
        frame_count.pop(0)
        object_count.pop(0)

    line.set_xdata(frame_count)
    line.set_ydata(object_count)
    ax.set_xlim(max(0, current_frame - 50), current_frame)
    ax.set_ylim(0, max(object_count) + 2 if object_count else 10)  #  errors
    plt.draw()
    plt.pause(0.01)  
    
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()  
