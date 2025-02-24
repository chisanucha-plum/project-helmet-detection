import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter  

model = YOLO('bests.pt')  
cap = cv2.VideoCapture('test3.mp4')

roi_points =  [(3, 443), (1, 446), (212, 411), (209, 226), (480, 253), (628, 283), (593, 358), (684, 533), (8, 539)]

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

# ตั้งค่า Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)  
kf.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])  
kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  
kf.P *= 1000  
kf.R *= 10  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้ Tracking แทนการ Detect ธรรมดา
    results = model.track(frame, persist=True, conf=0.5, iou=0.5)

    num_objects = 0
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  

            inside = cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False)
            if inside >= 0:  
                num_objects += 1

                # ใช้ Kalman Filter เพื่อทำให้ Bounding Box นิ่งขึ้น
                kf.predict()
                kf.update([center_x, center_y])  
                smooth_x, smooth_y = kf.x[0], kf.x[2]

                cv2.rectangle(frame, (int(smooth_x - 20), int(smooth_y - 20)), 
                                      (int(smooth_x + 20), int(smooth_y + 20)), 
                                      (0, 255, 0), 2)

            if len(roi_points) > 2:
                cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=2)

    # ใช้ Moving Average ลดการกระพริบ
    object_count_smoothed = np.mean(object_count[-5:]) if len(object_count) > 5 else num_objects

    frame_count.append(current_frame)
    object_count.append(object_count_smoothed)
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
