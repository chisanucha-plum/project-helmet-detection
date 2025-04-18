import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from ultralytics import YOLO

# สร้างโฟลเดอร์เก็บภาพ helmet_off
os.makedirs("helmet_off", exist_ok=True)

# โหลดโมเดล
model_helmet = YOLO(r'model\epoch150.pt')   # โมเดลตรวจจับ helmet off/on
model_person = YOLO('yolov8n.pt')    # โมเดลตรวจจับ person

# เปิดวิดีโอ
cap = cv2.VideoCapture(r'case\case_4.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# กำหนด ROI
roi_points = [(228, 443), (613, 833), (2, 832), (1, 484)]

# # ตั้งค่ากราฟ real-time
# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot([], [], 'r-')
# ax.set_xlim(0, 50)
# ax.set_ylim(0, 10)
# ax.set_xlabel("Frame")
# ax.set_ylabel("Objects Detected")
# ax.set_title("Real-time Object Detection Graph")

# ตั้งค่าตัวแปรเก็บผล
frame_count = []
object_count = []
current_frame = 0
last_helmet_detection = -50

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 940, 680)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับหมวกกันน็อค45%
    results_helmet = model_helmet(frame, conf=0.45)
    # ตรวจจับคน50%
    results_person = model_person(frame, conf=0.5)

    num_objects = 0

    # วาดผลการตรวจจับ helmet
    for result in results_helmet:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            if cv2.pointPolygonTest(np.array(roi_points), (cx, cy), False) >= 0:
                num_objects += 1
                label = 'helmet off' if cls == 0 else 'helmet on'
                color = (0,0,255) if cls == 0 else (0,255,0)

                # บันทึกภาพเมื่อ helmet off
                if cls == 0 and current_frame - last_helmet_detection > 50:
                    last_helmet_detection = current_frame
                    cv2.imwrite(f"helmet_off/{current_frame}.jpg", frame)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    # วาดผลการตรวจจับ person
    for result in results_person:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                if cv2.pointPolygonTest(np.array(roi_points), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (180,105,255), 2)  
                    cv2.putText(frame, "Person", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    # วาด ROI
    cv2.polylines(frame, [np.array(roi_points)], isClosed=True,
                  color=(0,0,255), thickness=2)

    # # อัพเดตกราฟ
    # frame_count.append(current_frame)
    # object_count.append(num_objects)
    # if len(frame_count) > 50:
    #     frame_count.pop(0)
    #     object_count.pop(0)
    # line.set_xdata(frame_count)
    # line.set_ydata(object_count)
    # ax.set_xlim(max(0, current_frame-50), current_frame)
    # ax.set_ylim(0, max(object_count)+2 if object_count else 10)
    # plt.draw()
    # plt.pause(0.001)

    # แสดง FPS
    fps = 1.0/(time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # แสดงผล
    cv2.imshow('Detection', frame)
    current_frame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# เคลียร์ทรัพยากร
cap.release()
cv2.destroyAllWindows()
# plt.ioff()
# plt.close()
