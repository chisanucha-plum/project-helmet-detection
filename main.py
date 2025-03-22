import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from ultralytics import YOLO

model_helmet = YOLO('epoch150.pt')  
# model_person = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('samui_.mp4')

roi_points = [(455, 341), (455, 341), (34, 469), (31, 833), (675, 832)]
frame_count = []
object_count = []
current_frame = 0

last_helmet_detection = -50

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
class_names_person = ['person']

while cap.isOpened():
    start_time = time.time()  
    ret, frame = cap.read()
    if not ret:
        break

    results_helmet = model_helmet(frame, conf=0.4) 
    # results_person = model_person(frame, conf=1, iou=0.5)

    num_objects = 0
    for result in results_helmet:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  
            x1, y1, x2, y2 = map(int, box[:4])  
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  

            inside = cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False)
            if inside >= 0:  
                num_objects += 1
                class_name = class_names[int(cls)]

                if class_name == 'helmet off' and current_frame - last_helmet_detection > 50:
                    last_helmet_detection = current_frame
                    print(f"Helmet Off detected at frame {current_frame}")
                    filename = f"helmet_off/{current_frame}.jpg"
                    cv2.imwrite(filename, frame)
                    
                if class_name == 'helmet off':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # for result in results_person:
    #     for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
    #         x1, y1, x2, y2 = map(int, box[:4])
    #         center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
    #         inside = cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False)
    #         if inside >= 0:
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0, 0), 2)  
    #             class_name = class_names_person[0]
    #             cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #             cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
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
    ax.set_ylim(0, max(object_count) + 2 if object_count else 10)  
    plt.draw()
    plt.pause(0.01)  
    
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()




  
