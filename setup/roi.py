import cv2
import numpy as np

roi_points = []  
selecting_roi = True 

# ฟังก์ชันเมื่อคลิกเมาส์เพื่อเลือก ROI
def draw_roi(event, x, y, flags, param):
    global roi_points, selecting_roi

    if event == cv2.EVENT_LBUTTONDOWN:  
        roi_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) > 2:  
        selecting_roi = False

cap = cv2.VideoCapture('samuis.mp4') 
ret, frame = cap.read()
cap.release()

cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', draw_roi)

# ตั้งค่าขนาดหน้าต่างหลังจากเปิดใช้งาน
cv2.resizeWindow('Select ROI', 940, 680)

while selecting_roi:
    temp_frame = frame.copy()
    for i in range(1, len(roi_points)):
        cv2.line(temp_frame, roi_points[i - 1], roi_points[i], (0, 255, 0), 2)

    if len(roi_points) > 2:
        cv2.polylines(temp_frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Select ROI", temp_frame)
    
    # ใช้ 'q' เพื่อออกจากโหมดเลือก ROI
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()  # ปิดหน้าต่างทั้งหมด
print("Selected ROI Points:", roi_points)  # บันทึกจุดที่เลือกไว้
