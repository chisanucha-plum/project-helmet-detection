import cv2
import numpy as np
from datetime import datetime

class DetectionVisualizer:
    def __init__(self):
        self.colors = {
            'person': (203, 192, 255),  # สีชมพู (BGR format)
            'helmet_on': (0, 255, 0),   # สีเขียว
            'helmet_off': (0, 0, 255),  # สีแดง
            'roi': (255, 0, 0)          # สีน้ำเงิน
        }

    def is_in_roi(self, point, roi_points):
        """ตรวจสอบว่าจุดอยู่ใน ROI หรือไม่"""
        # แปลง point เป็น tuple ของ int
        point_int = (int(point[0]), int(point[1]))
        return cv2.pointPolygonTest(np.array(roi_points), point_int, False) >= 0

    def is_valid_person_size(self, width, height):
        """ตรวจสอบขนาดของ bounding box ว่าเหมาะสมกับคนหรือไม่"""
        min_ratio = 1.2  # อัตราส่วนความสูงต่อความกว้างขั้นต่ำ
        ratio = height / width
        return ratio >= min_ratio

    def draw_detections(self, frame, results_helmet, results_person, roi_points):
        try:
            num_objects = 0
            
            # วาด ROI
            if roi_points is not None:
                roi_np = np.array(roi_points)
                cv2.polylines(frame, [roi_np], True, self.colors['roi'], 2)

            # วาด person detections
            if hasattr(results_person, 'boxes'):
                for box in results_person.boxes.data:
                    x1, y1, x2, y2, conf, cls = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # ตรวจสอบอัตราส่วนของ bounding box
                    if self.is_valid_person_size(width, height):
                        center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
                        if self.is_in_roi(center_point, roi_points):
                            cv2.rectangle(frame, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        self.colors['person'], 2)

            # วาด helmet detections
            if hasattr(results_helmet, 'boxes'):
                for box in results_helmet.boxes.data:
                    x1, y1, x2, y2, conf, cls = box
                    center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
                    if self.is_in_roi(center_point, roi_points):
                        color = self.colors['helmet_off'] if int(cls) == 0 else self.colors['helmet_on']
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    color, 2)
                        label = "No Helmet" if int(cls) == 0 else "Helmet"
                        cv2.putText(frame, f"{label} {conf:.2f}", 
                                  (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
                        num_objects += 1

            # เพิ่ม timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            return frame, num_objects

        except Exception as e:
            raise Exception(f"Visualization error: {str(e)}")

