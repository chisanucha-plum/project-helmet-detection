import cv2
import numpy as np
from datetime import datetime

class DetectionVisualizer:
    """
    Class สำหรับวาดผลลัพธ์การตรวจจับ (คนและหมวกกันน็อค) ลงบนเฟรมวิดีโอ
    และตรวจสอบเงื่อนไขต่างๆ เช่น ROI และขนาดของวัตถุ
    """
    def __init__(self):
        # กำหนดสีในรูปแบบ BGR (Blue, Green, Red)
        self.colors = {
            'person': (203, 192, 255),   # สีชมพูอ่อน
            'helmet_on': (0, 255, 0),    # สีเขียว
            'helmet_off': (0, 0, 255),   # สีแดง
            'roi': (255, 0, 0)           # สีน้ำเงิน
        }
        # กำหนดค่าคงที่สำหรับอัตราส่วนขนาดคน
        self.MIN_PERSON_HEIGHT_WIDTH_RATIO = 1.2 
        self.TIMESTAMP_FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.TIMESTAMP_SCALE = 1
        self.TIMESTAMP_COLOR = (255, 255, 255) # สีขาว
        self.TIMESTAMP_THICKNESS = 2

        self.DETECTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.DETECTION_SCALE = 0.5
        self.DETECTION_THICKNESS = 2

    def is_in_roi(self, point, roi_points):
        """
        ตรวจสอบว่าจุด (center_point ของ bounding box) อยู่ใน Region of Interest (ROI) หรือไม่
        Args:
            point (tuple): จุด (x, y) ที่ต้องการตรวจสอบ
            roi_points (list): ลิสต์ของจุด (x, y) ที่กำหนด Polygon ของ ROI
        Returns:
            bool: True ถ้าจุดอยู่ใน ROI, False ถ้าไม่อยู่
        """
        if not roi_points: # เพิ่มการตรวจสอบกรณีที่ roi_points เป็น None หรือว่างเปล่า
            return True # ถ้าไม่มี ROI กำหนด ก็ถือว่าอยู่ใน ROI เสมอ
        
        # แปลง point เป็น tuple ของ int
        point_int = (int(point[0]), int(point[1]))
        # cv2.pointPolygonTest จะคืนค่า >= 0 ถ้าจุดอยู่ภายในหรือบนเส้นขอบ
        return cv2.pointPolygonTest(np.array(roi_points), point_int, False) >= 0

    def is_valid_person_size(self, width, height):
        """
        ตรวจสอบขนาดของ bounding box ว่าเหมาะสมกับขนาดของคนหรือไม่ โดยดูจากอัตราส่วนความสูงต่อความกว้าง
        Args:
            width (float): ความกว้างของ bounding box
            height (float): ความสูงของ bounding box
        Returns:
            bool: True ถ้าขนาดเหมาะสม, False ถ้าไม่เหมาะสม
        """
        if width <= 0: # ป้องกันหารด้วยศูนย์
            return False
        ratio = height / width
        return ratio >= self.MIN_PERSON_HEIGHT_WIDTH_RATIO

    def draw_detections(self, frame, results_helmet, results_person, roi_points):
        """
        วาดผลลัพธ์การตรวจจับ (คน, หมวกกันน็อค) และ ROI ลงบนเฟรม
        Args:
            frame (np.array): เฟรมรูปภาพต้นฉบับ (BGR format)
            results_helmet (object): ผลลัพธ์การตรวจจับหมวกกันน็อค (จากโมเดล YOLO หรือคล้ายกัน)
                                     ควรมี attribute 'boxes.data'
            results_person (object): ผลลัพธ์การตรวจจับคน (จากโมเดล YOLO หรือคล้ายกัน)
                                     ควรมี attribute 'boxes.data'
            roi_points (list): ลิสต์ของจุด (x, y) ที่กำหนด Polygon ของ ROI
        Returns:
            tuple: (frame ที่วาดแล้ว, has_no_helmet_in_roi)
                frame_with_detections (np.array): เฟรมที่วาดผลลัพธ์ลงไป
                has_no_helmet_in_roi (bool): True ถ้าพบคนไม่สวมหมวกใน ROI, False ถ้าไม่พบ
        Raises:
            Exception: หากเกิดข้อผิดพลาดในการวาด
        """
        try:
            # ใช้ชื่อตัวแปรที่สื่อความหมายมากขึ้น
            found_person_no_helmet_in_roi = False 
            
            # 1. วาด ROI
            if roi_points: # ตรวจสอบว่ามี roi_points ก่อนวาด
                roi_np = np.array(roi_points, np.int32) # ระบุ dtype เป็น np.int32
                cv2.polylines(frame, [roi_np], True, self.colors['roi'], 2)

            # 2. วาด person detections
            # ตรวจสอบว่า results_person.boxes และ results_person.boxes.data มีอยู่จริง
            if hasattr(results_person, 'boxes') and results_person.boxes is not None and results_person.boxes.data is not None:
                for box in results_person.boxes.data:
                    # ตรวจสอบจำนวน element ใน box เพื่อป้องกัน IndexError
                    if len(box) < 6: 
                        continue # ข้าม box ที่ข้อมูลไม่ครบ

                    x1, y1, x2, y2, conf, cls = box[:6] # ตรวจสอบให้แน่ใจว่าใช้แค่ 6 ค่าแรก

                    width = x2 - x1
                    height = y2 - y1
                    
                    # ตรวจสอบอัตราส่วนและความถูกต้องของขนาด
                    if width > 0 and height > 0 and self.is_valid_person_size(width, height):
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # ตรวจสอบว่าจุดศูนย์กลางของคนอยู่ใน ROI
                        if self.is_in_roi((center_x, center_y), roi_points):
                            # วาดกรอบคน
                            cv2.rectangle(frame, 
                                          (int(x1), int(y1)), 
                                          (int(x2), int(y2)), 
                                          self.colors['person'], 2)
                            # ไม่ต้องนับ num_objects ตรงนี้ ถ้าจะนับคนจริงๆ ให้แยก logic
                            # หรือนำไปใช้ประโยชน์ในภายหลัง (เช่น count_persons_in_roi)
            
            # 3. วาด helmet detections
            if hasattr(results_helmet, 'boxes') and results_helmet.boxes is not None and results_helmet.boxes.data is not None:
                for box in results_helmet.boxes.data:
                    # ตรวจสอบจำนวน element ใน box เพื่อป้องกัน IndexError
                    if len(box) < 6:
                        continue # ข้าม box ที่ข้อมูลไม่ครบ

                    x1, y1, x2, y2, conf, cls = box[:6]
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # ตรวจสอบว่าจุดศูนย์กลางของหมวกอยู่ใน ROI
                    if self.is_in_roi((center_x, center_y), roi_points):
                        # class 0 มักจะเป็น 'no_helmet' และ class 1 เป็น 'helmet_on'
                        is_no_helmet = (int(cls) == 0) 
                        
                        color = self.colors['helmet_off'] if is_no_helmet else self.colors['helmet_on']
                        label = "No Helmet" if is_no_helmet else "Helmet"

                        # ถ้าเป็นคนไม่สวมหมวก และอยู่ใน ROI ให้ตั้งค่าแฟล็ก
                        if is_no_helmet: 
                            found_person_no_helmet_in_roi = True

                        # วาดกรอบหมวก
                        cv2.rectangle(frame, 
                                      (int(x1), int(y1)), 
                                      (int(x2), int(y2)), 
                                      color, self.DETECTION_THICKNESS)
                        
                        # วาดข้อความ
                        cv2.putText(frame, 
                                    f"{label} {conf:.2f}", 
                                    (int(x1), int(y1) - 10), # ตำแหน่งข้อความ
                                    self.DETECTION_FONT, 
                                    self.DETECTION_SCALE, 
                                    color, 
                                    self.DETECTION_THICKNESS)
                        
            # 4. เพิ่ม Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                        self.TIMESTAMP_FONT, self.TIMESTAMP_SCALE, self.TIMESTAMP_COLOR, self.TIMESTAMP_THICKNESS)

            # ส่งค่า frame ที่วาดแล้ว และสถานะการพบคนไม่สวมหมวกใน ROI
            return frame, found_person_no_helmet_in_roi

        except Exception as e:
            # การจัดการข้อผิดพลาดที่ดีขึ้น อาจจะ log ข้อผิดพลาดหรือจัดการเป็นพิเศษ

            raise Exception(f"Visualization error: {str(e)}") # ยังคง raise เพื่อให้โค้ดภายนอกรู้ว่ามีปัญหา