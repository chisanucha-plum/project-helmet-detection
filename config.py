# Configuration
HELMET_MODEL_PATH = r'model\epoch150.pt'
PERSON_MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = r'case\case_1.mp4'
ROI_POINTS =  [(228, 443), (613, 833), (2, 832), (1, 484)]
HELMET_CONF_THRESHOLD = 0.50 #50%
PERSON_CONF_THRESHOLD = 0.70 #70%
HELMET_DETECTION_INTERVAL = 50
# เพิ่มค่า config สำหรับกล้อง
# USE_WEBCAM = True  # เปลี่ยนเป็น False ถ้าต้องการใช้วิดีโอ
# WEBCAM_ID = 0     # เลข 0 คือกล้องแรกที่พบในระบบทำ