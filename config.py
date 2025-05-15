# Configuration
HELMET_MODEL_PATH = r'model\epoch250.pt'
PERSON_MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = r'case\case_03.mp4'
ROI_POINTS =  [(1, 309), (406, 218), (1113, 591), (942, 755), (8, 757) ]

HELMET_CONF_THRESHOLD = 0.50 #50%
PERSON_CONF_THRESHOLD = 0.70 #70%
HELMET_DETECTION_INTERVAL = 50

USE_WEBCAM = True  # เปลี่ยนเป็น False ใช้วิดีโอ
WEBCAM_ID = 0     