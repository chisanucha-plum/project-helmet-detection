HELMET_MODEL_PATH = r'train\epoch250.pt'
PERSON_MODEL_PATH = r'train\yolov8n.pt'
VIDEO_PATH = r'src\case\case_02.mp4'
ROI_POINTS =  [(130, 40), (1350, 37), (1332, 717), (107, 697)]

HELMET_CONF_THRESHOLD = 0.50 #50%
PERSON_CONF_THRESHOLD = 0.70 #70%
HELMET_DETECTION_INTERVAL = 50

USE_WEBCAM = False  # เปลี่ยนเป็น False ใช้วิดีโอ
WEBCAM_ID = 0