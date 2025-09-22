from ultralytics import YOLO


class ObjectDetect:
    def __init__(self, helmet_model_path, motorcycle_model_path):
        self.model_helmet = YOLO(helmet_model_path)
        # ใช้ YOLOv8 pre-trained model สำหรับจับมอเตอร์ไซค์ (class=3)
        self.model_motorcycle = YOLO(motorcycle_model_path)

    def detect(self, frame, helmet_conf, motorcycle_conf):
        results_helmet = self.model_helmet(frame, conf=helmet_conf)[0]
        results_motorcycle = self.model_motorcycle(
            frame, conf=motorcycle_conf, classes=[3]
        )[0]
        return results_helmet, results_motorcycle
