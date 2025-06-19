from ultralytics import YOLO  # เพิ่ม import YOLO

class ObjectDetect:
    def __init__(self, helmet_model_path, person_model_path):
        try:
            self.model_helmet = YOLO(helmet_model_path)
            self.model_person = YOLO(person_model_path)
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def detect(self, frame, helmet_conf, person_conf):
        try:
            results_helmet = self.model_helmet(frame, conf=helmet_conf)[0]
            results_person = self.model_person(frame, conf=person_conf)[0]
            return results_helmet, results_person
        except Exception as e:
            raise Exception(f"Detection error: {str(e)}")