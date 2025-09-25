import torch
from ultralytics import YOLO


class ObjectDetect:
    def __init__(self, helmet_model_path, motorcycle_model_path):
        # เช็คว่ามี CUDA พร้อมใช้งานหรือไม่
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # เพิ่มการปรับแต่งประสิทธิภาพ CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # ล้าง cache
            torch.backends.cudnn.benchmark = True  # เพิ่มความเร็ว
            torch.backends.cudnn.deterministic = False  # เพิ่มประสิทธิภาพ
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB"
            )

        self.model_helmet = YOLO(helmet_model_path)
        # ใช้ YOLOv8 pre-trained model สำหรับจับมอเตอร์ไซค์ (class=3)
        self.model_motorcycle = YOLO(motorcycle_model_path)

        if torch.cuda.is_available():
            self.model_helmet.fuse()  # เพิ่มความเร็ว inference
            self.model_motorcycle.fuse()

        # Frame Skipping สำหรับเพิ่ม FPS
        self.frame_skip_counter = 0
        self.skip_frames = 2  # ข้าม 2 เฟรม แล้วประมวลผล 1 เฟรม (เพิ่ม FPS 3 เท่า)
        self.last_results = None  # เก็บผลลัพธ์ล่าสุด

    def detect(self, frame, helmet_conf, motorcycle_conf):
        """Frame Skipping สำหรับเพิ่ม FPS"""
        # Frame Skipping: ข้ามเฟรมเพื่อเพิ่ม FPS
        self.frame_skip_counter += 1
        if (
            self.frame_skip_counter <= self.skip_frames
            and self.last_results is not None
        ):
            return self.last_results  # ใช้ผลลัพธ์เก่า

        # รีเซ็ต counter และประมวลผลเฟรมใหม่
        self.frame_skip_counter = 0

        # device (CUDA or CPU)
        with torch.no_grad():  # ประหยัด memory และเพิ่มความเร็ว
            results_helmet = self.model_helmet(
                frame,
                conf=helmet_conf,
                device=self.device,
                half=True,  # ใช้ FP16 เพื่อประหยัด VRAM
                verbose=False,
            )[0]

            results_motorcycle = self.model_motorcycle(
                frame,
                conf=motorcycle_conf,
                classes=[3],
                device=self.device,
                half=True,  # ใช้ FP16 เพื่อประหยัด VRAM
                verbose=False,
            )[0]

        # เก็บผลลัพธ์สำหรับ frame skipping
        self.last_results = (results_helmet, results_motorcycle)
        return results_helmet, results_motorcycle
