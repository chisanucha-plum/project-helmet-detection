import logging
import os
import cv2
from fastapi.responses import StreamingResponse
from app.configuration import *
from app.service.detector import ObjectDetect
from app.service.visualizer import DetectionVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_frames():
    detector = ObjectDetect(HELMET_MODEL_PATH, PERSON_MODEL_PATH)
    visualizer = DetectionVisualizer()
    cap = cv2.VideoCapture(WEBCAM_ID if USE_WEBCAM else VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Cannot open video source")
        raise ValueError("Cannot open video source")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results_helmet, results_person = detector.detect(
            frame, HELMET_CONF_THRESHOLD, PERSON_CONF_THRESHOLD
        )
        frame, has_no_helmet = visualizer.draw_detections(
            frame, results_helmet, results_person, ROI_POINTS
        )
        # encode frame as jpeg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

async def helmet_detection_stream():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')