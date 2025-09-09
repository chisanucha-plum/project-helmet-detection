import logging

import cv2
from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.configuration import Configuration
from app.service.detect import ObjectDetect
from app.service.visualizer import DetectionVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Helmet Detection"])


async def generate_frames():
    config = Configuration.get_config()
    detects = ObjectDetect(
        config.model_settings.helmet_model_path, config.model_settings.person_model_path
    )
    visualizer = DetectionVisualizer()
    cap = cv2.VideoCapture(
        config.application_settings.webcam_id
        if config.application_settings.use_webcam
        else config.application_settings.video_path
    )
    if not cap.isOpened():
        logger.error("Cannot open video source")
        raise ValueError("Cannot open video source")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results_helmet, results_person = detects.detect(
            frame,
            config.model_settings.helmet_conf_threshold,
            config.model_settings.person_conf_threshold,
        )
        frame, has_no_helmet = visualizer.draw_detections(
            frame,
            results_helmet,
            results_person,
            config.detection_visualizer.roi_points,
        )
        # encode frame as jpeg
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    cap.release()


@router.get("/detect", status_code=status.HTTP_200_OK)
async def helmet_detection_stream():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
