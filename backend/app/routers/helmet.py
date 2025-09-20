import json
import logging
import time
from datetime import datetime

import cv2
from app.configuration import Configuration
from app.service.detect import ObjectDetect
from app.service.visualizer import DetectionVisualizer
from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

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


def generate_events():
    """Synchronous generator that yields Server-Sent Events (SSE) with JSON payloads.

    Endpoint: /helmet/events
    """
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
        logger.error("Cannot open video source for events")
        raise ValueError("Cannot open video source")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # perform detection
            results_helmet, results_person = detects.detect(
                frame,
                config.model_settings.helmet_conf_threshold,
                config.model_settings.person_conf_threshold,
            )

            # visualizer returns (frame_with_drawings, has_no_helmet)
            _, has_no_helmet = visualizer.draw_detections(
                frame,
                results_helmet,
                results_person,
                config.detection_visualizer.roi_points,
            )

            person_count = 0
            try:
                person_count = (
                    len(results_person.boxes.xyxy)
                    if hasattr(results_person, "boxes")
                    else 0
                )
            except Exception:
                try:
                    person_count = (
                        len(results_person.boxes)
                        if hasattr(results_person, "boxes")
                        else 0
                    )
                except Exception:
                    person_count = 0

            payload = {
                "id": int(time.time() * 1000),
                "timestamp": datetime.now().isoformat(),
                "camera": config.application_settings.video_path
                if not config.application_settings.use_webcam
                else f"webcam:{config.application_settings.webcam_id}",
                "helmet": False if has_no_helmet else True,
                "person_count": int(person_count),
            }

            yield f"data: {json.dumps(payload)}\n\n"

            # sleep a bit to avoid flooding; use interval from config if available
            time.sleep(max(0.05, getattr(config, "detection_interval", 0.1)))
    finally:
        cap.release()


@router.get("/detect", status_code=status.HTTP_200_OK)
async def helmet_detection_stream():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/events", status_code=status.HTTP_200_OK)
async def helmet_events_stream():
    """Server-Sent Events endpoint returning JSON payloads for each detection frame."""
    return StreamingResponse(generate_events(), media_type="text/event-stream")
