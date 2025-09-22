import json
import logging
import os
import time
from datetime import datetime
from typing import Dict

import cv2
import numpy as np
from app.configuration import Configuration
from app.services.detect import ObjectDetect
from app.services.visualizer import DetectionVisualizer
from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Helmet Detection"])

# Constants
VIDEO_SOURCE_ERROR = "Cannot open video source"
SNAPSHOT_DIR = "snapshots"

# Global tracking variables
# ตัวแปรสำหรับติดตามวัตถุ
tracked_objects: Dict[str, float] = {}  # object_id -> last_seen_time (เวลาที่เจอครั้งล่าสุด)
COOLDOWN_SECONDS = (
    10.0  # Don't capture same object within 10 seconds (ห้ามจับวัตถุเดิมภายใน 10 วินาที)
)


def is_motorcycle_in_center_roi(bbox, roi_points) -> bool:
    """
    Check if motorcycle is in the center area of ROI.
    Only capture when motorcycle is in center, not just touching ROI edges.
    ตรวจสอบว่ามอเตอร์ไซค์อยู่ในพื้นที่กลาง ROI หรือไม่
    จับภาพเฉพาะเมื่ออยู่ตรงกลาง ไม่ใช่แค่สัมผัสขอบ ROI
    """
    if not roi_points or len(roi_points) < 3:
        return False

    # Get motorcycle center
    x1, y1, x2, y2 = bbox
    motorcycle_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Calculate ROI center and dimensions
    roi_array = np.array(roi_points, dtype=np.int32)
    roi_moments = cv2.moments(roi_array)

    if roi_moments["m00"] == 0:
        return False

    roi_center_x = int(roi_moments["m10"] / roi_moments["m00"])
    roi_center_y = int(roi_moments["m01"] / roi_moments["m00"])

    # Calculate ROI bounds
    roi_x_coords = [point[0] for point in roi_points]
    roi_y_coords = [point[1] for point in roi_points]
    roi_width = max(roi_x_coords) - min(roi_x_coords)
    roi_height = max(roi_y_coords) - min(roi_y_coords)

    # Define center area as 30% of ROI size around center
    center_threshold_x = roi_width * 0.15  # 15% on each side = 30% total
    center_threshold_y = roi_height * 0.15

    # Check if motorcycle center is in ROI center area
    distance_from_center_x = abs(motorcycle_center[0] - roi_center_x)
    distance_from_center_y = abs(motorcycle_center[1] - roi_center_y)

    return (
        distance_from_center_x <= center_threshold_x
        and distance_from_center_y <= center_threshold_y
    )


def calculate_motorcycle_id(bbox) -> str:
    """Calculate motorcycle ID based on approximate position only (not count).
    คำนวณ ID ของมอเตอร์ไซค์จากตำแหน่งเท่านั้น (ไม่ใช้จำนวน)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    # Create stable ID based on position grid (larger grid for less sensitivity)
    # สร้าง ID ที่เสถียรจากตาราง grid ตำแหน่ง (grid ใหญ่เพื่อลดความไว)
    return f"mc_{center_x // 100}_{center_y // 100}"


def should_capture_object(object_id: str) -> bool:
    """Check if we should capture this object based on cooldown timer.
    ตรวจสอบว่าควรจับภาพวัตถุนี้หรือไม่ตาม cooldown timer
    """
    current_time = time.time()
    if object_id in tracked_objects:
        time_diff = current_time - tracked_objects[object_id]
        if time_diff < COOLDOWN_SECONDS:
            return False  # ยังไม่ถึงเวลา ห้ามจับซ้ำ

    tracked_objects[object_id] = current_time
    return True


def crop_roi_area(frame, roi_points):
    """
    Crop frame to ROI area only to save storage space.
    เฉพาะพื้นที่ ROI เพื่อประหยัดเนื้อที่
    """
    if not roi_points or len(roi_points) < 3:
        return frame  # Return original frame if no ROI defined

    # Get ROI bounding rectangle
    x_coords = [point[0] for point in roi_points]
    y_coords = [point[1] for point in roi_points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add small padding (5% of ROI size)
    padding_x = int((x_max - x_min) * 0.05)
    padding_y = int((y_max - y_min) * 0.05)

    # Ensure bounds are within frame
    frame_h, frame_w = frame.shape[:2]
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(frame_w, x_max + padding_x)
    y_max = min(frame_h, y_max + padding_y)

    # Crop to ROI area
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    return cropped_frame


def capture_frame_on_roi_entry(
    frame, results_motorcycle, roi_points, has_no_helmet: bool
) -> bool:
    """
    Capture frame when motorcycle is in CENTER of ROI (not just touching edges).
    Use motorcycle-based ID to prevent duplicate captures.
    Returns True if frame was captured, False otherwise.

    จับภาพเมื่อมอเตอร์ไซค์อยู่ตรงกลาง ROI (ไม่ใช่แค่สัมผัสขอบ)
    ใช้ ID ของมอเตอร์ไซค์เพื่อป้องกันการจับภาพซ้ำ
    เก็บเฉพาะพื้นที่ ROI เพื่อประหยัดเนื้อที่
    """
    if not hasattr(results_motorcycle, "boxes") or len(results_motorcycle.boxes) == 0:
        return False

    # Ensure snapshots directory exists
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    for i, bbox in enumerate(results_motorcycle.boxes.xyxy):
        bbox_np = bbox.cpu().numpy()

        # Check if motorcycle is in CENTER of ROI (not just inside ROI)
        if is_motorcycle_in_center_roi(bbox_np, roi_points):
            # Generate motorcycle ID based on position only
            motorcycle_id = calculate_motorcycle_id(bbox_np)

            # Check cooldown to prevent duplicate captures
            if should_capture_object(motorcycle_id):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                helmet_status = "no_helmet" if has_no_helmet else "helmet"
                filename = f"capture_{timestamp}_{helmet_status}_mc_{motorcycle_id}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)

                # Crop to ROI area to save storage space
                # เฉพาะพื้นที่ ROI เพื่อประหยัดเนื้อที่
                cropped_frame = crop_roi_area(frame, roi_points)

                # Save cropped frame
                cv2.imwrite(filepath, cropped_frame)
                return True

    return False


async def generate_frames():
    config = Configuration.get_config()
    detects = ObjectDetect(
        config.model_settings.helmet_model_path,
        config.model_settings.motorcycle_model_path,
    )
    visualizer = DetectionVisualizer()
    cap = cv2.VideoCapture(
        config.application_settings.webcam_id
        if config.application_settings.use_webcam
        else config.application_settings.video_path
    )
    if not cap.isOpened():
        logger.error(VIDEO_SOURCE_ERROR)
        raise ValueError(VIDEO_SOURCE_ERROR)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_helmet, results_motorcycle = detects.detect(
            frame,
            config.model_settings.helmet_conf_threshold,
            config.model_settings.motorcycle_conf_threshold,
        )

        frame, has_no_helmet = visualizer.draw_detections(
            frame,
            results_helmet,
            results_motorcycle,
            config.detection_visualizer.roi_points,
        )

        # Capture frame when new motorcycle enters ROI
        capture_frame_on_roi_entry(
            frame,
            results_motorcycle,
            config.detection_visualizer.roi_points,
            has_no_helmet,
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
        config.model_settings.helmet_model_path,
        config.model_settings.motorcycle_model_path,
    )
    visualizer = DetectionVisualizer()
    cap = cv2.VideoCapture(
        config.application_settings.webcam_id
        if config.application_settings.use_webcam
        else config.application_settings.video_path
    )
    if not cap.isOpened():
        logger.error(VIDEO_SOURCE_ERROR)
        raise ValueError(VIDEO_SOURCE_ERROR)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # perform detection
            results_helmet, results_motorcycle = detects.detect(
                frame,
                config.model_settings.helmet_conf_threshold,
                config.model_settings.motorcycle_conf_threshold,
            )

            # visualizer returns (frame_with_drawings, has_no_helmet)
            _, has_no_helmet = visualizer.draw_detections(
                frame,
                results_helmet,
                results_motorcycle,
                config.detection_visualizer.roi_points,
            )

            motorcycle_count = 0
            try:
                motorcycle_count = (
                    len(results_motorcycle.boxes.xyxy)
                    if hasattr(results_motorcycle, "boxes")
                    else 0
                )
            except Exception:
                try:
                    motorcycle_count = (
                        len(results_motorcycle.boxes)
                        if hasattr(results_motorcycle, "boxes")
                        else 0
                    )
                except Exception:
                    motorcycle_count = 0

            payload = {
                "id": int(time.time() * 1000),
                "timestamp": datetime.now().isoformat(),
                "camera": config.application_settings.video_path
                if not config.application_settings.use_webcam
                else f"webcam:{config.application_settings.webcam_id}",
                "helmet": False if has_no_helmet else True,
                "motorcycle_count": int(motorcycle_count),
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
