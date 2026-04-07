import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
from app.configuration import Configuration
from app.database.analysis_job import AnalysisJob
from app.database.database import SessionLocal, get_db
from app.schemas.helmet import (
    HistoryStatusResponse,
)
from app.services.detect import ObjectDetect
from app.services.helmet_service import HelmetService
from app.services.visualizer import DetectionVisualizer
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Helmet Detection"])

# Constants
VIDEO_SOURCE_ERROR = "Cannot open video source"
SNAPSHOT_DIR = "snapshots"
COOLDOWN_SECONDS = 20.0  # Prevent duplicate captures within 20 seconds

# Global object tracking
tracked_objects: Dict[str, float] = {}  # object_id -> last_seen_time
# Simple in-memory tracker to assign stable IDs across frames
# track_id -> (center_x, center_y, last_seen_time)
recent_tracks: Dict[str, tuple] = {}
# Next numeric id for new tracks
next_track_id = 1
# Matching params
TRACK_MATCH_DISTANCE = 75.0  # pixels
TRACK_MAX_AGE = 5.0  # seconds before we consider a track stale


def is_motorcycle_in_center_roi(bbox, roi_points) -> bool:
    """
    Check if motorcycle is in the center area of ROI (not just touching edges).

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        roi_points: List of ROI polygon points

    Returns:
        bool: True if motorcycle center is within 26% of ROI center area
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

    # Define center area as 26% of ROI size around center
    center_threshold_x = roi_width * 0.13  # 13% on each side = 26% total
    center_threshold_y = roi_height * 0.13

    # Check if motorcycle center is within the threshold
    distance_from_center_x = abs(motorcycle_center[0] - roi_center_x)
    distance_from_center_y = abs(motorcycle_center[1] - roi_center_y)

    return (
        distance_from_center_x <= center_threshold_x
        and distance_from_center_y <= center_threshold_y
    )


def calculate_motorcycle_id(bbox) -> str:
    """
    Generate motorcycle ID based on position grid.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        str: Motorcycle ID in format 'mc_x_y'
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # Create ID based on 85x85 grid for stable tracking
    grid_x = center_x // 85
    grid_y = center_y // 85

    return f"mc_{grid_x}_{grid_y}"


def get_track_id_from_bbox(bbox) -> str:
    """
    Assign or retrieve a stable track id for a bounding box using nearest-neighbor
    matching against recent tracks.

    Returns a string track id like 'trk_1'.
    """
    global next_track_id
    # compute centroid
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    now_ts = time.time()

    # remove stale tracks
    stale = [
        tid for tid, (_, _, t) in recent_tracks.items() if (now_ts - t) > TRACK_MAX_AGE
    ]
    for tid in stale:
        recent_tracks.pop(tid, None)

    # find nearest track
    best_id = None
    best_dist = None
    for tid, (tx, ty, t) in recent_tracks.items():
        dx = tx - center_x
        dy = ty - center_y
        dist = (dx * dx + dy * dy) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = tid

    if (
        best_id is not None
        and best_dist is not None
        and best_dist <= TRACK_MATCH_DISTANCE
    ):
        # update track centroid/time
        recent_tracks[best_id] = (center_x, center_y, now_ts)
        return best_id

    # create new track
    tid = f"trk_{next_track_id}"
    next_track_id += 1
    recent_tracks[tid] = (center_x, center_y, now_ts)
    return tid


def should_capture_object(object_id: str) -> bool:
    """
    Check if object should be captured based on cooldown timer.

    Args:
        object_id: Unique identifier for the motorcycle

    Returns:
        bool: True if capture is allowed, False if in cooldown period
    """
    current_time = time.time()

    if object_id in tracked_objects:
        time_since_last_capture = current_time - tracked_objects[object_id]
        if time_since_last_capture < COOLDOWN_SECONDS:
            return False  # Still in cooldown period

    # Update last capture time and allow capture
    tracked_objects[object_id] = current_time
    return True


def crop_roi_area(frame, roi_points):
    """
    Crop frame to ROI area with padding to save storage space.

    Args:
        frame: Input frame to crop
        roi_points: ROI polygon points

    Returns:
        Cropped frame or original frame if no ROI defined
    """
    if not roi_points or len(roi_points) < 3:
        return frame

    # Calculate ROI bounding rectangle
    x_coords = [point[0] for point in roi_points]
    y_coords = [point[1] for point in roi_points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add 5% padding
    padding_x = int((x_max - x_min) * 0.05)
    padding_y = int((y_max - y_min) * 0.05)

    # Ensure bounds are within frame dimensions
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
    Capture frame when motorcycle is in center of ROI with violation detected.

    Args:
        frame: Current video frame
        results_motorcycle: Motorcycle detection results
        roi_points: ROI polygon points
        has_no_helmet: Whether helmet violation was detected

    Returns:
        bool: True if frame was captured, False otherwise
    """
    if not hasattr(results_motorcycle, "boxes") or len(results_motorcycle.boxes) == 0:
        return False

    # Ensure snapshots directory exists
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    for i, bbox in enumerate(results_motorcycle.boxes.xyxy):
        bbox_np = bbox.cpu().numpy()

        # Check if motorcycle is in CENTER of ROI (not just inside ROI)
        if is_motorcycle_in_center_roi(bbox_np, roi_points):
            # Generate stable track ID (nearest-neighbor tracker) instead of grid ID
            motorcycle_id = get_track_id_from_bbox(bbox_np)

            # Check cooldown to prevent duplicate captures
            if should_capture_object(motorcycle_id):
                # Timestamp formatted like 20251001_101711_166 (milliseconds precision)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # Save filename with only the timestamp wrapped by underscores
                # Example: _20251001_101711_166_.jpg
                filename = f"_{timestamp}_.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)

                # Crop to ROI area to save storage space
                # เฉพาะพื้นที่ ROI เพื่อประหยัดเนื้อที่
                cropped_frame = crop_roi_area(frame, roi_points)

                # Save cropped frame (always try to save snapshot file first)
                cv2.imwrite(filepath, cropped_frame)

                # Enqueue analysis job in DB so worker will pick it up.
                # Use context-managed session and clearer logging on failure.
                try:
                    with SessionLocal() as db:
                        job = AnalysisJob(
                            image_path=filepath,
                            status="queued",
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        db.add(job)
                        db.commit()
                        db.refresh(job)
                        logger.info(f"Enqueued analysis job id={job.id} for {filepath}")
                except Exception:
                    # Log full exception (stack trace) so debugging is easier.
                    logger.exception(f"Failed to enqueue analysis job for {filepath}")
                    # Note: we deliberately do NOT re-raise here because
                    # snapshot saving succeeded and we don't want a DB
                    # enqueue failure to interrupt the capture loop.
                return True

    return False


async def generate_frames():
    """Generate frames for REST API streaming (existing method)"""
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

        # ใช้ asyncio.to_thread เพื่อไม่ block event loop
        results_helmet, results_motorcycle = await asyncio.to_thread(
            detects.detect,
            frame,
            config.model_settings.helmet_conf_threshold,
            config.model_settings.motorcycle_conf_threshold,
        )

        # เลือก ROI ตาม input source (webcam หรือ video)
        roi_points = config.detection_visualizer.get_roi_points(
            config.application_settings.use_webcam
        )

        # ใช้ asyncio.to_thread เพื่อไม่ block event loop
        frame, has_no_helmet = await asyncio.to_thread(
            visualizer.draw_detections,
            frame,
            results_helmet,
            results_motorcycle,
            roi_points,
        )

        # Capture frame when new motorcycle enters ROI (non-blocking)
        await asyncio.to_thread(
            capture_frame_on_roi_entry,
            frame,
            results_motorcycle,
            roi_points,
            has_no_helmet,
        )

        # encode frame as jpeg
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@router.get("/detect", status_code=status.HTTP_200_OK)
async def helmet_detection_stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/history", response_model=List[HistoryStatusResponse])
async def get_history_records(
    limit: int = Query(default=50, ge=1, le=500), db: Session = Depends(get_db)
):
    """Get history status records with violation details"""
    history = HelmetService.get_history_with_details(db=db, limit=limit)
    return history
