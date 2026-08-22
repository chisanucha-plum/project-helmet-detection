"""Integration test for video upload and detection via web API."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add backend path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))
os.chdir(str(backend_path))

import cv2
import numpy as np
from app.models.detection import DetectionRecord


def create_test_video(video_path: Path, num_frames: int = 3) -> Path:
    """Create a short test video file."""
    width, height = 640, 480
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [50 + i * 10, 100, 150]
        cv2.putText(frame, f"Frame {i+1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    return video_path


def test_detection_record_json_serialization():
    """Test detection record serializes to JSON."""
    record = DetectionRecord(
        motorcycle_track_id=1,
        helmet_status=True,
        passenger_count=1,
        over_capacity=False,
        violation=False,
    )
    
    data = record.to_dict()
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    
    assert parsed["motorcycle_track_id"] == 1
    assert parsed["helmet_status"] is True
    assert parsed["violation"] is False


def test_detection_record_construction_from_json_data():
    """Test creating detection record from JSON-like dict via constructor."""
    json_data = {
        "motorcycle_track_id": 2,
        "helmet_status": False,
        "passenger_count": 1,
        "over_capacity": False,
        "violation": True,
    }

    record = DetectionRecord(**json_data)

    assert record.motorcycle_track_id == 2
    assert record.helmet_status is False
    assert record.violation is True


def test_video_file_creation():
    """Test test video file can be created."""
    test_video = Path("test_short.mp4")
    
    try:
        video_path = create_test_video(test_video, num_frames=2)
        
        assert video_path.exists()
        assert video_path.stat().st_size > 0
        
        # Verify video is readable
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened()
        
        ret, frame = cap.read()
        assert ret
        assert frame.shape == (480, 640, 3)
        
        cap.release()
    finally:
        if test_video.exists():
            test_video.unlink()


@patch("app.services.camera_hub.DetectionService")
def test_camera_hub_subscription(mock_service):
    """Test CameraHub frame subscription."""
    from app.services.camera_hub import CameraHub
    
    hub = CameraHub()
    
    # Subscribe
    q = hub.subscribe_frames()
    assert q is not None
    
    # Unsubscribe
    hub.unsubscribe_frames(q)


@patch("app.services.camera_hub.DetectionService")
def test_camera_hub_detection_subscription(mock_service):
    """Test CameraHub detection subscription."""
    from app.services.camera_hub import CameraHub
    
    hub = CameraHub()
    
    # Subscribe
    q = hub.subscribe_detections()
    assert q is not None
    
    # Unsubscribe
    hub.unsubscribe_detections(q)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
