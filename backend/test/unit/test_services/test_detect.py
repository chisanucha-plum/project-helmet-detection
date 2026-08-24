"""Unit tests for detection service."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app.configuration import DetectionConfig
from app.models.detection import BoundingBox, DetectionRecord
from app.services.detect import DetectionService


def create_mock_config() -> DetectionConfig:
    """Create a DetectionConfig with realistic test values."""
    config = Mock(spec=DetectionConfig)
    config.pad_filter = 80
    config.helmet_detect_confidence = 0.20
    config.helmet_detect_imgsz = 1280
    config.motorcycle_confidence = 0.5
    config.line_position_percent = 0.5
    config.line_overlay_alpha = 0.5
    return config


def make_service() -> DetectionService:
    """Create a DetectionService with mocked YOLO model loading."""
    with patch("app.services.detect.YOLO"), patch(
        "app.services.detect.Path.exists", return_value=True
    ):
        return DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=create_mock_config(),
        )


class TestBoundingBoxExtraction:
    """Test bounding box coordinate extraction."""

    def test_extract_box_coords(self):
        """Coordinates are extracted from YOLO xyxy array as BoundingBox."""
        service = make_service()

        mock_box = Mock()
        mock_box.xyxy = [np.array([10, 20, 100, 150])]

        result = service._extract_box_coords(mock_box)

        assert isinstance(result, BoundingBox)
        assert result.x1 == 10
        assert result.y1 == 20
        assert result.x2 == 100
        assert result.y2 == 150


class TestDetectionRecordGeneration:
    """Test detection record generation from track data."""

    def test_detection_record_creation_helmet_present(self):
        """Test creating detection record when helmet is present."""
        record = DetectionRecord(
            motorcycle_track_id=1,
            helmet_status=True,
            passenger_count=1,
            over_capacity=False,
            violation=False,
        )

        assert record.motorcycle_track_id == 1
        assert record.helmet_status is True
        assert record.violation is False

    def test_detection_record_creation_helmet_missing(self):
        """Test creating detection record when helmet is missing."""
        record = DetectionRecord(
            motorcycle_track_id=2,
            helmet_status=False,
            passenger_count=1,
            over_capacity=False,
            violation=True,
        )

        assert record.motorcycle_track_id == 2
        assert record.helmet_status is False
        assert record.violation is True

    def test_detection_record_with_multiple_passengers(self):
        """Test detection record with multiple passengers."""
        record = DetectionRecord(
            motorcycle_track_id=3,
            helmet_status=True,
            passenger_count=3,
            over_capacity=True,
            violation=False,
        )

        assert record.passenger_count == 3
        assert record.over_capacity is True


class TestTrackReset:
    """Test track reset functionality."""

    def test_reset_tracks_clears_state(self):
        """Reset clears both track history and counted IDs."""
        service = make_service()
        service._track_history = {1: 100, 2: 150}
        service._counted_ids = {1, 2}

        service.reset_tracks()

        assert service._track_history == {}
        assert service._counted_ids == set()


class TestHelmetDetectionLogic:
    """Test helmet proximity detection logic."""

    def test_helmet_proximity_check_near(self):
        """Helmet center inside padded motorcycle area returns True."""
        service = make_service()
        helmet_box = BoundingBox(x1=60, y1=40, x2=90, y2=70)  # center (75, 55)
        moto_box = BoundingBox(x1=50, y1=50, x2=150, y2=150)

        assert service._is_helmet_near_motorcycle(helmet_box, moto_box) is True

    def test_helmet_proximity_check_far(self):
        """Helmet center outside padded motorcycle area returns False."""
        service = make_service()
        helmet_box = BoundingBox(x1=300, y1=300, x2=350, y2=350)  # center (325, 325)
        moto_box = BoundingBox(x1=50, y1=50, x2=150, y2=150)

        assert service._is_helmet_near_motorcycle(helmet_box, moto_box) is False


class TestLineCrossing:
    """Test motorcycle line crossing detection."""

    def test_cross_from_right_to_left_detected(self):
        """Moving from right of line to left of line counts as crossing."""
        service = make_service()
        service._line_x = 320
        service._track_history[1] = 400  # previously right of line

        assert service._has_crossed_line(track_id=1, center_x=300) is True

    def test_no_history_returns_false(self):
        """First sighting of a track never counts as crossing."""
        service = make_service()
        service._line_x = 320

        assert service._has_crossed_line(track_id=1, center_x=300) is False

    def test_already_counted_returns_false(self):
        """A track already counted is not reported again."""
        service = make_service()
        service._line_x = 320
        service._track_history[1] = 400
        service._counted_ids.add(1)

        assert service._has_crossed_line(track_id=1, center_x=300) is False

    def test_moving_left_to_right_not_crossed(self):
        """Crossing direction right-to-left only; left-to-right is ignored."""
        service = make_service()
        service._line_x = 320
        service._track_history[2] = 300  # previously left of line

        assert service._has_crossed_line(track_id=2, center_x=400) is False

    def test_staying_right_of_line_not_crossed(self):
        """Movement entirely on the right side is not a crossing."""
        service = make_service()
        service._line_x = 320
        service._track_history[3] = 400

        assert service._has_crossed_line(track_id=3, center_x=350) is False


class TestFrameDrawing:
    """Test frame drawing utilities."""

    @patch("app.services.detect.cv2.rectangle")
    def test_draw_box_with_frame(self, mock_rectangle):
        """Drawing a box calls cv2.rectangle with box coordinates."""
        service = make_service()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)

        service._draw_box(frame, bbox, (0, 255, 0), "Helmet")

        assert mock_rectangle.called

    @patch("app.services.detect.cv2.line")
    def test_draw_detection_line_on_frame(self, mock_line):
        """Drawing the detection line calls cv2.line at line_x."""
        service = make_service()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        service._line_x = 320

        service._draw_detection_line(frame, height=240)

        assert mock_line.called


class TestDetectionServiceInitialization:
    """Test DetectionService initialization."""

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_service_initialization_with_models(self, mock_exists, mock_yolo):
        """Both YOLO models are loaded once each."""
        mock_motorcycle_model = Mock()
        mock_helmet_model = Mock()
        mock_yolo.side_effect = [mock_motorcycle_model, mock_helmet_model]

        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=create_mock_config(),
        )

        assert service._moto_model is mock_motorcycle_model
        assert service._helmet_model is mock_helmet_model
        assert mock_yolo.call_count == 2

    def test_service_initialization_empty_tracks(self):
        """New service starts with empty tracking state."""
        service = make_service()

        assert service._track_history == {}
        assert service._counted_ids == set()
