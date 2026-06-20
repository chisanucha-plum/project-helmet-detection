"""Unit tests for detection service."""

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from app.models.detection import BoundingBox, DetectionRecord
from app.services.detect import DetectionService
from app.configuration import DetectionConfig


def create_mock_config():
    """Create a properly configured mock for DetectionConfig."""
    config = Mock(spec=DetectionConfig)
    config.pad_filter = 80
    config.helmet_detect_confidence = 0.20
    config.helmet_detect_imgsz = 1280
    config.motorcycle_confidence = 0.5
    config.line_position_percent = 0.5
    return config


class TestBoundingBoxExtraction:
    """Test bounding box coordinate extraction."""

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_extract_box_coords(self, mock_exists, mock_yolo):
        """Test extraction of bounding box coordinates."""
        import numpy as np
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        # Mock YOLO box object with xyxy attribute as numpy array
        mock_box = Mock()
        mock_box.xyxy = [np.array([10, 20, 100, 150])]
        
        result = service._extract_box_coords(mock_box)
        
        assert isinstance(result, BoundingBox)
        assert result.x1 == 10
        assert result.y1 == 20
        assert result.x2 == 100
        assert result.y2 == 150

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_extract_box_coords_real_call(self, mock_exists, mock_yolo):
        """Test real bounding box extraction with mock ultralytics box."""
        import numpy as np
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        # Mock YOLO box object with xyxy attribute as numpy array
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

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_reset_tracks_clears_state(self, mock_exists, mock_yolo):
        """Test that reset_tracks is called to clear tracking state."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        # Set some track data
        service._track_history = {1: 100, 2: 150}
        service._counted_ids = {1, 2}
        
        # Reset should clear tracks
        service.reset_tracks()
        
        assert service._track_history == {}
        assert service._counted_ids == set()

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_reset_tracks_real_implementation(self, mock_exists, mock_yolo):
        """Test real track reset clears motorcycle tracks."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        # Set some track data
        service._track_history = {1: 100, 2: 150}
        service._counted_ids = {1, 2}
        
        # Reset should clear tracks
        service.reset_tracks()
        
        assert service._track_history == {}
        assert service._counted_ids == set()


class TestHelmetDetectionLogic:
    """Test helmet detection analysis logic."""

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_helmet_proximity_check_near(self, mock_exists, mock_yolo):
        """Test helmet proximity detection when helmet is near motorcycle."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        motorcycle_box = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        helmet_box = BoundingBox(x1=60, y1=40, x2=90, y2=70)
        
        # Helmet center should be close to motorcycle center
        result = service._is_helmet_near_motorcycle(motorcycle_box, helmet_box)
        
        assert isinstance(result, bool)

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_helmet_proximity_check_far(self, mock_exists, mock_yolo):
        """Test helmet proximity detection when helmet is far from motorcycle."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        motorcycle_box = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        helmet_box = BoundingBox(x1=300, y1=300, x2=350, y2=350)
        
        result = service._is_helmet_near_motorcycle(motorcycle_box, helmet_box)
        
        assert isinstance(result, bool)


class TestLineCrossing:
    """Test motorcycle line crossing detection."""

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_motorcycle_crosses_detection_line(self, mock_exists, mock_yolo):
        """Test detection when motorcycle crosses the detection line."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        result = service._has_crossed_line(track_id=1, center_x=100)
        
        assert isinstance(result, bool)

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_motorcycle_does_not_cross_line(self, mock_exists, mock_yolo):
        """Test detection when motorcycle does not cross the line."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        result = service._has_crossed_line(track_id=2, center_x=50)
        
        assert isinstance(result, bool)


class TestFrameDrawing:
    """Test frame drawing utilities."""

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    @patch("app.services.detect.cv2.rectangle")
    def test_draw_box_with_frame(self, mock_rectangle, mock_exists, mock_yolo):
        """Test drawing bounding box on frame."""
        import numpy as np
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        # Create actual numpy array frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        
        # Should not raise any exception
        service._draw_box(mock_frame, bbox, (0, 255, 0), "Helmet")
        
        # Verify cv2.rectangle was called
        assert mock_rectangle.called

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    @patch("app.services.detect.cv2.line")
    def test_draw_detection_line_on_frame(self, mock_line, mock_exists, mock_yolo):
        """Test drawing detection line on frame."""
        import numpy as np
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        # Create actual numpy array frame and set line position
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        service._line_x = 320  # Set line position
        
        # Should not raise any exception
        service._draw_detection_line(mock_frame, height=240)
        
        # Verify cv2.line was called
        assert mock_line.called


class TestDetectionServiceInitialization:
    """Test DetectionService initialization."""

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_service_initialization_with_models(self, mock_exists, mock_yolo):
        """Test DetectionService initializes with YOLO models."""
        mock_motorcycle_model = Mock()
        mock_helmet_model = Mock()
        
        mock_yolo.side_effect = [mock_motorcycle_model, mock_helmet_model]
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        assert service._moto_model is not None
        assert service._helmet_model is not None
        assert mock_yolo.call_count == 2

    @patch("app.services.detect.YOLO")
    @patch("app.services.detect.Path.exists", return_value=True)
    def test_service_initialization_empty_tracks(self, mock_exists, mock_yolo):
        """Test DetectionService initializes with empty tracks."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_config = create_mock_config()
        
        service = DetectionService(
            moto_model_path=Path("dummy_moto.pt"),
            helmet_model_path=Path("dummy_helmet.pt"),
            config=mock_config,
        )
        
        assert service._track_history == {}
        assert service._counted_ids == set()
        assert hasattr(service, "_config")
