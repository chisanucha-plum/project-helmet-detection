import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from app.configuration import DetectionConfig
from app.models.detection import BoundingBox, DetectionRecord

logger = logging.getLogger(__name__)


class DetectionService:
    """Performs two-stage detection: motorcycle tracking and helmet analysis.

    This service uses YOLOv8 to:
    1. Track motorcycles crossing a reference line (50% of frame width)
    2. Detect helmets on riders when motorcycle crosses the line

    The detection line is drawn at 50% of frame width. When a motorcycle crosses
    from right to left, helmet detection is performed and recorded.
    """

    def __init__(
        self,
        moto_model_path: Path,
        helmet_model_path: Path,
        config: DetectionConfig,
    ) -> None:
        """Initialize detection service with models and configuration.

        Args:
            moto_model_path: Path to motorcycle detection model (YOLOv8)
            helmet_model_path: Path to helmet detection model (YOLOv8)
            config: Detection configuration (thresholds, padding, etc.)

        Raises:
            FileNotFoundError: If model files do not exist
            OSError: If CUDA is available but fails to initialize
        """
        if not Path(moto_model_path).exists():
            raise FileNotFoundError(f"Motorcycle model not found: {moto_model_path}")
        if not Path(helmet_model_path).exists():
            raise FileNotFoundError(f"Helmet model not found: {helmet_model_path}")

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._moto_model: YOLO = YOLO(str(moto_model_path))
        self._helmet_model: YOLO = YOLO(str(helmet_model_path))
        self._config: DetectionConfig = config

        # Track state
        self._track_history: dict[int, int] = {}  # {track_id: last_center_x}
        self._counted_ids: set[int] = set()  # Track IDs that crossed the line
        self._line_x: int | None = None  # Detection line position

        logger.info(
            "DetectionService initialized",
            extra={
                "device": self._device,
                "pad_filter": config.pad_filter,
                "line_position_percent": config.line_position_percent,
            },
        )

    def detect_and_track(
        self, frame: np.ndarray, conf: float | None = None
    ) -> tuple[np.ndarray, list[DetectionRecord] | None]:
        """Detect motorcycles and analyze helmets in a frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            conf: Confidence threshold (uses config default if None)

        Returns:
            Tuple of (annotated_frame, detection_records)
            - annotated_frame: Frame with drawn boxes and line
            - detection_records: List of DetectionRecord objects

        Raises:
            ValueError: If frame is invalid or empty
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame provided")

        if conf is None:
            conf = self._config.motorcycle_confidence

        new_records: list[DetectionRecord] = []
        h, w = frame.shape[:2]

        # Set LINE_X on first frame
        if self._line_x is None:
            self._line_x = int(w * self._config.line_position_percent)
            logger.info(
                f"Detection line set to x={self._line_x} "
                f"({self._config.line_position_percent * 100:.0f}% of width {w})"
            )

        # Stage 1: Track motorcycles
        new_records.extend(self._process_motorcycle_tracks(frame, conf))

        # Draw detection line
        self._draw_detection_line(frame, h)

        return frame, new_records

    def _process_motorcycle_tracks(
        self, frame: np.ndarray, conf: float
    ) -> list[DetectionRecord]:
        """Process motorcycle tracking and helmet detection for crossed motorcycles.

        Args:
            frame: Current frame
            conf: Confidence threshold

        Returns:
            List of DetectionRecord objects for motorcycles crossing the line
        """
        records: list[DetectionRecord] = []

        try:
            result = self._moto_model.track(
                frame,
                conf=conf,
                persist=True,
                tracker=self._config.tracker_name,
                classes=[self._config.motorcycle_class_id],
                verbose=False,
                device=self._device,
            )[0]
        except Exception as e:
            logger.error(f"Motorcycle tracking failed: {e}")
            return records

        if result.boxes is None or len(result.boxes) == 0:
            return records

        # Process each detected motorcycle
        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            moto_box = self._extract_box_coords(box)
            center_x = moto_box.center_x

            # Draw motorcycle detection box
            self._draw_box(frame, moto_box, (255, 0, 0), f"ID:{track_id}")

            # Check if motorcycle crossed detection line
            if self._has_crossed_line(track_id, center_x):
                self._counted_ids.add(track_id)
                logger.info(
                    f"Motorcycle ID:{track_id} crossed detection line at x={center_x}"
                )

                # Stage 2: Analyze helmets for this motorcycle
                record = self._analyze_helmets(frame, moto_box, track_id)
                records.append(record)

            # Update track history
            self._track_history[track_id] = center_x

        return records

    def _has_crossed_line(self, track_id: int, center_x: int) -> bool:
        """Check if motorcycle crossed detection line from right to left.

        Args:
            track_id: Motorcycle track ID
            center_x: Current horizontal center position

        Returns:
            True if motorcycle just crossed from right to left
        """
        if track_id in self._counted_ids:
            return False

        prev_center_x = self._track_history.get(track_id)
        if prev_center_x is None:
            return False

        # Crossed if was on right (>line_x) and now on left (<=line_x)
        return prev_center_x > self._line_x and center_x <= self._line_x

    def _analyze_helmets(
        self, frame: np.ndarray, moto_box: BoundingBox, track_id: int
    ) -> DetectionRecord:
        """Detect and classify helmets for a motorcycle that crossed the line.

        Args:
            frame: Current frame
            moto_box: Bounding box of motorcycle
            track_id: Motorcycle track ID

        Returns:
            DetectionRecord with helmet status and passenger count
        """
        record = DetectionRecord(
            motorcycle_track_id=track_id,
            helmet_status=False,
            passenger_count=0,
            over_capacity=False,
            violation=False,
        )

        try:
            helmet_result = self._helmet_model(
                frame,
                conf=self._config.helmet_detect_confidence,
                imgsz=self._config.helmet_detect_imgsz,
                verbose=False,
            )[0]
        except Exception as e:
            logger.error(f"Helmet detection failed for track {track_id}: {e}")
            record.violation = True
            return record

        helmet_labels: list[str] = []

        if helmet_result.boxes and len(helmet_result.boxes) > 0:
            for hbox in helmet_result.boxes:
                helmet_box = self._extract_box_coords(hbox)

                # Check if helmet is near motorcycle
                if self._is_helmet_near_motorcycle(helmet_box, moto_box):
                    label = helmet_result.names[int(hbox.cls.item())]
                    helmet_labels.append(label)

                    # Draw helmet detection box with color coding
                    color = (
                        (0, 255, 0)
                        if label == self._config.helmet_on_label
                        else (0, 0, 255)
                    )
                    self._draw_box(frame, helmet_box, color, label)

        # Analyze helmet status
        record.passenger_count = len(helmet_labels)
        record.over_capacity = record.passenger_count > 2
        record.helmet_status = (
            all(l == self._config.helmet_on_label for l in helmet_labels)
            if helmet_labels
            else False
        )
        record.violation = len(helmet_labels) > 0 and not record.helmet_status

        logger.info(
            f"Detection ID:{track_id} | "
            f"Helmets:{helmet_labels} | "
            f"Status:{'OK' if record.helmet_status else 'VIOLATION' if record.violation else 'NOT_DETECTED'}"
        )

        return record

    def _is_helmet_near_motorcycle(
        self, helmet_box: BoundingBox, moto_box: BoundingBox
    ) -> bool:
        """Check if helmet is near motorcycle using padding filter.

        Args:
            helmet_box: Helmet bounding box
            moto_box: Motorcycle bounding box

        Returns:
            True if helmet center is within padded motorcycle area
        """
        pad = self._config.pad_filter
        return (moto_box.x1 - pad) <= helmet_box.center_x <= (moto_box.x2 + pad) and (
            moto_box.y1 - pad
        ) <= helmet_box.center_y <= (moto_box.y2 + pad)

    def _extract_box_coords(self, box: Boxes) -> BoundingBox:
        """Extract bounding box coordinates from YOLO box object.

        Args:
            box: YOLO detection box

        Returns:
            BoundingBox with integer coordinates
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _draw_box(
        self,
        frame: np.ndarray,
        box: BoundingBox,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw bounding box and label on frame.

        Args:
            frame: Frame to draw on (modified in-place)
            box: Bounding box coordinates
            color: RGB color tuple
            label: Label text
        """
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        cv2.putText(
            frame,
            label,
            (box.x1, box.y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _draw_detection_line(self, frame: np.ndarray, height: int) -> None:
        """Draw semi-transparent detection line on frame.

        Args:
            frame: Frame to draw on (modified in-place)
            height: Frame height
        """
        overlay = frame.copy()
        cv2.line(overlay, (self._line_x, 0), (self._line_x, height), (255, 0, 0), 3)
        cv2.addWeighted(
            overlay,
            self._config.line_overlay_alpha,
            frame,
            1 - self._config.line_overlay_alpha,
            0,
            frame,
        )

    def reset_tracks(self) -> None:
        """Reset track history and counted IDs.

        Call this when switching to a new video or frame sequence.
        """
        self._track_history.clear()
        self._counted_ids.clear()
        logger.debug("Track history reset")
