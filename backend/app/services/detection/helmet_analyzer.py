"""Helmet stage: ROI geometry, helmet inference, and the violation verdict."""

import logging

import numpy as np
from ultralytics.engine.results import Boxes

from app.configuration import DetectionConfig
from app.models.detection import BoundingBox, DetectionRecord
from app.services.detection.annotate import draw_box, helmet_color

logger = logging.getLogger(__name__)

# A motorcycle seat legally fits a rider plus one pillion
MAX_PASSENGERS = 2


def extract_box_coords(box: Boxes) -> BoundingBox:
    """Extract integer bounding box coordinates from a YOLO box object."""
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def roi_bounds(
    frame_shape: tuple[int, ...], moto_box: BoundingBox, pad_filter: int
) -> tuple[int, int, int, int]:
    """Padded ROI around a motorcycle as (x1, y1, x2, y2).

    Heads ride above the motorcycle bbox, so the top pad is the largest and
    the bottom the smallest. This single function is the only place the pad
    geometry is defined.
    """
    height, width = frame_shape[:2]
    box_width = moto_box.x2 - moto_box.x1
    box_height = moto_box.y2 - moto_box.y1
    horizontal_pad = max(pad_filter * 2, box_width * 2)
    top_pad = max(pad_filter * 3, box_height * 3)
    bottom_pad = max(pad_filter, box_height)
    return (
        max(0, moto_box.x1 - horizontal_pad),
        max(0, moto_box.y1 - top_pad),
        min(width, moto_box.x2 + horizontal_pad),
        min(height, moto_box.y2 + bottom_pad),
    )


def contains_center(
    bounds: tuple[int, int, int, int], box: BoundingBox
) -> bool:
    """True when a box's center point lies inside the (x1, y1, x2, y2) bounds."""
    x1, y1, x2, y2 = bounds
    return x1 <= box.center_x <= x2 and y1 <= box.center_y <= y2


def classify(
    labels: list[str], on_label: str
) -> tuple[bool, bool, bool]:
    """Reduce helmet labels to (helmet_status, over_capacity, violation).

    No labels means nobody was detected: not compliant, but also no proof of a
    violation, so the record reads NOT_DETECTED.
    """
    helmet_status = all(label == on_label for label in labels) if labels else False
    violation = bool(labels) and not helmet_status
    return helmet_status, len(labels) > MAX_PASSENGERS, violation


class HelmetAnalyzer:
    """Runs the helmet model on a motorcycle ROI and builds the record."""

    def __init__(self, model: object, config: DetectionConfig, device: str) -> None:
        """Bind the helmet model, its config, and the inference device.

        Args:
            model: Loaded helmet YOLO model (any exported format)
            config: Detection configuration
            device: Torch-style device string passed through to inference
        """
        self._model = model
        self._config = config
        self._device = device

    def analyze(
        self, frame: np.ndarray, moto_box: BoundingBox, track_id: int
    ) -> DetectionRecord:
        """Detect helmets around one motorcycle and classify the outcome.

        Args:
            frame: Current full frame
            moto_box: Bounding box of the tracked motorcycle
            track_id: Motorcycle track ID

        Returns:
            DetectionRecord with passenger count and violation verdict
        """
        record = DetectionRecord(
            motorcycle_track_id=track_id,
            helmet_status=False,
            passenger_count=0,
            over_capacity=False,
            violation=False,
        )

        x1, y1, x2, y2 = roi_bounds(frame.shape, moto_box, self._config.pad_filter)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning(f"Empty motorcycle crop for track {track_id}, skipping")
            return record

        try:
            result = self._model(
                crop,
                conf=self._config.helmet_confidence,
                imgsz=self._config.helmet_imgsz,
                verbose=False,
                device=self._device,
            )[0]
        except Exception as e:
            logger.error(f"Helmet detection failed for track {track_id}: {e}")
            record.violation = True
            return record

        helmet_labels: list[str] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for hbox in result.boxes:
                helmet_box = extract_box_coords(hbox)
                helmet_box = BoundingBox(
                    x1=helmet_box.x1 + x1,
                    y1=helmet_box.y1 + y1,
                    x2=helmet_box.x2 + x1,
                    y2=helmet_box.y2 + y1,
                )
                if not contains_center((x1, y1, x2, y2), helmet_box):
                    continue

                label = result.names[int(hbox.cls.item())]
                helmet_labels.append(label)
                draw_box(frame, helmet_box, helmet_color(label, self._config.helmet_on_label), label)

        record.passenger_count = len(helmet_labels)
        record.helmet_status, record.over_capacity, record.violation = classify(
            helmet_labels, self._config.helmet_on_label
        )

        logger.info(
            f"Detection ID:{track_id} | "
            f"Helmets:{helmet_labels} | "
            f"Status:{'OK' if record.helmet_status else 'VIOLATION' if record.violation else 'NOT_DETECTED'}"
        )
        return record
