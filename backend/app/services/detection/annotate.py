"""OpenCV drawing helpers for annotated stream frames."""

import cv2
import numpy as np

from app.models.detection import BoundingBox

MOTO_COLOR = (255, 0, 0)
HELMET_ON_COLOR = (0, 255, 0)
HELMET_OFF_COLOR = (0, 0, 255)


def draw_box(
    frame: np.ndarray,
    box: BoundingBox,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a bounding box and its label on the frame (modified in-place).

    Args:
        frame: Frame to draw on
        box: Bounding box coordinates
        color: BGR color tuple
        label: Text drawn just above the box
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


def draw_detection_line(frame: np.ndarray, line_x: int, alpha: float) -> None:
    """Draw a semi-transparent vertical detection line (modified in-place).

    Args:
        frame: Frame to draw on
        line_x: Fixed x-position of the line
        alpha: Blend strength of the line overlay (0-1)
    """
    height, width = frame.shape[:2]
    thickness = 3
    half = thickness // 2
    x1 = max(0, line_x - half)
    x2 = min(width, line_x + half + 1)
    if x2 <= x1:
        return
    sub = frame[:, x1:x2]
    overlay = sub.copy()
    cv2.line(overlay, (line_x - x1, 0), (line_x - x1, height), (255, 0, 0), thickness)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)



def helmet_color(
    label: str, on_label: str
) -> tuple[int, int, int]:
    """Green for helmets-on, red for anything else."""
    return HELMET_ON_COLOR if label == on_label else HELMET_OFF_COLOR
