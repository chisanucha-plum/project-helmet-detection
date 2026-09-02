"""Helmet detection pipeline package.

Split by responsibility:
- service: model loading + motorcycle tracking loop
- line_counter: line-crossing state machine
- helmet_analyzer: ROI geometry, helmet inference, verdict
- annotate: OpenCV drawing helpers
"""

from app.services.detection.service import DetectionService

__all__ = ["DetectionService"]
