import json
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self):
        return asdict(self)


@dataclass
class PersonDetection:
    id: Optional[int]  # optional id assigned by tracker (if any)
    bbox: BBox
    center_x: float
    center_y: float
    width: int
    height: int
    in_roi: bool

    def to_dict(self):
        d = asdict(self)
        d["bbox"] = self.bbox.to_dict()
        return d


@dataclass
class HelmetDetection:
    id: Optional[int]
    bbox: BBox
    confidence: float
    cls: int  # class id from the model (e.g., 0=no_helmet, 1=helmet)
    label: str
    in_roi: bool

    def to_dict(self):
        d = asdict(self)
        d["bbox"] = self.bbox.to_dict()
        return d


@dataclass
class FrameDetections:
    timestamp: str
    total_persons: int
    total_helmets: int
    helmets_off_count: int
    persons: List[PersonDetection]
    helmets: List[HelmetDetection]

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "total_persons": self.total_persons,
            "total_helmets": self.total_helmets,
            "helmets_off_count": self.helmets_off_count,
            "persons": [p.to_dict() for p in self.persons],
            "helmets": [h.to_dict() for h in self.helmets],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


# Example JSON schema produced by FrameDetections.to_dict():
# {
#   "timestamp": "2025-09-15 12:34:56",
#   "total_persons": 3,
#   "total_helmets": 2,
#   "helmets_off_count": 1,
#   "persons": [
#     {"id": 1, "bbox": {"x1": 10, "y1": 20, "x2": 110, "y2": 220}, "center_x": 60.0, "center_y": 120.0, "width": 100, "height": 200, "in_roi": true}
#   ],
#   "helmets": [
#     {"id": null, "bbox": {"x1": 10, "y1": 20, "x2": 110, "y2": 220}, "confidence": 0.95, "cls": 1, "label": "Helmet", "in_roi": true}
#   ]
# }
