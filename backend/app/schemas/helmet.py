from typing import Optional

from pydantic import BaseModel


class HelmetDetectionResponse(BaseModel):
    """Response model for helmet detection"""

    id: int
    helmet_detected: bool
    motorcycle_detected: bool
    no_helmet_in_roi: bool
    timestamp: str
    message: Optional[str] = None

    class Config:
        from_attributes = True


class DetectionStatsResponse(BaseModel):
    """Response model for detection statistics"""

    total_detections: int
    violations: int
    helmet_detected: int
    motorcycle_detected: int
    compliance_rate: float
    date_filter: Optional[str] = None


class ViolationDetail(BaseModel):
    """Model for violation details"""

    detection_type: str
    confidence: float
    bbox: list
    timestamp: str


class HistoryStatusResponse(BaseModel):
    """Response model for history status with violation details"""

    id: str
    helmet_status: Optional[bool]
    passenger_count: Optional[int]
    violations: Optional[list] = None
    timestamp: Optional[str]
