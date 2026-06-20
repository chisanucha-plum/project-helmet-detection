from typing import Optional
from pydantic import BaseModel


class HistoryStatusResponse(BaseModel):
    """API response for a single detection history record.

    Contains the detection results of a motorcycle that crossed the
    detection line, including helmet status and passenger information.
    """

    id: str  # Unique record ID
    track_id: Optional[int] = None  # Motorcycle track ID
    helmet_status: Optional[bool] = None  # True if helmet detected
    passenger_count: Optional[int] = None  # Number of passengers
    over_capacity: Optional[bool] = None  # True if >2 passengers
    violation: Optional[bool] = None  # True if helmet missing
    timestamp: Optional[str] = None  # Detection timestamp
    frame_path: Optional[str] = None  # Path to saved frame image

    class Config:
        from_attributes = True
