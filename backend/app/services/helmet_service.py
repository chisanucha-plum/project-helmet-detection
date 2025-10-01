import json
from datetime import datetime
from typing import List

from app.database.history_status import HelmetStatus, HistoryStatus
from sqlalchemy import desc
from sqlalchemy.orm import Session


class HelmetService:
    """Service class for helmet detection data operations"""

    @staticmethod
    def save_helmet_detection(
        db: Session,
        helmet_detected: bool,
        motorcycle_detected: bool,
        no_helmet_in_roi: bool,
        message: str = None,
    ) -> HelmetStatus:
        """Save helmet detection result to database"""
        timestamp = datetime.now().isoformat()

        helmet_status = HelmetStatus(
            helmet_detected=helmet_detected,
            motorcycle_detected=motorcycle_detected,
            no_helmet_in_roi=no_helmet_in_roi,
            timestamp=timestamp,
            message=message,
        )

        db.add(helmet_status)
        db.commit()
        db.refresh(helmet_status)
        return helmet_status

    @staticmethod
    def save_history_status(
        db: Session,
        detection_id: str,
        helmet_status: bool,
        passenger_count: int,
        violations: List[dict] = None,
    ) -> HistoryStatus:
        """Save detection history with detailed information"""
        timestamp = datetime.now().isoformat()
        violations_json = json.dumps(violations) if violations else None

        history = HistoryStatus(
            id=detection_id,
            helmet_status=helmet_status,
            passenger_count=passenger_count,
            violations=violations_json,
            timestamp=timestamp,
        )

        db.add(history)
        db.commit()
        db.refresh(history)
        return history

    @staticmethod
    def get_history_with_details(db: Session, limit: int = 50) -> List[dict]:
        """Get history status with parsed violation details"""
        histories = (
            db.query(HistoryStatus)
            .order_by(desc(HistoryStatus.timestamp))
            .limit(limit)
            .all()
        )

        result = []
        for history in histories:
            violations_data = None
            if history.violations:
                try:
                    violations_data = json.loads(history.violations)
                except json.JSONDecodeError:
                    violations_data = None

            result.append(
                {
                    "id": history.id,
                    "helmet_status": history.helmet_status,
                    "passenger_count": history.passenger_count,
                    "violations": violations_data,
                    "timestamp": history.timestamp,
                }
            )

        return result
