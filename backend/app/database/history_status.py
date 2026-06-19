from app.database.database import Base
from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column


class HistoryStatus(Base):
    """Database model for storing historical motorcycle helmet detection events."""

    __tablename__ = "history_status"
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    helmet_status: Mapped[bool] = mapped_column(Boolean, nullable=True)
    passenger_count: Mapped[int] = mapped_column(Integer, nullable=True)
    over_capacity: Mapped[bool] = mapped_column(Boolean, nullable=True)
    violation: Mapped[bool] = mapped_column(Boolean, nullable=True)
    timestamp: Mapped[str] = mapped_column(String, nullable=True)


class HelmetStatus(Base):
    """Database model for recording general helmet status metrics (legacy/extended status)."""

    __tablename__ = "helmet_status"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    helmet_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    motorcycle_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    no_helmet_in_roi: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=True)

