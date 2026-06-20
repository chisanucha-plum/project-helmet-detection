import asyncio
import logging
from typing import Any

from app.database.database import get_db
from app.database.history_status import HistoryStatus
from app.schemas.helmet import HistoryStatusResponse
from app.services.camera_hub import camera_hub
from app.services.frame_storage import frame_storage
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Helmet Detection"])

# Query parameter limits for history endpoint
DEFAULT_HISTORY_LIMIT = 50
MAX_HISTORY_LIMIT = 500

# MJPEG streaming boundary
MJPEG_BOUNDARY = b"--frame"


@router.get("/stream", status_code=status.HTTP_200_OK)
async def helmet_video_stream() -> StreamingResponse:
    """Return MJPEG video stream with YOLO detection annotations."""
    logger.info("Video stream client connected")
    q = camera_hub.subscribe_frames()

    async def generate() -> Any:
        try:
            while True:
                frame_bytes: bytes = await q.get()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
        except asyncio.CancelledError:
            logger.info("Video stream client disconnected")
        finally:
            camera_hub.unsubscribe_frames(q)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/events", status_code=status.HTTP_200_OK)
async def helmet_detection_events() -> StreamingResponse:
    """Return Server-Sent Events stream for detection records.

    Emits detection JSON payload whenever motorcycle crosses detection line.

    Example payload:
        {"motorcycle_track_id": 1, "helmet_status": true, "violation": false, ...}
    """
    logger.info("Detection events stream client connected")
    q = camera_hub.subscribe_detections()

    async def generate() -> Any:
        try:
            while True:
                payload: str = await q.get()
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            logger.info("Detection events stream client disconnected")
        finally:
            camera_hub.unsubscribe_detections(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history", response_model=list[HistoryStatusResponse])
async def get_history_records(
    limit: int = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT),
    db: Session = Depends(get_db),
) -> list[HistoryStatusResponse]:
    """Get recent detection history records from database.

    Args:
        limit: Number of records to return (1-500, default 50)
        db: Database session

    Returns:
        List of detection history records sorted by timestamp descending
    """
    logger.info(f"Fetching {limit} detection history records")
    return (
        db.query(HistoryStatus)
        .order_by(desc(HistoryStatus.timestamp))
        .limit(limit)
        .all()
    )


@router.get("/frame/{date}/{filename}")
async def get_frame(date: str, filename: str) -> FileResponse:
    """Get saved frame image by date and filename.
    
    Args:
        date: Date folder (YYYY-MM-DD)
        filename: Filename in format track_{id}_{status}_{timestamp}.jpg
        
    Returns:
        JPEG image file
    """
    filepath = frame_storage.base_dir / date / filename
    
    if not filepath.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Frame not found")
    
    logger.debug(f"Serving frame: {filepath}")
    return FileResponse(filepath, media_type="image/jpeg")
