import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.database.analysis_job import AnalysisJob
from app.database.database import get_db
from app.database.history_status import HistoryStatus
from app.schemas.gemeni import (
    AnalysisResult,
    GeminiServiceInfo,
    HelmetComplianceResponse,
    ImageInfo,
    LatestSnapshotResponse,
    SnapshotDirectoryInfo,
)
from app.services.gemini import gemini_service
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Analysis"])

# Constants
SNAPSHOT_DIR = "snapshots"
# Thailand timezone (UTC+7)
THAILAND_TZ = timezone(timedelta(hours=7))


def get_thailand_datetime():
    """Get current datetime in Thailand timezone."""
    return datetime.now(THAILAND_TZ)


def get_latest_snapshot() -> Optional[str]:
    """Get the latest snapshot image file from snapshots directory."""
    if not os.path.exists(SNAPSHOT_DIR):
        return None

    snapshot_files = [
        f
        for f in os.listdir(SNAPSHOT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not snapshot_files:
        return None

    # Sort by modification time (latest first)
    snapshot_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x)), reverse=True
    )

    return os.path.join(SNAPSHOT_DIR, snapshot_files[0])


@router.get(
    "/helmet",
    status_code=status.HTTP_200_OK,
    response_model=HelmetComplianceResponse,
)
async def analyze_helmet_compliance_endpoint(db: Session = Depends(get_db)):
    """Analyze helmet compliance in the latest snapshot image."""
    latest_image = get_latest_snapshot()
    if not latest_image:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "No snapshot images found",
                "message": "ไม่พบไฟล์ภาพ snapshot",
            },
        )

    # ดึง ID จากชื่อไฟล์ snapshot (ชื่อไฟล์เต็มไม่รวม extension)
    def extract_file_id(filename: str) -> str:
        """Extract file ID from snapshot filename (full filename without extension)"""
        try:
            # ตัวอย่าง: capture_20250925_004942_812_no_helmet_mc_mc_4_6.jpg
            # เอาเฉพาะชื่อไฟล์ (ไม่รวม path และ extension)
            basename = os.path.basename(filename)
            # ตัด .jpg, .jpeg, .png ออก
            file_id = os.path.splitext(basename)[0]
            # Normalize: if filename is like _20251001_101711_166_ strip surrounding underscores
            file_id = file_id.strip("_")
            return file_id
        except Exception:
            pass
        # ถ้าดึงไม่ได้ให้ใช้ timestamp ปัจจุบันแทน
        return f"unknown_{int(datetime.now(THAILAND_TZ).timestamp() * 1000)}"

    file_id = extract_file_id(latest_image)

    # Enqueue analysis job (worker will call Gemini/service in background)
    try:
        job = AnalysisJob(
            image_path=latest_image,
            status="queued",
            created_at=get_thailand_datetime().strftime("%Y-%m-%d %H:%M:%S"),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
    except Exception as e:
        logger.warning(f"Failed to enqueue analysis job: {e}")
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to enqueue analysis job"},
        )

    # Return queued response with placeholder analysis result (worker will update DB)
    analysis_result = AnalysisResult(
        id=file_id,
        helmet_status=None,
        passenger_count=None,
        violations="queued",
    )

    # Get file info
    file_stats = os.stat(latest_image)

    # Create image info
    image_info = ImageInfo(
        filename=latest_image,
        timestamp=datetime.fromtimestamp(file_stats.st_mtime, tz=THAILAND_TZ).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        file_size=file_stats.st_size,
    )

    response_data = HelmetComplianceResponse(
        success=True,
        analysis=analysis_result,
        image_info=image_info,
        analysis_timestamp=get_thailand_datetime().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # บันทึกข้อมูลลงฐานข้อมูล
    try:
        # Check if record already exists
        existing_record = (
            db.query(HistoryStatus)
            .filter(HistoryStatus.id == analysis_result.id)
            .first()
        )

        if existing_record:
            # Update existing record
            existing_record.helmet_status = analysis_result.helmet_status
            existing_record.passenger_count = analysis_result.passenger_count
            existing_record.violations = analysis_result.violations
            existing_record.timestamp = response_data.analysis_timestamp
            logger.info(
                f"Updated existing history record with ID: {analysis_result.id}"
            )
        else:
            # Create new record
            history_record = HistoryStatus(
                id=analysis_result.id,
                helmet_status=analysis_result.helmet_status,
                passenger_count=analysis_result.passenger_count,
                violations=analysis_result.violations,
                timestamp=response_data.analysis_timestamp,
            )
            db.add(history_record)
            logger.info(f"Created new history record with ID: {analysis_result.id}")

        db.commit()

    except Exception as e:
        logger.warning(f"Failed to save history (continuing without DB): {e}")
        db.rollback()

    return response_data


@router.get(
    "/snapshots",
    status_code=status.HTTP_200_OK,
    response_model=LatestSnapshotResponse,
)
async def get_latest_snapshot_info():
    """Get information about the latest snapshot without AI analysis."""
    latest_image = get_latest_snapshot()
    if not latest_image:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "No snapshot images found",
                "message": "ไม่พบไฟล์ภาพ snapshot",
            },
        )

    file_stats = os.stat(latest_image)

    # Count all snapshots in directory
    snapshot_count = 0
    if os.path.exists(SNAPSHOT_DIR):
        snapshot_count = len(
            [
                f
                for f in os.listdir(SNAPSHOT_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    response_data = LatestSnapshotResponse(
        success=True,
        image_info=ImageInfo(
            filename=latest_image,
            timestamp=datetime.fromtimestamp(
                file_stats.st_mtime, tz=THAILAND_TZ
            ).strftime("%Y-%m-%d %H:%M:%S"),
            file_size=file_stats.st_size,
        ),
        snapshots_directory=SnapshotDirectoryInfo(
            path=SNAPSHOT_DIR,
            total_files=snapshot_count,
            exists=os.path.exists(SNAPSHOT_DIR),
        ),
        gemini_service=GeminiServiceInfo(
            available=gemini_service.is_service_available(),
            status="ready" if gemini_service.is_service_available() else "unavailable",
        ),
        message=f"ข้อมูล snapshot ล่าสุด (มีทั้งหมด {snapshot_count} ไฟล์)",
    )

    return response_data
