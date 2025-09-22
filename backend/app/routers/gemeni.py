import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.schemas.gemeni import (
    AnalysisResult,
    GeminiServiceInfo,
    HelmetComplianceResponse,
    ImageInfo,
    LatestSnapshotResponse,
    SnapshotDirectoryInfo,
)
from app.services.gemini import gemini_service
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

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
    "/helmet-compliance",
    status_code=status.HTTP_200_OK,
    response_model=HelmetComplianceResponse,
)
async def analyze_helmet_compliance_endpoint():
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

    # Check if Gemini service is available
    if gemini_service.is_service_available():
        analysis_result = gemini_service.analyze_helmet_compliance(latest_image)
        if analysis_result is None:
            # Set default values if analysis failed
            analysis_result = AnalysisResult(
                helmet=None, total_person=None, violations="Analysis failed"
            )
    else:
        analysis_result = AnalysisResult(
            helmet=None, total_person=None, violations="Gemini service unavailable"
        )

    # Get file info
    file_stats = os.stat(latest_image)

    # Create image info
    image_info = ImageInfo(
        filename=os.path.basename(latest_image),
        full_path=latest_image,
        timestamp=datetime.fromtimestamp(
            file_stats.st_mtime, tz=THAILAND_TZ
        ).isoformat(),
        file_size=file_stats.st_size,
        exists=os.path.exists(latest_image),
    )

    response_data = HelmetComplianceResponse(
        success=True,
        analysis=analysis_result,
        image_info=image_info,
        analysis_timestamp=get_thailand_datetime().isoformat(),
    )

    return response_data


@router.get(
    "/snapshots/latest",
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
            filename=os.path.basename(latest_image),
            full_path=latest_image,
            timestamp=datetime.fromtimestamp(
                file_stats.st_mtime, tz=THAILAND_TZ
            ).isoformat(),
            file_size=file_stats.st_size,
            exists=os.path.exists(latest_image),
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
