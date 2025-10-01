import asyncio
import logging
import os
import time

from app.database.analysis_job import AnalysisJob
from app.database.database import SessionLocal
from app.database.history_status import HistoryStatus
from app.routers.gemeni import get_thailand_datetime
from app.schemas.gemeni import AnalysisResult
from app.services.gemini import gemini_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POLL_INTERVAL = 2.0  # seconds


def process_job(job, db):
    try:
        analysis_result = asyncio.run(
            asyncio.to_thread(gemini_service.analyze_helmet_compliance, job.image_path)
        )
        if analysis_result is None:
            analysis_result = AnalysisResult(
                id=None,
                helmet_status=None,
                passenger_count=None,
                violations="analysis_failed",
            )

        # Ensure we always have a non-null id for history records. Prefer the analysis_result.id
        # but fall back to deriving one from the job.image_path (filename without extension).
        result_id = (
            analysis_result.id
            if getattr(analysis_result, "id", None)
            else os.path.splitext(os.path.basename(job.image_path))[0]
        )
        # update the AnalysisResult id so subsequent code/logs can use it
        analysis_result.id = result_id

        # Upsert into history_status
        existing = db.query(HistoryStatus).filter(HistoryStatus.id == result_id).first()
        timestamp = get_thailand_datetime().strftime("%Y-%m-%d %H:%M:%S")
        if existing:
            existing.helmet_status = analysis_result.helmet_status
            existing.passenger_count = analysis_result.passenger_count
            existing.violations = analysis_result.violations
            existing.timestamp = timestamp
        else:
            rec = HistoryStatus(
                id=result_id,
                helmet_status=analysis_result.helmet_status,
                passenger_count=analysis_result.passenger_count,
                violations=analysis_result.violations,
                timestamp=timestamp,
            )
            db.add(rec)
        # mark job done
        job.status = "done"
        db.add(job)
        db.commit()
    except Exception:
        logger.exception(f"Failed to process job id={job.id}")
        db.rollback()
        job.status = "failed"
        db.add(job)
        db.commit()


if __name__ == "__main__":
    # Simple polling worker loop. Keep minimal logging to reduce noise.
    while True:
        db = SessionLocal()
        try:
            job = (
                db.query(AnalysisJob)
                .filter(AnalysisJob.status == "queued")
                .order_by(AnalysisJob.id.asc())
                .first()
            )
            if job:
                process_job(job, db)
            else:
                time.sleep(POLL_INTERVAL)
        finally:
            db.close()
