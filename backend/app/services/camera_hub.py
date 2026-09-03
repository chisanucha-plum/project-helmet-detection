import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from app.configuration import ApplicationSettingsConfig, Configuration
from app.database.database import SessionLocal
from app.database.history_status import HistoryStatus
from app.models.detection import DetectionRecord
from app.services.detection import DetectionService
from app.services.frame_storage import frame_storage

logger = logging.getLogger(__name__)

# Seconds to wait before retrying a dropped live stream (RTSP/webcam)
RECONNECT_DELAY_SEC = 2.0
# Frame queues are latest-wins (drop stale JPEGs); detection queues must not lose records
FRAME_QUEUE_SIZE = 2
DETECTION_QUEUE_SIZE = 100
# Seconds between polls while waiting for the grabber thread's next frame
IDLE_POLL_SEC = 0.02


def _is_stream_url(video_path: str) -> bool:
    """Return True if the video source is a network stream rather than a file."""
    return (
        str(video_path)
        .lower()
        .startswith(("rtsp://", "rtsps://", "http://", "https://"))
    )


def _describe_source(app_settings: ApplicationSettingsConfig) -> str:
    """Return a human-readable name of the configured video source for logs."""
    if app_settings.use_webcam:
        return f"webcam id={app_settings.webcam_id}"
    return app_settings.video_path


class CameraHub:
    """
    Runs capture + YOLO in a background thread (not asyncio) for minimal latency.

    - Opens the camera when the first subscriber connects.
    - Closes the camera when all subscribers disconnect.
    - Broadcasts JPEG frames to MJPEG subscribers.
    - Broadcasts detection payloads to SSE subscribers and persists them to DB.

    Live sources (RTSP/webcam) use a dedicated grabber thread that keeps only
    the newest frame, so slow inference never lets a backlog build up and
    stream latency stays bounded at roughly one inference step.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_subs: list[asyncio.Queue] = []
        self._detect_subs: list[asyncio.Queue] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._service: DetectionService | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Latest-frame slot shared between grabber thread and processing loop
        self._latest_frame: np.ndarray | None = None
        self._latest_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-db")


    def subscribe_frames(self) -> asyncio.Queue:
        """Return a queue that receives JPEG bytes for every captured frame."""
        q: asyncio.Queue = asyncio.Queue(maxsize=FRAME_QUEUE_SIZE)
        with self._lock:
            self._frame_subs.append(q)
            self._ensure_running()
        return q

    def unsubscribe_frames(self, q: asyncio.Queue) -> None:
        """Remove a frame subscriber and stop the camera if no subscribers remain."""
        with self._lock:
            if q in self._frame_subs:
                self._frame_subs.remove(q)
            self._stop_if_idle()

    def subscribe_detections(self) -> asyncio.Queue:
        """Return a queue that receives JSON-encoded detection payloads."""
        q: asyncio.Queue = asyncio.Queue(maxsize=DETECTION_QUEUE_SIZE)
        with self._lock:
            self._detect_subs.append(q)
            self._ensure_running()
        return q

    def unsubscribe_detections(self, q: asyncio.Queue) -> None:
        """Remove a detection subscriber and stop the camera if no subscribers remain."""
        with self._lock:
            if q in self._detect_subs:
                self._detect_subs.remove(q)
            self._stop_if_idle()

    def _ensure_running(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._loop = asyncio.get_event_loop()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("Camera capture thread started")

    def _stop_if_idle(self) -> None:
        if not self._frame_subs and not self._detect_subs:
            self._stop_event.set()
            logger.info("Stopping camera capture (no subscribers)")

    def _push_frame(self, data: bytes) -> None:
        """Push JPEG frame to all MJPEG subscribers.

        Drops oldest frames if queue is full to prevent blocking and keep stream live.

        Args:
            data: JPEG frame bytes to broadcast
        """
        if self._loop is None:
            return

        def _put(q: asyncio.Queue[bytes], d: bytes) -> None:
            """Add frame to queue, dropping old frames if necessary to prevent blocking."""
            while q.full():
                try:
                    q.get_nowait()
                except Exception:
                    break
            try:
                q.put_nowait(d)
            except Exception as e:
                logger.debug(f"Failed to push frame to queue: {e}")

        with self._lock:
            subs = list(self._frame_subs)
        for q in subs:
            self._loop.call_soon_threadsafe(_put, q, data)

    def _push_detections(
        self, records: list[DetectionRecord], frame: np.ndarray | None = None
    ) -> None:
        """Push detection records to all subscribers and persist to database.

        Args:
            records: List of detection records to broadcast and save
            frame: OpenCV frame (optional) to save as snapshot
        """
        if self._loop is None:
            return

        self._save_to_db(records, frame)
        payload = json.dumps([r.to_dict() for r in records], ensure_ascii=False)

        def _put(q: asyncio.Queue[str], p: str) -> None:
            """Add detection payload to queue."""
            try:
                q.put_nowait(p)
            except Exception as e:
                logger.debug(f"Failed to push detection to queue: {e}")

        with self._lock:
            subs = list(self._detect_subs)
        for q in subs:
            self._loop.call_soon_threadsafe(_put, q, payload)

    # ── persistence ──────────────────────────────────────────────────────────

    def _save_to_db(
        self, records: list[DetectionRecord], frame: np.ndarray | None = None
    ) -> None:
        """Persist detection records to history_status table with frame.

        Args:
            records: List of detection records to save
            frame: OpenCV frame (optional) for saving snapshot
        """
        try:
            jpeg_bytes = None
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    jpeg_bytes = buf.tobytes()

            with SessionLocal() as db:
                for r in records:
                    record_id = (
                        f"trk_{r.motorcycle_track_id}_"
                        f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    )

                    # Save frame snapshot if provided (before storing in DB)
                    frame_path = None
                    if frame is not None:
                        frame_path = frame_storage.save_frame(
                            frame,
                            r.motorcycle_track_id,
                            r.violation,
                            jpeg_bytes=jpeg_bytes,
                        )
                        # Update record with frame_path
                        r.frame_path = frame_path

                    db.add(
                        HistoryStatus(
                            id=record_id,
                            track_id=r.motorcycle_track_id,
                            helmet_status=r.helmet_status,
                            passenger_count=r.passenger_count,
                            over_capacity=r.over_capacity,
                            violation=r.violation,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            frame_path=frame_path,
                        )
                    )
                db.commit()
                logger.info(
                    "Saved detections to database",
                    extra={"record_count": len(records)},
                )
        except Exception as e:
            logger.error(
                "Failed to save detections to database",
                extra={"error": str(e)},
            )


    # ── worker thread ────────────────────────────────────────────────────────

    def _pop_latest_frame(self) -> np.ndarray | None:
        """Return the newest grabbed frame (if any) and clear the slot."""
        with self._latest_lock:
            frame, self._latest_frame = self._latest_frame, None
        return frame

    def _grab_loop(self, config: Configuration, service: DetectionService) -> None:
        """Continuously grab frames from a live source into the latest-frame slot.

        Only the newest frame is kept, so the processing loop always works on
        fresh input no matter how long inference takes. Reconnects when the
        source drops instead of seeking (live streams cannot seek).

        Args:
            config: Application configuration
            service: Detection service (reset when the stream reconnects)
        """
        cap = self._open_capture(config)
        if not cap.isOpened():
            logger.error(
                "Failed to open live video source: "
                f"{_describe_source(config.application_settings)}"
            )
            return

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame grab failed on live source, reconnecting...")
                    cap.release()
                    time.sleep(RECONNECT_DELAY_SEC)
                    cap = self._open_capture(config)
                    if cap.isOpened():
                        service.reset_tracks()
                        logger.info("Reconnected to live source")
                    continue

                with self._latest_lock:
                    self._latest_frame = frame
        finally:
            cap.release()
            logger.info("Grabber thread stopped")

    def _get_service(self) -> DetectionService:
        if self._service is None:
            config = Configuration.get_config()
            self._service = DetectionService(
                bike_model=Path(config.models.bike_model),
                helmet_model=Path(config.models.helmet_model),
                config=config.detection,
            )
        return self._service

    def _open_capture(self, config: Configuration) -> cv2.VideoCapture:
        """Open video capture from configured source (webcam, RTSP stream, or file).

        Args:
            config: Application configuration

        Returns:
            cv2.VideoCapture instance

        Note:
            Uses DirectShow on Windows to avoid MSMF grab errors
        """
        app = config.application_settings
        if app.use_webcam:
            # DirectShow avoids MSMF grab errors on Windows
            cap = cv2.VideoCapture(app.webcam_id, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        if _is_stream_url(app.video_path):
            # Never pass stream URLs through Path() — on Windows it rewrites
            # "/" to "\", corrupting credentials and host in the URL
            cap = cv2.VideoCapture(app.video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        return cv2.VideoCapture(str(Path(app.video_path)))

    def _process_frame(
        self, frame: np.ndarray, service: DetectionService, jpeg_quality: int
    ) -> None:
        """Run detection on one frame and broadcast the annotation and any records."""
        annotated, new_records = service.detect_and_track(frame)

        with self._lock:
            has_frame_subs = bool(self._frame_subs)

        if has_frame_subs:
            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok:
                self._push_frame(buf.tobytes())

        if new_records:
            # Offload persistence and SSE push to background worker thread
            self._executor.submit(self._push_detections, new_records, annotated.copy())


    def _run(self) -> None:
        """Background thread loop for continuous capture and detection.

        For live sources (RTSP/webcam) a dedicated grabber thread keeps only the
        newest frame so inference lag never accumulates into stream delay. File
        sources are read sequentially and loop back to the start when finished.
        """
        config = Configuration.get_config()
        service = self._get_service()
        service.reset_tracks()

        app_settings = config.application_settings
        is_live_source = app_settings.use_webcam or _is_stream_url(
            app_settings.video_path
        )

        grab_thread: threading.Thread | None = None
        cap: cv2.VideoCapture | None = None

        if is_live_source:
            grab_thread = threading.Thread(
                target=self._grab_loop, args=(config, service), daemon=True
            )
            grab_thread.start()
            logger.info("Live source grabber thread started")
        else:
            cap = self._open_capture(config)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {app_settings.video_path}")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video properties - FPS: {fps}, Total Frames: {frame_count}")

        jpeg_quality = config.models.jpeg_quality
        logger.info(f"Started capturing video frames: {_describe_source(app_settings)}")

        try:
            processed_count = 0
            while not self._stop_event.is_set():
                if is_live_source:
                    # Always process the freshest grabbed frame; skip stale ones
                    frame = self._pop_latest_frame()
                    if frame is None:
                        if self._stop_event.wait(IDLE_POLL_SEC):
                            break
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(
                            f"Video loop complete (read {processed_count} frames), restarting..."
                        )
                        # Restart video from beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        service.reset_tracks()
                        continue

                processed_count += 1

                try:
                    self._process_frame(frame, service, jpeg_quality)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)
                    continue

        except Exception as e:
            logger.error(f"Unexpected error in video capture loop: {e}", exc_info=True)
        finally:
            if cap is not None:
                cap.release()
            logger.info("Video processing stopped")


# Module-level singleton — one camera per process
camera_hub = CameraHub()
