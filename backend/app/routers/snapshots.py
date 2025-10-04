import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

# Directory where snapshots are stored (relative to project root)
SNAPSHOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "snapshots")
)


@router.get("/snapshots/{filename}")
def serve_snapshot(filename: str):
    # sanitize filename
    safe = os.path.basename(filename)
    path = os.path.join(SNAPSHOT_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"detail": "Snapshot not found"})
    return FileResponse(path, media_type="image/jpeg")
