from app.mock.datamock import router as mock_router
from app.routers.gemeni import router as analysis_router
from app.routers.helmet import router as helmet_router
from app.routers.snapshots import router as snapshots_router
from fastapi import APIRouter


def get_router():
    router = APIRouter()

    router.include_router(helmet_router, prefix="/helmet")
    router.include_router(analysis_router, prefix="/analysis")
    router.include_router(snapshots_router, prefix="")
    router.include_router(mock_router, prefix="/mock")
    return router
