from app.routers.gemeni import router as analysis_router
from app.routers.helmet import router as helmet_router
from app.routers.snapshots import router as snapshots_router
from app.routers.user import router as user_router
from fastapi import APIRouter


def get_router():
    router = APIRouter()

    router.include_router(helmet_router, prefix="/helmet")
    router.include_router(analysis_router, prefix="/analysis")
    router.include_router(user_router, prefix="/user")
    router.include_router(snapshots_router, prefix="")
    return router
