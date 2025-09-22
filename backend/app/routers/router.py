from app.routers.gemeni import router as analysis_router
from app.routers.helmet import router as helmet_router
from fastapi import APIRouter


def get_router():
    router = APIRouter()

    router.include_router(helmet_router, prefix="/helmet")
    router.include_router(analysis_router, prefix="/analysis")
    return router
