from app.routers.helmet import router as helmet_router
from fastapi import APIRouter


def get_router():
    router = APIRouter()

    router.include_router(helmet_router, prefix="/helmet")

    return router