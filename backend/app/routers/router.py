from app.routers.helmet import router as helmet_router
from app.routers.user import router as user_router
from fastapi import APIRouter


def get_router() -> APIRouter:
    """Compose and return the main APIRouter containing helmet and user endpoints."""
    router = APIRouter()
    router.include_router(helmet_router, prefix="/helmet")
    router.include_router(user_router, prefix="/user")
    return router

