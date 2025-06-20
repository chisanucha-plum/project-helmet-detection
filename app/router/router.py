from fastapi import APIRouter

from app.controller.controller import (
    helmet_detection_stream,
)

router = APIRouter()


generate_router = APIRouter(prefix="/detection")
generate_router.get("/helmet")(helmet_detection_stream)
router.include_router(generate_router)