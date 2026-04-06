import logging
from contextlib import asynccontextmanager

from app.database.database import init_database
from app.routers.router import get_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_database()
    except SQLAlchemyError as e:
        logger.warning(f"Database init skipped: {e}")
    yield


app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(get_router())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
    )
