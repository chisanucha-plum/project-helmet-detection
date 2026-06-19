import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from app.core.exceptions import ServiceError
from app.database.database import init_database
from app.routers.router import get_router

logger = logging.getLogger(__name__)

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://172.16.1.222:3000",
]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown events:
    - Startup: Initialize database tables
    - Shutdown: (None currently)

    Args:
        _: FastAPI application instance (not used)

    Yields:
        None
    """
    try:
        logger.info("Initializing database")
        init_database()
        logger.info("Database initialized successfully")
    except SQLAlchemyError as e:
        logger.warning(f"Database initialization skipped: {e}")

    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    """Map ServiceError to appropriate HTTP status codes."""
    msg = str(exc)
    if "Invalid credentials" in msg:
        code = status.HTTP_401_UNAUTHORIZED
    elif "User account is disabled" in msg:
        code = status.HTTP_403_FORBIDDEN
    else:
        code = status.HTTP_400_BAD_REQUEST
    return JSONResponse(status_code=code, content={"detail": msg})


# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(get_router())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
