import functools
import logging
from typing import Any, Callable

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def handle_db_errors(func: Callable) -> Callable:
    """
    Decorator to handle repeated error handling in endpoints.

    Catches HTTPException, SQLAlchemyError, and Exception.
    Logs with function context and returns appropriate error responses.

    Usage:
        @handle_db_errors
        def my_endpoint(...):
            # Just implement the logic
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTPException as-is
            raise
        except SQLAlchemyError as e:
            logger.exception(
                "Database error occurred in %s",
                func.__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred",
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error in %s",
                func.__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            ) from e

    return wrapper
