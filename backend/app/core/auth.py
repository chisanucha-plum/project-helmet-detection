from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core import security
from app.database.database import get_db
from app.database.user import User
from app.services.auth import AuthService


def get_user(
    db: Session = Depends(get_db),
    token: str | None = Depends(security.oauth2_scheme),
) -> User | None:
    """Get the current user from the authentication token.

    Args:
        db: Database session.
        token: OAuth2 token string.

    Returns:
        The User object if token is valid, or None.
    """
    if not token:
        return None
    try:
        if token.startswith("Bearer "):
            token = token.replace("Bearer ", "")
        auth_service = AuthService(db)
        return auth_service.get_current_user(token)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=401, detail="Invalid authentication token"
        ) from error


def get_any_user(user: User | None = Depends(get_user)) -> User:
    """Get any authenticated user regardless of role.

    Args:
        user: The authenticated user instance.

    Returns:
        The authenticated User instance.
    """
    if user is not None:
        return user
    raise HTTPException(
        status_code=401, detail="Unauthorized access - authentication required"
    )
