from typing import Annotated, Optional

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core import security
from app.database.database import get_db
from app.database.user import User
from app.schemas.user import UserRole
from app.services.auth import AuthService


def get_user(
    db: Session = Depends(get_db),
    token: str | None = Depends(security.oauth2_scheme),
) -> Optional[User]:
    """Get the current user from the authentication token."""
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


def require_roles(roles: list[str]):
    """Reusable RBAC dependency for allowed roles."""

    def role_checker(user: User = Depends(get_user)) -> User:
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"You do not have permission to access this resource. Required roles: {roles}",
            )
        return user

    return role_checker


def get_security_user(user: User | None = Depends(get_user)) -> User:
    """Get security user - requires security role."""
    if user is not None and user.role == UserRole.SECURITY:
        return user
    raise HTTPException(status_code=403, detail="Forbidden - security access required")


def get_admin_user(user: User | None = Depends(get_user)) -> User:
    """Get admin user - requires admin role."""
    if user is not None and user.role == UserRole.ADMIN:
        return user
    raise HTTPException(status_code=403, detail="Forbidden - admin access required")


def get_any_user(user: User | None = Depends(get_user)) -> User:
    """Get any authenticated user regardless of role."""
    if user is not None:
        return user
    raise HTTPException(
        status_code=401, detail="Unauthorized access - authentication required"
    )


def get_admin_only():
    """Admin only access"""
    return require_roles([UserRole.ADMIN])


def get_security_or_admin():
    """Security or admin access"""
    return require_roles([UserRole.SECURITY, UserRole.ADMIN])


def get_any_role():
    """Any supported role."""
    return require_roles([UserRole.SECURITY, UserRole.ADMIN])


UserDep = Annotated[User | None, Depends(get_user)]
SecurityUserDep = Annotated[User, Depends(get_security_user)]
AdminUserDep = Annotated[User, Depends(get_admin_user)]
AnyUserDep = Annotated[User, Depends(get_any_user)]
