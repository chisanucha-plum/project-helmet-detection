from typing import Annotated, Callable, Optional

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


def require_roles(roles: list[str]) -> Callable[[User], User]:
    """Reusable RBAC dependency for allowed roles.

    Args:
        roles: List of role strings that are authorized.

    Returns:
        A checker dependency function.
    """

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
    """Get security user - requires security role.

    Args:
        user: The authenticated user instance.

    Returns:
        The verified User instance with security role.
    """
    if user is not None and user.role == UserRole.SECURITY:
        return user
    raise HTTPException(status_code=403, detail="Forbidden - security access required")


def get_admin_user(user: User | None = Depends(get_user)) -> User:
    """Get admin user - requires admin role.

    Args:
        user: The authenticated user instance.

    Returns:
        The verified User instance with admin role.
    """
    if user is not None and user.role == UserRole.ADMIN:
        return user
    raise HTTPException(status_code=403, detail="Forbidden - admin access required")


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


def get_admin_only() -> Callable[[User], User]:
    """Admin only access dependency.

    Returns:
        Role checker dependency function.
    """
    return require_roles([UserRole.ADMIN])


def get_security_or_admin() -> Callable[[User], User]:
    """Security or admin access dependency.

    Returns:
        Role checker dependency function.
    """
    return require_roles([UserRole.SECURITY, UserRole.ADMIN])


def get_any_role() -> Callable[[User], User]:
    """Any supported role dependency.

    Returns:
        Role checker dependency function.
    """
    return require_roles([UserRole.SECURITY, UserRole.ADMIN])



UserDep = Annotated[User | None, Depends(get_user)]
SecurityUserDep = Annotated[User, Depends(get_security_user)]
AdminUserDep = Annotated[User, Depends(get_admin_user)]
AnyUserDep = Annotated[User, Depends(get_any_user)]
