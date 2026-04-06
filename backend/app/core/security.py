from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Optional

import bcrypt
from fastapi import Response
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt

from app.configuration import Configuration
from app.core.exceptions import TokenDecodeError
from app.schemas.user import UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="user/login")

config = Configuration.get_config()


class RolePermissions:
    """Manages role hierarchy and permissions."""

    ROLE_HIERARCHY: ClassVar[dict[UserRole, int]] = {
        UserRole.ADMIN: 2,
        UserRole.SECURITY: 1,
    }

    @classmethod
    def has_permission(cls, user_role: UserRole, required_role: UserRole) -> bool:
        """Check if a user's role meets the required permission level."""
        user_level = cls.ROLE_HIERARCHY.get(user_role, 0)
        required_level = cls.ROLE_HIERARCHY.get(required_role, 0)
        return user_level >= required_level


def set_refresh_token_cookie(response: Response, refresh_token: str) -> None:
    """Helper function to set secure refresh token cookie with consistent settings."""
    response.set_cookie(
        key=config.refresh_token_cookie.value,
        value=refresh_token,
        max_age=config.refresh_token_cookie.max_age,
        httponly=config.refresh_token_cookie.httponly,
        secure=config.refresh_token_cookie.secure,
        samesite=config.refresh_token_cookie.samesite,
        path=config.refresh_token_cookie.path,
        domain=config.refresh_token_cookie.domain,
    )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=config.key.access_token_minutes
        )

    to_encode.update({"exp": expire})
    to_encode.update({"iat": datetime.now(timezone.utc)})  # issued at
    to_encode.update({"type": "access"})

    return jwt.encode(to_encode, config.key.secret_key, algorithm=config.key.algorithm)


def decode_token(token: str) -> Any:
    try:
        payload = jwt.decode(
            token, config.key.secret_key, algorithms=[config.key.algorithm]
        )
        sub = payload.get("sub")
        if sub is None:
            raise TokenDecodeError("Token payload missing 'sub' claim.")  # noqa: TRY301
    except ExpiredSignatureError:
        raise TokenDecodeError("Token has expired.") from None
    except JWTError as e:
        raise TokenDecodeError(f"Invalid token: {e}") from e
    except Exception as e:
        raise TokenDecodeError(f"Unexpected error during token decoding: {e}") from e
    else:
        return sub


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token (for future use)"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        days=30
    )  # Longer expiry for refresh tokens

    to_encode.update({"exp": expire})
    to_encode.update({"iat": datetime.now(timezone.utc)})
    to_encode.update({"type": "refresh"})

    return jwt.encode(to_encode, config.key.secret_key, algorithm=config.key.algorithm)


def verify_refresh_token(token: str) -> dict:
    """
    Decode and verify JWT refresh token, return payload dict if valid.
    """
    try:
        payload = jwt.decode(
            token, config.key.secret_key, algorithms=[config.key.algorithm]
        )
        if payload.get("type") != "refresh":
            raise TokenDecodeError("Wrong token type.")  # noqa: TRY301
    except ExpiredSignatureError:
        raise TokenDecodeError("Token has expired.") from None
    except JWTError as e:
        raise TokenDecodeError(f"Invalid token: {e}") from e
    except Exception as e:
        raise TokenDecodeError(f"Unexpected error during token decoding: {e}") from e
    else:
        return payload
