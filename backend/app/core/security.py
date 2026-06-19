from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Response
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt

from app.configuration import Configuration
from app.core.exceptions import TokenDecodeError
from app.schemas.user import UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="user/login")

config = Configuration.get_config()

# Token expiry constants
REFRESH_TOKEN_EXPIRY_DAYS = 30


def set_refresh_token_cookie(response: Response, refresh_token: str) -> None:
    """Set secure refresh token cookie with consistent settings."""
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
    """Verify a plain password against its hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _decode_and_verify_token(
    token: str, token_type: str | None = None
) -> dict[str, Any]:
    """Helper to decode and verify JWT token with consistent error handling.

    Args:
        token: JWT token string
        token_type: Expected token type (e.g., "access", "refresh"). If None, skip check.

    Returns:
        Token payload dictionary

    Raises:
        TokenDecodeError: If token is invalid, expired, or wrong type
    """
    try:
        payload = jwt.decode(
            token, config.key.secret_key, algorithms=[config.key.algorithm]
        )

        if token_type and payload.get("type") != token_type:
            raise TokenDecodeError(f"Wrong token type. Expected {token_type}.")

        return payload
    except ExpiredSignatureError as e:
        raise TokenDecodeError("Token has expired.") from e
    except JWTError as e:
        raise TokenDecodeError(f"Invalid token: {e}") from e
    except TokenDecodeError:
        raise
    except Exception as e:
        raise TokenDecodeError(f"Unexpected error during token decoding: {e}") from e


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create JWT access token.

    Args:
        data: Payload data to encode in token
        expires_delta: Custom expiration time. Uses config default if None.

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=config.key.access_token_minutes
        )

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
    )

    return jwt.encode(to_encode, config.key.secret_key, algorithm=config.key.algorithm)


def decode_token(token: str) -> str:
    """Decode and verify JWT access token.

    Args:
        token: JWT token string

    Returns:
        Subject claim value from token

    Raises:
        TokenDecodeError: If token is invalid or expired
    """
    payload = _decode_and_verify_token(token, token_type="access")
    sub = payload.get("sub")

    if sub is None:
        raise TokenDecodeError("Token payload missing 'sub' claim.")

    return sub


def create_refresh_token(data: dict[str, Any]) -> str:
    """Create JWT refresh token with longer expiry.

    Args:
        data: Payload data to encode in token

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
        }
    )

    return jwt.encode(to_encode, config.key.secret_key, algorithm=config.key.algorithm)


def verify_refresh_token(token: str) -> dict[str, Any]:
    """Decode and verify JWT refresh token.

    Args:
        token: JWT token string

    Returns:
        Token payload dictionary

    Raises:
        TokenDecodeError: If token is invalid, expired, or wrong type
    """
    return _decode_and_verify_token(token, token_type="refresh")
