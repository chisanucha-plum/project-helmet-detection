import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.configuration import Configuration
from app.core import security
from app.core.auth import get_any_user
from app.core.error_handlers import handle_db_errors
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    set_refresh_token_cookie,
    verify_refresh_token,
)
from app.database.database import get_db
from app.database.user import User
from app.schemas.user import (
    AuthUserSnapshot,
    LoginResponse,
    PasswordResetConfirm,
    PasswordResetRequest,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.services.auth import AuthService

logger = logging.getLogger(__name__)

config = Configuration.get_config()

router = APIRouter(tags=["user"])


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=Token)
@handle_db_errors
def register(
    user: UserCreate,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> Token:
    """Register a new user."""
    auth_service = AuthService(db)
    db_user = auth_service.create_user(user)
    access_token = security.create_access_token(
        data={"sub": db_user.id, "role": db_user.role}
    )
    refresh_token = security.create_refresh_token(
        data={"sub": db_user.id, "role": db_user.role}
    )

    set_refresh_token_cookie(response, refresh_token)
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/login", status_code=status.HTTP_200_OK, response_model=LoginResponse)
@handle_db_errors
def login(
    response: Response,
    data: UserLogin,
    db: Annotated[Session, Depends(get_db)],
) -> LoginResponse:
    """Login user and return access token."""
    auth_service = AuthService(db)
    user_obj = auth_service.authenticate_user(data.email, data.password)
    access_token = security.create_access_token(
        data={"sub": user_obj.id, "role": user_obj.role}
    )
    refresh_token = security.create_refresh_token(
        data={"sub": user_obj.id, "role": user_obj.role}
    )

    set_refresh_token_cookie(response, refresh_token)
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=AuthUserSnapshot(
            id=user_obj.id,
            username=user_obj.username,
            email=user_obj.email,
            full_name=user_obj.full_name,
            role=user_obj.role,
        ),
    )


@router.post("/password-request", status_code=status.HTTP_200_OK)
@handle_db_errors
def password_request(
    payload: PasswordResetRequest, db: Annotated[Session, Depends(get_db)]
) -> dict[str, str]:
    """Request a password reset: generates a token and (optionally) sends email."""
    auth_service = AuthService(db)
    auth_service.req_password_reset(payload.email)
    return {"message": "If the email exists, a reset link has been sent."}


@router.post("/password-reset", status_code=status.HTTP_200_OK)
@handle_db_errors
def password_reset(
    payload: PasswordResetConfirm, db: Annotated[Session, Depends(get_db)]
) -> dict[str, str]:
    """Confirm password reset using token."""
    auth_service = AuthService(db)
    auth_service.reset_password(payload.token, payload.new_password)
    return {"message": "Password has been reset successfully."}


@router.post(path="/refresh_token", status_code=status.HTTP_200_OK, response_model=Token)
@handle_db_errors
def refresh_access_token(
    response: Response,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> Token:
    """Refresh access token using refresh token."""
    refresh_token = request.cookies.get(
        config.refresh_token_cookie.value
    ) or request.cookies.get(config.refresh_token_cookie.key)
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    _payload = verify_refresh_token(refresh_token)
    user_id = decode_token(refresh_token)
    role = db.query(User.role).filter(User.id == user_id).scalar()

    if not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    access_token = create_access_token({"sub": user_id, "role": role})
    new_refresh_token = create_refresh_token({"sub": user_id, "role": role})

    set_refresh_token_cookie(response, new_refresh_token)
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


@router.post("/logout", status_code=status.HTTP_200_OK)
@handle_db_errors
def logout(response: Response) -> dict[str, str]:
    """Logout user by clearing refresh token cookie."""
    response.delete_cookie(
        key=config.refresh_token_cookie.value,
        httponly=config.refresh_token_cookie.httponly,
        secure=config.refresh_token_cookie.secure,
        samesite=config.refresh_token_cookie.samesite,
        path=config.refresh_token_cookie.path,
        domain=config.refresh_token_cookie.domain,
    )
    # delete legacy cookie name if it differs
    if config.refresh_token_cookie.key and (
        config.refresh_token_cookie.key != config.refresh_token_cookie.value
    ):
        response.delete_cookie(
            key=config.refresh_token_cookie.key,
            httponly=config.refresh_token_cookie.httponly,
            secure=config.refresh_token_cookie.secure,
            samesite=config.refresh_token_cookie.samesite,
            path=config.refresh_token_cookie.path,
            domain=config.refresh_token_cookie.domain,
        )
    return {"message": "Successfully logged out"}


@router.get("/me", status_code=status.HTTP_200_OK, response_model=UserResponse)
@handle_db_errors
def get_user_me(user: Annotated[User, Depends(get_any_user)]) -> UserResponse:
    """Get current user information. Returns null for any fields that are null in database."""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        created_at=user.created_at,
        last_login=user.last_login,
    )
