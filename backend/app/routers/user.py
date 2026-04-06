from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from structlog import get_logger

from app.configuration import Configuration
from app.core import security
from app.core.auth import get_any_user
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
    PasswordResetConfirm,
    PasswordResetRequest,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.services.auth import AuthService

logger = get_logger(__name__)

router = APIRouter(tags=["user"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    user: UserCreate,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    """
    Register a new user.
    """
    try:
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
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.exception("Database error occurred in register")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurreds",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in register")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        ) from e


@router.post("/login", status_code=status.HTTP_200_OK)
def login(
    response: Response,
    data: UserLogin,
    db: Annotated[Session, Depends(get_db)],
):
    """Login user and return access token."""
    try:
        auth_service = AuthService(db)
        user_obj = auth_service.authenticate_user(data.email, data.password)
        access_token = security.create_access_token(
            data={"sub": user_obj.id, "role": user_obj.role}
        )
        refresh_token = security.create_refresh_token(
            data={"sub": user_obj.id, "role": user_obj.role}
        )

        set_refresh_token_cookie(response, refresh_token)
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.exception("Database error occurred in login")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in login")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        ) from e


@router.post("/password-request", status_code=status.HTTP_200_OK)
def password_request(
    payload: PasswordResetRequest, db: Annotated[Session, Depends(get_db)]
):
    """Request a password reset: generates a token and (optionally) sends email."""
    try:
        auth_service = AuthService(db)
        auth_service.req_password_reset(payload.email)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.exception("Database error occurred in password-request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in password-request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to request password reset",
        ) from e
    else:
        return {"message": "If the email exists, a reset link has been sent."}


@router.post("/password-reset", status_code=status.HTTP_200_OK)
def password_reset(
    payload: PasswordResetConfirm, db: Annotated[Session, Depends(get_db)]
):
    """Confirm password reset using token."""
    try:
        auth_service = AuthService(db)
        auth_service.reset_password(payload.token, payload.new_password)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.exception("Database error occurred in password-reset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in password-reset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password",
        ) from e
    else:
        return {"message": "Password has been reset successfully."}


@router.post(path="/refresh_token", status_code=status.HTTP_200_OK)
def refresh_access_token(
    response: Response,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
):
    """Refresh access token using refresh token."""
    refresh_token = request.cookies.get(
        config.refresh_token_cookie.value
    ) or request.cookies.get(config.refresh_token_cookie.key)
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )
    try:
        _payload = verify_refresh_token(refresh_token)
        user_id = decode_token(refresh_token)
        role = db.query(User.role).filter(User.id == user_id).scalar()

        if not role:
            raise HTTPException(  # noqa: TRY301
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
    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to refresh token",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh token: {e}",
        ) from e


config = Configuration.get_config()


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(response: Response):
    """Logout user by clearing refresh token cookie."""
    try:
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
    except Exception as e:
        logger.exception("Unexpected error in logout")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed",
        ) from e
    else:
        return {"message": "Successfully logged out"}


@router.get("/me", status_code=status.HTTP_200_OK, response_model=UserResponse)
def get_user_me(user: Annotated[User, Depends(get_any_user)]):
    """Get current user information. Returns null for any fields that are null in database."""
    try:
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            created_at=user.created_at,
            last_login=user.last_login,
        )
    except Exception as e:
        logger.exception("Unexpected error in authtest_get")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify token",
        ) from e
