import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core import security
from app.database.user import User
from app.schemas.user import UserCreate, UserRole


class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_by_username(self, username: str) -> Optional[User]:
        return self.db.query(User).filter(User.username == username).first()

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def generate_username_from_email(self, email: str) -> str:
        """Generate unique username from email"""
        base_username = email.split("@")[0]
        base_username = re.sub(r"\W", "", base_username)

        # Ensure it's not empty
        if not base_username:
            base_username = "user"

        # Check if username exists, if so add number suffix
        username = base_username
        counter = 1
        while self.get_user_by_username(username):
            username = f"{base_username}{counter}"
            counter += 1

        return username

    def promote_user_to_admin(self, user_id: str) -> User:
        """Promote an existing user to admin role."""
        user = (
            self.get_user_by_id(user_id)
            if user_id.isdigit()
            else self.db.query(User).filter(User.id == user_id).first()
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        user.role = UserRole.ADMIN
        self.db.commit()
        self.db.refresh(user)
        return user

    def promote_user_by_email_to_admin(self, email: str) -> User:
        """Promote an existing user to admin role by email."""
        user = self.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        user.role = UserRole.ADMIN
        self.db.commit()
        self.db.refresh(user)
        return user

    def create_user(self, user: UserCreate) -> User:
        existing_user = self.get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        username = self.generate_username_from_email(user.email)
        hashed_password = security.get_password_hash(user.password)
        db_user = User(
            username=username,
            email=user.email,
            full_name=user.email.split("@")[0],
            hashed_password=hashed_password,
            role=UserRole.SECURITY,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def authenticate_user(self, email_or_username: str, password: str) -> User:
        user_obj = self.get_user_by_email(email_or_username)
        if not user_obj:
            user_obj = self.get_user_by_username(email_or_username)

        if not user_obj:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if user_obj.hashed_password is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            )
        if not security.verify_password(password, user_obj.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            )

        if user_obj.disabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )

        user_obj.last_login = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user_obj)

        return user_obj

    def get_current_user(
        self,
        token: str = Depends(security.oauth2_scheme),
    ) -> User:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            user_id = security.decode_token(token)
        except Exception as e:
            raise credentials_exception from e

        user_obj = self.get_user_by_id(user_id)
        if user_obj is None:
            raise credentials_exception

        if user_obj.disabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )

        return user_obj

    def check_role(self, user_obj: User, allowed_roles: list[str]):
        if user_obj.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
            )

    def req_password_reset(self, email: str) -> None:
        """Set reset token and expiry for user, to be sent via email."""
        user_obj = self.get_user_by_email(email)
        if not user_obj:
            return
        token = secrets.token_urlsafe(32)
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        user_obj.reset_token = token
        user_obj.reset_token_expiry = expiry
        self.db.commit()

    def reset_password(self, token: str, new_password: str) -> User:
        """Reset user password using a valid token."""
        user_obj = self.db.query(User).filter(User.reset_token == token).first()
        if not user_obj:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired token",
            )
        expiry = user_obj.reset_token_expiry
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if not expiry or expiry < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired token",
            )
        user_obj.hashed_password = security.get_password_hash(new_password)
        user_obj.reset_token = None
        user_obj.reset_token_expiry = None
        self.db.commit()
        return user_obj
