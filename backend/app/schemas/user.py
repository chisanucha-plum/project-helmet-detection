from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, model_validator


class PasswordResetRequest(BaseModel):
    """Schema for requesting a password reset link by email."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for confirming a password reset request using a token."""

    token: str
    new_password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_passwords_match(self) -> "PasswordResetConfirm":
        """Verify that the new password and confirmation password match."""
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserCreate(BaseModel):
    """Schema for creating a new user account."""

    email: EmailStr
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_passwords_match(self) -> "UserCreate":
        """Verify that the password and confirmation password match."""
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserLogin(BaseModel):
    """Schema for user login credentials."""

    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for returning JWT access and refresh tokens."""

    access_token: str
    refresh_token: str


class AuthUserSnapshot(BaseModel):
    """Detailed snapshot of authenticated user details returned on login."""

    id: str
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: str


class LoginResponse(Token):
    """Schema representing the successful authentication response containing tokens and user snapshot."""

    user: AuthUserSnapshot


class UserResponse(BaseModel):
    """Schema for user details retrieved from the system."""

    id: str
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: str
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class UserRole(str, Enum):
    """Enumeration of available application security roles."""

    ADMIN = "admin"
    SECURITY = "security"

