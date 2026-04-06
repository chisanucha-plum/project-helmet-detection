from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, model_validator


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_passwords_match(self):
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: str
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class UserRole(str, Enum):
    ADMIN = "admin"
    SECURITY = "security"


class RolePermissions:
    ROLE_HIERARCHY = {
        UserRole.ADMIN: 2,
        UserRole.SECURITY: 1,
    }

    @classmethod
    def has_permission(cls, user_role: str, required_role: str) -> bool:
        """Check if user role has permission for required role"""
        user_level = cls.ROLE_HIERARCHY.get(user_role, 0)
        required_level = cls.ROLE_HIERARCHY.get(required_role, 0)
        return user_level >= required_level


class ChatLogCreate(BaseModel):
    message_id: Optional[str]
    user_id: str
    conversation_id: str
    created_at: datetime
    prompt_token: Optional[int] = 0
    completion_token: Optional[int] = 0
    total_token: Optional[int] = 0
    price_usd: Optional[float] = 0.0
    latency: Optional[float] = 0.0


class ChatLogResponse(BaseModel):
    message_id: Optional[str]
    user_id: str
    conversation_id: Optional[str]
    created_at: datetime
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    price_usd: Optional[float]
    latency: Optional[float]


class Total(BaseModel):
    total: int
    items: list[ChatLogResponse]
