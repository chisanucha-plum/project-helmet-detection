from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(tags=["Mock Auth"])


class MockLoginRequest(BaseModel):
    email: str
    password: str


class MockUser(BaseModel):
    id: int
    name: str
    email: str
    role: Literal["admin", "user"]


class MockLoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    user: MockUser


MOCK_USERS = {
    "admin@kmutt.ac.th": {
        "id": 1,
        "name": "System Admin",
        "password": "admin123",
        "role": "admin",
    },
    "user@kmutt.ac.th": {
        "id": 2,
        "name": "Operator User",
        "password": "user123",
        "role": "user",
    },
}


@router.post("/login", response_model=MockLoginResponse)
async def mock_login(payload: MockLoginRequest):
    user = MOCK_USERS.get(payload.email.lower())
    if not user or payload.password != user["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Non-secure token for local mock flow only.
    mock_token = f"mock-token-{user['role']}-{user['id']}"
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()

    return MockLoginResponse(
        access_token=mock_token,
        token_type="bearer",
        expires_at=expires_at,
        user=MockUser(
            id=user["id"],
            name=user["name"],
            email=payload.email.lower(),
            role=user["role"],
        ),
    )
