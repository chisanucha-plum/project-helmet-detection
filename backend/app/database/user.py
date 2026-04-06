from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base

if TYPE_CHECKING:
    pass


# user_roles = Table(
#     "user_roles",
#     Base.metadata,
#     Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
#     Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
# )

# role_permissions = Table(
#     "role_permissions",
#     Base.metadata,
#     Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
#     Column("permission_id", Integer, ForeignKey("permissions.id"), primary_key=True),
# )


class User(Base):
    __tablename__ = "users"
    __allow_unmapped__ = True
    __table_args__ = {"extend_existing": True}

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    username: Mapped[str | None] = mapped_column(
        String, unique=True, index=True, nullable=True
    )
    email: Mapped[str | None] = mapped_column(
        String, unique=True, index=True, nullable=True
    )
    full_name: Mapped[str | None] = mapped_column(
        String, unique=True, index=True, nullable=True
    )
    hashed_password: Mapped[str | None] = mapped_column(
        String, unique=True, index=True, nullable=True
    )
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # role: Mapped[list["Role"]] = relationship(
    #     "Role", secondary=user_roles, back_populates="users"
    # )

    role: Mapped[str] = mapped_column(String, default="security")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reset_token: Mapped[str | None] = mapped_column(String, nullable=True)
    reset_token_expiry: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# class Role(Base):
#     __tablename__ = "roles"
#     __allow_unmapped__ = True
#     __table_args__ = {"extend_existing": True}

#     id: Mapped[str] = mapped_column(
#         String, primary_key=True, default=lambda: str(uuid4())
#     )
#     name: Mapped[str] = mapped_column(String, unique=True, index=True)
#     permissions: Mapped[list["Permission"]] = relationship(
#         "Permission", secondary=role_permissions, back_populates="roles"
#     )
#     users: Mapped[list[User]] = relationship("User", back_populates="roles")
