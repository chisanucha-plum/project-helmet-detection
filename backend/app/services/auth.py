import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.core import security
from app.core.exceptions import ServiceError
from app.database.user import User
from app.schemas.user import UserCreate, UserRole

logger = logging.getLogger(__name__)


class AuthService:
    """Service for user authentication and authorization operations."""

    def __init__(self, db: Session) -> None:
        """Initialize auth service with database session."""
        self.db = db

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username from database."""
        return self.db.query(User).filter(User.username == username).first()

    def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID from database."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email from database."""
        return self.db.query(User).filter(User.email == email).first()

    def generate_username_from_email(self, email: str) -> str:
        """Generate unique username from email address.

        Extracts base username from email and appends counter if needed to ensure uniqueness.

        Args:
            email: Email address to derive username from

        Returns:
            Unique username string
        """
        base_username = re.sub(r"\W", "", email.split("@")[0]) or "user"
        username = base_username
        counter = 1
        while self.get_user_by_username(username):
            username = f"{base_username}{counter}"
            counter += 1

        return username


    def create_user(self, user: UserCreate) -> User:
        """Create new user account.

        Args:
            user: User creation schema with email and password

        Returns:
            Created user object

        Raises:
            ServiceError: If email already registered
        """
        existing_user = self.get_user_by_email(user.email)
        if existing_user:
            logger.warning(f"Registration attempted with existing email: {user.email}")
            raise ServiceError(f"Email already registered: {user.email}")

        try:
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
            logger.info(f"New user created: {user.email}")
            return db_user
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create user {user.email}: {e}")
            raise ServiceError(f"Failed to create user: {e}") from e

    def authenticate_user(self, email_or_username: str, password: str) -> User:
        """Authenticate user with email/username and password.

        Args:
            email_or_username: User email or username
            password: Plain text password

        Returns:
            Authenticated user object

        Raises:
            ServiceError: If credentials invalid or user disabled
        """
        user_obj = self.get_user_by_email(email_or_username) or self.get_user_by_username(
            email_or_username
        )
        if (
            not user_obj
            or not user_obj.hashed_password
            or not security.verify_password(password, user_obj.hashed_password)
        ):
            logger.warning(
                f"Login attempt with invalid credentials: {email_or_username}"
            )
            raise ServiceError("Invalid credentials")

        if user_obj.disabled:
            logger.warning(f"Login attempt by disabled user: {email_or_username}")
            raise ServiceError("User account is disabled")


        try:
            user_obj.last_login = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(user_obj)
            logger.info(f"User authenticated: {email_or_username}")
        except Exception as e:
            logger.error(f"Failed to update last_login for {email_or_username}: {e}")

        return user_obj

    def get_current_user(self, token: str) -> User:
        """Get current user from JWT token.

        Args:
            token: JWT access token

        Returns:
            User object associated with token

        Raises:
            ServiceError: If token invalid or user not found/disabled
        """
        try:
            user_id = security.decode_token(token)
        except Exception as e:
            logger.warning(f"Invalid token decode attempt: {e}")
            raise ServiceError(f"Invalid token: {e}") from e

        user_obj = self.get_user_by_id(user_id)
        if user_obj is None:
            logger.warning(f"Token references non-existent user: {user_id}")
            raise ServiceError("User not found")

        if user_obj.disabled:
            logger.warning(f"Access attempt by disabled user: {user_id}")
            raise ServiceError("User account is disabled")

        return user_obj

    def req_password_reset(self, email: str) -> None:
        """Request password reset by creating reset token for user.

        Args:
            email: Email address of user requesting reset
        """
        user_obj = self.get_user_by_email(email)
        if not user_obj:
            logger.info(f"Password reset requested for non-existent email: {email}")
            return

        try:
            token = secrets.token_urlsafe(32)
            expiry = datetime.now(timezone.utc) + timedelta(hours=1)
            user_obj.reset_token = token
            user_obj.reset_token_expiry = expiry
            self.db.commit()
            logger.info(f"Password reset token generated for: {email}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to generate reset token for {email}: {e}")

    def reset_password(self, token: str, new_password: str) -> User:
        """Reset user password with reset token.

        Args:
            token: Password reset token
            new_password: New password to set

        Returns:
            Updated user object

        Raises:
            ServiceError: If token invalid, expired, or reset fails
        """
        user_obj = self.db.query(User).filter(User.reset_token == token).first()
        if not user_obj:
            logger.warning("Password reset attempted with invalid token")
            raise ServiceError("Invalid or expired reset token")

        expiry = user_obj.reset_token_expiry
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        if not expiry or expiry < datetime.now(timezone.utc):
            logger.warning(
                f"Password reset attempted with expired token for: {user_obj.email}"
            )
            raise ServiceError("Invalid or expired reset token")

        try:
            user_obj.hashed_password = security.get_password_hash(new_password)
            user_obj.reset_token = None
            user_obj.reset_token_expiry = None
            self.db.commit()
            logger.info(f"Password reset successful for: {user_obj.email}")
            return user_obj
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to reset password: {e}")
            raise ServiceError("Failed to reset password") from e
