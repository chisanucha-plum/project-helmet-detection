"""
Custom exceptions for the application.
"""


class ServiceError(Exception):
    """Base exception for service layer errors"""


class ValidationError(ServiceError):
    """Raised when input validation fails"""


class ExternalServiceError(ServiceError):
    """Raised when external service (like Dify) fails"""


class NetworkError(ServiceError):
    """Raised when network operations fail"""


class TokenDecodeError(Exception):
    """Custom exception for token decoding failures."""
