"""
Custom exceptions for the application.
"""


class ServiceError(Exception):
    """Base exception for service layer errors"""


class TokenDecodeError(Exception):
    """Custom exception for token decoding failures."""
