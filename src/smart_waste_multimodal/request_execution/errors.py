"""Exceptions raised by RequestExecutionStore implementations."""


class RequestExecutionStoreError(Exception):
    """Base exception for request execution storage operations."""


class RequestExecutionNotFoundError(RequestExecutionStoreError):
    """Raised when a request execution does not exist."""


class RequestExecutionOwnershipError(RequestExecutionStoreError):
    """Raised when an actor attempts to access another actor's execution."""


class IdempotencyConflictError(RequestExecutionStoreError):
    """Raised when a request ID is reused with a different fingerprint."""


class RequestExecutionStateError(RequestExecutionStoreError):
    """Raised when an execution receives an invalid state transition."""


class InvalidRequestExecutionError(RequestExecutionStoreError):
    """Raised when request execution input is invalid."""
