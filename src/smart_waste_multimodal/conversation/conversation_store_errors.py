"""Exceptions raised by ConversationStore implementations."""


class ConversationStoreError(Exception):
    """Base exception for conversation store operations."""


class ConversationNotFoundError(ConversationStoreError):
    """Raised when a conversation does not exist."""


class ConversationOwnershipError(ConversationStoreError):
    """Raised when an actor attempts to access another actor's conversation."""


class ConversationClosedError(ConversationStoreError):
    """Raised when an operation requires an active conversation."""


class DuplicateConversationTurnError(ConversationStoreError):
    """Raised when a turn ID already exists in a conversation."""


class ConversationTurnMismatchError(ConversationStoreError):
    """Raised when a turn belongs to a different conversation."""


class ConversationTurnOrderError(ConversationStoreError):
    """Raised when a turn timestamp violates chronological ordering."""
