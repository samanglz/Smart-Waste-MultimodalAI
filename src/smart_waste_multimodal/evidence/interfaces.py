"""Storage-independent evidence registry interface."""

from typing import Protocol, runtime_checkable

from ..contracts.evidence import Evidence


class EvidenceRegistryError(Exception):
    """Base exception for registry operations."""


class DuplicateEvidenceError(EvidenceRegistryError):
    """Raised when an evidence identifier is already registered."""


class EvidenceRequestMismatchError(EvidenceRegistryError):
    """Raised when evidence belongs to a different request."""


@runtime_checkable
class EvidenceRegistry(Protocol):
    """Request-scoped interface implemented by evidence stores."""

    @property
    def request_id(self) -> str:
        """Request identifier that scopes this registry."""

    def register(self, evidence: Evidence) -> None:
        """Register one immutable evidence snapshot."""

    def get(self, evidence_id: str) -> Evidence | None:
        """Return one evidence item, or None when it does not exist."""

    def get_many(self, evidence_ids: list[str]) -> list[Evidence]:
        """Return found items in requested order; omit unknown identifiers."""

    def list_all(self) -> list[Evidence]:
        """Return all registered items in deterministic identifier order."""


@runtime_checkable
class EvidenceRegistryFactory(Protocol):
    """Create an isolated evidence registry for one request."""

    def create(self, request_id: str) -> EvidenceRegistry:
        """Return a fresh registry scoped to the supplied request."""


class ConversationOwnershipError(EvidenceRegistryError):
    """Raised when an actor does not own the target conversation."""


class ConversationNotFoundError(EvidenceRegistryError):
    """Raised when a conversation has no registry."""


class ConversationClosedError(EvidenceRegistryError):
    """Raised when an operation requires an active conversation."""


class EvidenceConversationMismatchError(EvidenceRegistryError):
    """Raised when evidence belongs to another conversation."""


@runtime_checkable
class ConversationEvidenceRegistry(Protocol):
    """Evidence store scoped to one owned conversation."""

    @property
    def conversation_id(self) -> str:
        """Conversation identifier that scopes this registry."""

    @property
    def owner_id(self) -> str:
        """Owner identifier bound to this registry."""

    @property
    def status(self) -> str:
        """Current conversation lifecycle status."""

    def register(self, evidence: Evidence, *, actor_id: str) -> None:
        """Register evidence after ownership and conversation checks."""

    def get(self, evidence_id: str, *, actor_id: str) -> Evidence | None:
        """Retrieve evidence from this active conversation."""

    def get_many(self, evidence_ids: list[str], *, actor_id: str) -> list[Evidence]:
        """Retrieve requested evidence in input order."""

    def list_all(self, *, actor_id: str) -> list[Evidence]:
        """List all evidence in deterministic identifier order."""

    def list_by_turn(self, turn_id: str, *, actor_id: str) -> list[Evidence]:
        """List evidence originating from one conversation turn."""


@runtime_checkable
class ConversationEvidenceRegistryFactory(Protocol):
    """Shared factory managing registries across conversations."""

    def create(self, conversation, *, actor_id: str) -> ConversationEvidenceRegistry:
        """Return the shared registry for an active owned conversation."""

    def get(
        self, conversation_id: str, *, actor_id: str
    ) -> ConversationEvidenceRegistry:
        """Retrieve an existing registry after ownership validation."""

    def close_conversation(
        self,
        conversation_id: str,
        *,
        actor_id: str,
        closed_at=None,
    ) -> None:
        """Close a conversation and block subsequent evidence access."""

    def delete_conversation(self, conversation_id: str, *, actor_id: str) -> bool:
        """Delete a conversation registry and all of its evidence."""
