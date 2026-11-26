"""Request-scoped in-memory evidence registry."""

from __future__ import annotations

import json
from threading import RLock

from ..contracts.evidence import Evidence
from .interfaces import (
    DuplicateEvidenceError,
    EvidenceRequestMismatchError,
)


class InMemoryEvidenceRegistry:
    """Store isolated JSON snapshots for a single request.

    The registry stores serialized snapshots rather than caller-owned
    Pydantic objects. Mutating nested dictionaries in an input or returned
    Evidence object therefore cannot mutate the registered state.
    """

    def __init__(self, request_id: str) -> None:
        normalized_request_id = request_id.strip()
        if not normalized_request_id:
            raise ValueError("request_id must not be empty.")

        self._request_id = normalized_request_id
        self._records: dict[str, str] = {}
        self._lock = RLock()

    @property
    def request_id(self) -> str:
        return self._request_id

    @staticmethod
    def _serialize(evidence: Evidence) -> str:
        payload = evidence.model_dump(mode="json")
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize(payload: str) -> Evidence:
        return Evidence.model_validate_json(payload)

    def register(self, evidence: Evidence) -> None:
        if evidence.request_id != self._request_id:
            raise EvidenceRequestMismatchError(
                "Evidence request_id does not match this registry."
            )

        snapshot = self._serialize(evidence)

        with self._lock:
            if evidence.evidence_id in self._records:
                raise DuplicateEvidenceError(
                    f"Evidence already registered: {evidence.evidence_id}"
                )
            self._records[evidence.evidence_id] = snapshot

    def get(self, evidence_id: str) -> Evidence | None:
        with self._lock:
            payload = self._records.get(evidence_id)

        if payload is None:
            return None
        return self._deserialize(payload)

    def get_many(self, evidence_ids: list[str]) -> list[Evidence]:
        with self._lock:
            payloads = [
                self._records[evidence_id]
                for evidence_id in evidence_ids
                if evidence_id in self._records
            ]

        return [self._deserialize(payload) for payload in payloads]

    def list_all(self) -> list[Evidence]:
        with self._lock:
            payloads = [
                self._records[evidence_id] for evidence_id in sorted(self._records)
            ]

        return [self._deserialize(payload) for payload in payloads]


class InMemoryEvidenceRegistryFactory:
    """Create a fresh in-memory registry for each request."""

    def create(self, request_id: str) -> InMemoryEvidenceRegistry:
        return InMemoryEvidenceRegistry(request_id)


# ----------------------------------------------------------------------
# Conversation-scoped implementation.
# The request-scoped classes above remain temporarily available while
# existing pipeline consumers are migrated in a later stage.
# ----------------------------------------------------------------------

from datetime import UTC, datetime

from ..contracts.conversation import Conversation
from .interfaces import (
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationOwnershipError,
    EvidenceConversationMismatchError,
)


class ConversationScopedInMemoryEvidenceRegistry:
    """Shared, owner-protected evidence snapshots for one conversation."""

    def __init__(self, conversation: Conversation) -> None:
        if conversation.status != "active":
            raise ConversationClosedError(
                "A registry can only be created for an active conversation."
            )

        # Keep an independent metadata snapshot.
        self._conversation = Conversation.model_validate(
            conversation.model_dump(mode="python")
        )
        self._records: dict[str, str] = {}
        self._lock = RLock()
        self._deleted = False

    @property
    def conversation_id(self) -> str:
        return self._conversation.conversation_id

    @property
    def owner_id(self) -> str:
        return self._conversation.owner_id

    @property
    def status(self) -> str:
        with self._lock:
            return self._conversation.status

    def _authorize(self, actor_id: str) -> None:
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ConversationOwnershipError("A valid actor_id is required.")
        if actor_id != self.owner_id:
            raise ConversationOwnershipError(
                "The actor does not own this conversation."
            )
        if self._deleted:
            raise ConversationNotFoundError(
                "The conversation registry has been deleted."
            )

    def _require_active(self) -> None:
        if self._conversation.status != "active":
            raise ConversationClosedError(
                "The conversation is closed; evidence access is blocked."
            )

    @staticmethod
    def _serialize(evidence: Evidence) -> str:
        payload = evidence.model_dump(mode="json")
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize(payload: str) -> Evidence:
        return Evidence.model_validate_json(payload)

    def register(self, evidence: Evidence, *, actor_id: str) -> None:
        with self._lock:
            self._authorize(actor_id)
            self._require_active()

            if evidence.conversation_id != self.conversation_id:
                raise EvidenceConversationMismatchError(
                    "Evidence conversation_id does not match this registry."
                )

            snapshot = self._serialize(evidence)

            if evidence.evidence_id in self._records:
                raise DuplicateEvidenceError(
                    f"Evidence already registered: {evidence.evidence_id}"
                )

            # request_id and created_by_request_id are not rewritten.
            self._records[evidence.evidence_id] = snapshot

    def get(self, evidence_id: str, *, actor_id: str) -> Evidence | None:
        with self._lock:
            self._authorize(actor_id)
            self._require_active()
            payload = self._records.get(evidence_id)

        return None if payload is None else self._deserialize(payload)

    def get_many(self, evidence_ids: list[str], *, actor_id: str) -> list[Evidence]:
        with self._lock:
            self._authorize(actor_id)
            self._require_active()
            payloads = [
                self._records[evidence_id]
                for evidence_id in evidence_ids
                if evidence_id in self._records
            ]

        return [self._deserialize(payload) for payload in payloads]

    def list_all(self, *, actor_id: str) -> list[Evidence]:
        with self._lock:
            self._authorize(actor_id)
            self._require_active()
            payloads = [
                self._records[evidence_id] for evidence_id in sorted(self._records)
            ]

        return [self._deserialize(payload) for payload in payloads]

    def list_by_turn(self, turn_id: str, *, actor_id: str) -> list[Evidence]:
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("turn_id must not be empty.")

        records = self.list_all(actor_id=actor_id)
        return [item for item in records if item.origin_turn_id == turn_id]

    def _close(self, closed_at: datetime | None = None) -> None:
        with self._lock:
            if self._deleted:
                raise ConversationNotFoundError(
                    "The conversation registry has been deleted."
                )
            if self._conversation.status == "closed":
                return

            timestamp = closed_at or datetime.now(UTC)
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("closed_at must be timezone-aware.")

            # Preserve valid monotonic lifecycle timestamps.
            if timestamp < self._conversation.created_at:
                raise ValueError("closed_at must not precede created_at.")

            updated_at = max(timestamp, self._conversation.updated_at)
            self._conversation = self._conversation.model_copy(
                update={
                    "status": "closed",
                    "closed_at": timestamp,
                    "updated_at": updated_at,
                }
            )

    def _delete(self) -> None:
        with self._lock:
            self._records.clear()
            self._deleted = True


class ConversationScopedInMemoryEvidenceRegistryFactory:
    """One shared registry per conversation within this factory instance.

    Instantiate this factory once at application/process scope and inject
    the same instance into requests belonging to that application.
    """

    def __init__(self) -> None:
        self._registries: dict[str, ConversationScopedInMemoryEvidenceRegistry] = {}
        self._lock = RLock()

    @staticmethod
    def _validate_actor(conversation: Conversation, actor_id: str) -> None:
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ConversationOwnershipError("A valid actor_id is required.")
        if actor_id != conversation.owner_id:
            raise ConversationOwnershipError(
                "The actor does not own this conversation."
            )

    def create(
        self, conversation: Conversation, *, actor_id: str
    ) -> ConversationScopedInMemoryEvidenceRegistry:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation instance.")

        self._validate_actor(conversation, actor_id)

        if conversation.status != "active":
            raise ConversationClosedError(
                "Cannot create or retrieve a registry for a closed conversation."
            )

        with self._lock:
            existing = self._registries.get(conversation.conversation_id)

            if existing is not None:
                existing._authorize(actor_id)
                existing._require_active()
                return existing

            registry = ConversationScopedInMemoryEvidenceRegistry(conversation)
            self._registries[conversation.conversation_id] = registry
            return registry

    def get(
        self, conversation_id: str, *, actor_id: str
    ) -> ConversationScopedInMemoryEvidenceRegistry:
        with self._lock:
            registry = self._registries.get(conversation_id)
            if registry is None:
                raise ConversationNotFoundError(
                    "No registry exists for this conversation."
                )
            registry._authorize(actor_id)
            return registry

    def close_conversation(
        self,
        conversation_id: str,
        *,
        actor_id: str,
        closed_at: datetime | None = None,
    ) -> None:
        registry = self.get(conversation_id, actor_id=actor_id)
        registry._close(closed_at)

    def delete_conversation(self, conversation_id: str, *, actor_id: str) -> bool:
        with self._lock:
            registry = self._registries.get(conversation_id)
            if registry is None:
                return False

            registry._authorize(actor_id)
            registry._delete()
            del self._registries[conversation_id]
            return True
