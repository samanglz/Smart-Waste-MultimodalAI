"""Thread-safe in-memory implementation of ConversationStore."""

from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from smart_waste_multimodal.contracts.conversation import (
    Conversation,
    ConversationTurn,
)

from .conversation_store import ConversationStore
from .conversation_store_errors import (
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationOwnershipError,
    ConversationTurnMismatchError,
    ConversationTurnOrderError,
    DuplicateConversationTurnError,
)


class InMemoryConversationStore:
    """Store conversations and their ordered turn history in memory."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._lock = RLock()

    def create(self, *, actor_id: str) -> Conversation:
        """Create a new active conversation owned by actor_id."""
        self._validate_actor_id(actor_id)

        with self._lock:
            conversation_id = str(uuid4())
            while conversation_id in self._conversations:
                conversation_id = str(uuid4())

            now = datetime.now(UTC)
            conversation = Conversation(
                conversation_id=conversation_id,
                owner_id=actor_id,
                status="active",
                created_at=now,
                updated_at=now,
            )

            self._conversations[conversation_id] = (
                conversation.model_copy(deep=True)
            )
            self._turns[conversation_id] = []

            return conversation.model_copy(deep=True)

    def get(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> Conversation:
        """Retrieve an active conversation after verifying ownership."""
        with self._lock:
            conversation = self._get_owned_conversation(
                conversation_id,
                actor_id=actor_id,
            )
            self._ensure_active(conversation)
            return conversation.model_copy(deep=True)

    def append_turn(
        self,
        conversation_id: str,
        turn: ConversationTurn,
        *,
        actor_id: str,
    ) -> ConversationTurn:
        """Append a validated turn to an active conversation."""
        with self._lock:
            conversation = self._get_owned_conversation(
                conversation_id,
                actor_id=actor_id,
            )
            self._ensure_active(conversation)

            if not isinstance(turn, ConversationTurn):
                raise TypeError("turn must be a ConversationTurn instance.")

            if (
                turn.conversation_id is not None
                and turn.conversation_id != conversation_id
            ):
                raise ConversationTurnMismatchError(
                    "The turn belongs to a different conversation."
                )

            stored_turns = self._turns[conversation_id]

            if any(
                existing.turn_id == turn.turn_id
                for existing in stored_turns
            ):
                raise DuplicateConversationTurnError(
                    f"Turn ID already exists: {turn.turn_id}"
                )

            if stored_turns and turn.timestamp < stored_turns[-1].timestamp:
                raise ConversationTurnOrderError(
                    "Turn timestamp cannot precede the latest stored turn."
                )

            if turn.timestamp < conversation.created_at:
                raise ConversationTurnOrderError(
                    "Turn timestamp cannot precede conversation creation."
                )

            normalized_turn = ConversationTurn.model_validate(
                {
                    **turn.model_dump(mode="python"),
                    "conversation_id": conversation_id,
                }
            )

            updated_at = max(datetime.now(UTC), normalized_turn.timestamp)

            updated_conversation = Conversation.model_validate(
                {
                    **conversation.model_dump(mode="python"),
                    "updated_at": updated_at,
                }
            )

            stored_turn = normalized_turn.model_copy(deep=True)

            # Commit both changes while holding the same lock.
            stored_turns.append(stored_turn)
            self._conversations[conversation_id] = updated_conversation

            return stored_turn.model_copy(deep=True)

    def update_turn(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        actor_id: str,
        vision=None,
        evidence_ids: tuple[str, ...] | None = None,
    ) -> ConversationTurn:
        """Update pipeline results on an existing turn atomically.

        A None value means that the corresponding field is unchanged.
        The stored turn and conversation metadata are committed under
        the same lock.
        """
        with self._lock:
            conversation = self._get_owned_conversation(
                conversation_id,
                actor_id=actor_id,
            )
            self._ensure_active(conversation)

            if not isinstance(turn_id, str) or not turn_id.strip():
                raise ValueError("turn_id must be a non-empty string.")

            stored_turns = self._turns[conversation_id]

            turn_index = next(
                (
                    index
                    for index, existing in enumerate(stored_turns)
                    if existing.turn_id == turn_id
                ),
                None,
            )

            if turn_index is None:
                raise ConversationNotFoundError(
                    f"Turn not found: {turn_id}"
                )

            current_turn = stored_turns[turn_index]
            updated_data = current_turn.model_dump(mode="python")

            has_changes = False

            if vision is not None:
                updated_data["vision"] = vision
                has_changes = True

            if evidence_ids is not None:
                updated_data["evidence_ids"] = evidence_ids
                has_changes = True

            if not has_changes:
                return current_turn.model_copy(deep=True)

            # Validate the complete replacement before mutating the store.
            updated_turn = ConversationTurn.model_validate(updated_data)

            updated_at = max(datetime.now(UTC), conversation.updated_at)

            updated_conversation = Conversation.model_validate(
                {
                    **conversation.model_dump(mode="python"),
                    "updated_at": updated_at,
                }
            )

            # Commit both changes while holding the same lock.
            stored_turns[turn_index] = updated_turn
            self._conversations[conversation_id] = updated_conversation

            return updated_turn.model_copy(deep=True)

    def get_history(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> tuple[ConversationTurn, ...]:
        """Return a deep-copied chronological history."""
        with self._lock:
            conversation = self._get_owned_conversation(
                conversation_id,
                actor_id=actor_id,
            )
            self._ensure_active(conversation)

            return tuple(
                turn.model_copy(deep=True)
                for turn in self._turns[conversation_id]
            )

    def close(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> Conversation:
        """Close an active conversation."""
        with self._lock:
            conversation = self._get_owned_conversation(
                conversation_id,
                actor_id=actor_id,
            )
            self._ensure_active(conversation)

            closed_at = max(datetime.now(UTC), conversation.updated_at)

            closed_conversation = Conversation.model_validate(
                {
                    **conversation.model_dump(mode="python"),
                    "status": "closed",
                    "updated_at": closed_at,
                    "closed_at": closed_at,
                }
            )

            self._conversations[conversation_id] = closed_conversation

            return closed_conversation.model_copy(deep=True)

    def delete(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> bool:
        """Delete a conversation and its history."""
        self._validate_actor_id(actor_id)

        with self._lock:
            conversation = self._conversations.get(conversation_id)

            if conversation is None:
                return False

            self._authorize(conversation, actor_id)

            del self._conversations[conversation_id]
            del self._turns[conversation_id]

            return True

    def _get_owned_conversation(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> Conversation:
        """Retrieve a conversation and enforce ownership."""
        self._validate_actor_id(actor_id)

        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ConversationNotFoundError(
                "Conversation ID is invalid or missing."
            )

        conversation = self._conversations.get(conversation_id)

        if conversation is None:
            raise ConversationNotFoundError(
                f"Conversation not found: {conversation_id}"
            )

        self._authorize(conversation, actor_id)
        return conversation

    @staticmethod
    def _validate_actor_id(actor_id: str) -> None:
        """Reject missing or invalid actor identifiers."""
        if (
            not isinstance(actor_id, str)
            or not actor_id
            or actor_id != actor_id.strip()
            or len(actor_id) > 200
        ):
            raise ConversationOwnershipError(
                "A valid actor ID is required."
            )

    @staticmethod
    def _authorize(
        conversation: Conversation,
        actor_id: str,
    ) -> None:
        """Ensure the actor owns the conversation."""
        if conversation.owner_id != actor_id:
            raise ConversationOwnershipError(
                "The actor does not own this conversation."
            )

    @staticmethod
    def _ensure_active(conversation: Conversation) -> None:
        """Reject operations that require an active conversation."""
        if conversation.status != "active":
            raise ConversationClosedError(
                "The conversation is closed."
            )
