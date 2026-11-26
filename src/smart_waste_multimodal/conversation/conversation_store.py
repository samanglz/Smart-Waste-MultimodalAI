"""Protocol defining conversation storage operations."""

from typing import Protocol

from smart_waste_multimodal.contracts import Conversation, ConversationTurn
from smart_waste_multimodal.contracts.vision import VisionResult


class ConversationStore(Protocol):
    """Storage interface for conversations and their turn history."""

    def create(self, *, actor_id: str) -> Conversation:
        """Create a new conversation owned by the specified actor."""

    def get(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> Conversation:
        """Retrieve an active conversation after verifying ownership."""

    def append_turn(
        self,
        conversation_id: str,
        turn: ConversationTurn,
        *,
        actor_id: str,
    ) -> ConversationTurn:
        """Append a turn to an active conversation after validation."""

    def update_turn(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        actor_id: str,
        vision: VisionResult | None = None,
        evidence_ids: tuple[str, ...] | None = None,
    ) -> ConversationTurn:
        """Update pipeline results attached to an existing turn.

        Fields passed as None are left unchanged.
        """

    def get_history(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> tuple[ConversationTurn, ...]:
        """Return the ordered turn history of an active conversation."""

    def close(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> Conversation:
        """Close an active conversation."""

    def delete(
        self,
        conversation_id: str,
        *,
        actor_id: str,
    ) -> bool:
        """Delete a conversation and its history; return False if absent."""
