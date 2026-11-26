"""Unit tests for InMemoryConversationStore."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

import smart_waste_multimodal.conversation.in_memory_conversation_store as store_module
from smart_waste_multimodal.conversation.in_memory_conversation_store import (
    InMemoryConversationStore,
)
from smart_waste_multimodal.conversation.conversation_store_errors import (
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationOwnershipError,
    ConversationTurnMismatchError,
    ConversationTurnOrderError,
    DuplicateConversationTurnError,
)
from smart_waste_multimodal.contracts.conversation import ConversationTurn
from smart_waste_multimodal.contracts.vision import VisionResult
from pydantic import ValidationError


def make_turn(
    turn_id: str = "turn-001",
    *,
    timestamp: datetime | None = None,
    conversation_id: str | None = None,
    role: str = "user",
    content: str = "Test message",
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        timestamp=timestamp or datetime.now(UTC),
    )


def create_store_and_conversation(
    actor_id: str = "owner-001",
) -> tuple[InMemoryConversationStore, object]:
    store = InMemoryConversationStore()
    conversation = store.create(actor_id=actor_id)
    return store, conversation


def test_create_returns_active_conversation_with_owner():
    store = InMemoryConversationStore()

    conversation = store.create(actor_id="owner-001")

    assert conversation.owner_id == "owner-001"
    assert conversation.status == "active"
    assert conversation.closed_at is None
    assert conversation.created_at.tzinfo is not None
    assert conversation.updated_at >= conversation.created_at
    assert store.get(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    ) == conversation


def test_create_generates_unique_conversation_ids():
    store = InMemoryConversationStore()

    first = store.create(actor_id="owner-001")
    second = store.create(actor_id="owner-001")

    assert first.conversation_id != second.conversation_id


def test_create_rejects_noncanonical_actor_ids():
    store = InMemoryConversationStore()

    for actor_id in ("", " ", " owner-001", "owner-001 ", " owner-001 "):
        with pytest.raises(ConversationOwnershipError):
            store.create(actor_id=actor_id)


@pytest.mark.parametrize("actor_id", [None, 123, [], {}])
def test_create_rejects_non_string_actor_ids(actor_id):
    store = InMemoryConversationStore()

    with pytest.raises(ConversationOwnershipError):
        store.create(actor_id=actor_id)


def test_create_rejects_actor_id_longer_than_200_characters():
    store = InMemoryConversationStore()

    with pytest.raises(ConversationOwnershipError):
        store.create(actor_id="a" * 201)


def test_create_retries_when_uuid_collides(monkeypatch):
    store = InMemoryConversationStore()
    existing = store.create(actor_id="owner-001")

    colliding_uuid = UUID(existing.conversation_id)
    replacement_uuid = UUID("12345678-1234-5678-1234-567812345678")
    generated = iter([colliding_uuid, replacement_uuid])

    monkeypatch.setattr(store_module, "uuid4", lambda: next(generated))

    created = store.create(actor_id="owner-001")

    assert created.conversation_id == str(replacement_uuid)
    assert created.conversation_id != existing.conversation_id
    assert store.get(
        conversation_id=existing.conversation_id,
        actor_id="owner-001",
    ) == existing


def test_create_returns_isolated_conversation_snapshot():
    store, created = create_store_and_conversation()

    created.owner_id = "modified-owner"

    stored = store.get(
        conversation_id=created.conversation_id,
        actor_id="owner-001",
    )
    assert stored.owner_id == "owner-001"


def test_get_rejects_unknown_conversation():
    store = InMemoryConversationStore()

    with pytest.raises(ConversationNotFoundError):
        store.get(conversation_id="missing", actor_id="owner-001")


def test_get_rejects_non_owner():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationOwnershipError):
        store.get(
            conversation_id=conversation.conversation_id,
            actor_id="another-owner",
        )


def test_get_rejects_invalid_actor_id():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationOwnershipError):
        store.get(
            conversation_id=conversation.conversation_id,
            actor_id=" owner-001 ",
        )


def test_append_turn_sets_missing_conversation_id():
    store, conversation = create_store_and_conversation()
    turn = make_turn(conversation_id=None)

    stored = store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=turn,
    )

    assert stored.conversation_id == conversation.conversation_id
    assert store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    ) == (stored,)


def test_append_turn_rejects_mismatched_conversation_id():
    store, conversation = create_store_and_conversation()
    turn = make_turn(conversation_id="another-conversation")

    with pytest.raises(ConversationTurnMismatchError):
        store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=turn,
        )


def test_append_turn_rejects_duplicate_turn_id():
    store, conversation = create_store_and_conversation()
    timestamp = datetime.now(UTC)

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn("duplicate", timestamp=timestamp),
    )

    with pytest.raises(DuplicateConversationTurnError):
        store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=make_turn("duplicate", timestamp=timestamp),
        )


def test_append_turn_rejects_timestamp_older_than_last_turn():
    store, conversation = create_store_and_conversation()
    first_timestamp = datetime.now(UTC)

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn("first", timestamp=first_timestamp),
    )

    with pytest.raises(ConversationTurnOrderError):
        store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=make_turn(
                "second",
                timestamp=first_timestamp - timedelta(seconds=1),
            ),
        )


def test_append_turn_rejects_timestamp_before_conversation_creation():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationTurnOrderError):
        store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=make_turn(
                "earlier",
                timestamp=conversation.created_at - timedelta(seconds=1),
            ),
        )


def test_append_turn_accepts_equal_timestamps():
    store, conversation = create_store_and_conversation()
    timestamp = datetime.now(UTC)

    first = store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn("first", timestamp=timestamp),
    )
    second = store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn("second", timestamp=timestamp),
    )

    assert [turn.turn_id for turn in store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )] == [first.turn_id, second.turn_id]


def test_append_turn_updates_conversation_updated_at():
    store, conversation = create_store_and_conversation()
    timestamp = datetime.now(UTC) + timedelta(seconds=2)

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(timestamp=timestamp),
    )

    updated = store.get(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )
    assert updated.updated_at >= timestamp


def test_append_turn_does_not_mutate_input_turn():
    store, conversation = create_store_and_conversation()
    turn = make_turn(conversation_id=None)

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=turn,
    )

    assert turn.conversation_id is None


def test_get_history_returns_ordered_turns():
    store, conversation = create_store_and_conversation()
    start = datetime.now(UTC)

    for index in range(3):
        store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=make_turn(
                f"turn-{index}",
                timestamp=start + timedelta(seconds=index),
            ),
        )

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert [turn.turn_id for turn in history] == [
        "turn-0", "turn-1", "turn-2"
    ]
    assert isinstance(history, tuple)


def test_append_and_get_history_preserve_image_artifact_id():
    store, conversation = create_store_and_conversation()
    timestamp = datetime.now(UTC)

    turn = ConversationTurn(
        turn_id="turn-with-image",
        conversation_id=conversation.conversation_id,
        role="user",
        content="What material is visible?",
        timestamp=timestamp,
        image_artifact_id="image-artifact-001",
    )

    stored = store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=turn,
    )

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert stored.image_artifact_id == "image-artifact-001"
    assert history[0].image_artifact_id == "image-artifact-001"


def test_get_history_returns_isolated_turn_snapshots():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(content="Original content"),
    )

    first_read = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )
    first_read[0].content = "Modified returned snapshot"

    second_read = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert second_read[0].content == "Original content"


def test_get_history_rejects_non_owner():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationOwnershipError):
        store.get_history(
            conversation_id=conversation.conversation_id,
            actor_id="another-owner",
        )


def test_close_marks_conversation_closed():
    store, conversation = create_store_and_conversation()

    closed = store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert closed.status == "closed"
    assert closed.closed_at is not None
    assert closed.updated_at == closed.closed_at
    assert closed.closed_at >= conversation.created_at


def test_close_rejects_second_close():
    store, conversation = create_store_and_conversation()

    store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    with pytest.raises(ConversationClosedError):
        store.close(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
        )


def test_closed_conversation_rejects_get():
    store, conversation = create_store_and_conversation()

    store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    with pytest.raises(ConversationClosedError):
        store.get(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
        )


def test_closed_conversation_rejects_history_access():
    store, conversation = create_store_and_conversation()

    store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    with pytest.raises(ConversationClosedError):
        store.get_history(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
        )


def test_closed_conversation_rejects_append():
    store, conversation = create_store_and_conversation()

    store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    with pytest.raises(ConversationClosedError):
        store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=make_turn(),
        )


def test_close_rejects_non_owner():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationOwnershipError):
        store.close(
            conversation_id=conversation.conversation_id,
            actor_id="another-owner",
        )


def test_delete_removes_conversation_and_history():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(),
    )

    assert store.delete(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    ) is True

    with pytest.raises(ConversationNotFoundError):
        store.get(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
        )


def test_delete_missing_conversation_returns_false():
    store = InMemoryConversationStore()

    assert store.delete(
        conversation_id="missing",
        actor_id="owner-001",
    ) is False


def test_delete_rejects_non_owner():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationOwnershipError):
        store.delete(
            conversation_id=conversation.conversation_id,
            actor_id="another-owner",
        )


def test_delete_can_remove_closed_conversation():
    store, conversation = create_store_and_conversation()

    store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert store.delete(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    ) is True


def test_concurrent_appends_preserve_all_distinct_turns():
    store, conversation = create_store_and_conversation()
    timestamp = datetime.now(UTC)

    def append(index: int):
        return store.append_turn(
            conversation_id=conversation.conversation_id,
            actor_id="owner-001",
            turn=make_turn(
                f"concurrent-{index}",
                timestamp=timestamp,
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(append, range(24)))

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert len(results) == 24
    assert len(history) == 24
    assert len({turn.turn_id for turn in history}) == 24


def test_concurrent_duplicate_turn_id_is_accepted_exactly_once():
    store, conversation = create_store_and_conversation()
    timestamp = datetime.now(UTC)

    def append_same_turn(_):
        try:
            store.append_turn(
                conversation_id=conversation.conversation_id,
                actor_id="owner-001",
                turn=make_turn("same-turn", timestamp=timestamp),
            )
            return "accepted"
        except DuplicateConversationTurnError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(append_same_turn, range(8)))

    assert outcomes.count("accepted") == 1
    assert outcomes.count("duplicate") == 7

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )
    assert len(history) == 1
    assert history[0].turn_id == "same-turn"


def make_vision_result() -> VisionResult:
    return VisionResult(
        status="no_detections",
        detections=[],
        image_width=100,
        image_height=100,
        model_version="test-model",
    )


def test_update_turn_attaches_vision_and_evidence_ids():
    store, conversation = create_store_and_conversation()
    original = make_turn(
        turn_id="user-turn-001",
        content="Analyze this image",
    )

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=original,
    )

    vision = make_vision_result()

    updated = store.update_turn(
        conversation_id=conversation.conversation_id,
        turn_id="user-turn-001",
        actor_id="owner-001",
        vision=vision,
        evidence_ids=("evidence-001", "evidence-002"),
    )

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert updated.vision == vision
    assert updated.evidence_ids == ("evidence-001", "evidence-002")
    assert history[0].vision == vision
    assert history[0].evidence_ids == ("evidence-001", "evidence-002")


def test_update_turn_preserves_fields_not_included_in_update():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(
            turn_id="user-turn-002",
            content="Original content",
        ),
    )

    vision = make_vision_result()

    store.update_turn(
        conversation_id=conversation.conversation_id,
        turn_id="user-turn-002",
        actor_id="owner-001",
        vision=vision,
        evidence_ids=("evidence-001",),
    )

    updated = store.update_turn(
        conversation_id=conversation.conversation_id,
        turn_id="user-turn-002",
        actor_id="owner-001",
        evidence_ids=("evidence-002",),
    )

    assert updated.vision == vision
    assert updated.evidence_ids == ("evidence-002",)
    assert updated.content == "Original content"


def test_update_turn_preserves_original_turn_metadata():
    store, conversation = create_store_and_conversation()
    original = make_turn(
        turn_id="user-turn-003",
        content="Keep this content",
    )

    stored = store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=original,
    )

    updated = store.update_turn(
        conversation_id=conversation.conversation_id,
        turn_id="user-turn-003",
        actor_id="owner-001",
        evidence_ids=("evidence-001",),
    )

    assert updated.turn_id == stored.turn_id
    assert updated.conversation_id == stored.conversation_id
    assert updated.request_id == stored.request_id
    assert updated.role == stored.role
    assert updated.content == stored.content
    assert updated.timestamp == stored.timestamp
    assert updated.image_artifact_id == stored.image_artifact_id


def test_update_turn_rejects_duplicate_evidence_ids_atomically():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(turn_id="user-turn-004"),
    )

    with pytest.raises(ValidationError):
        store.update_turn(
            conversation_id=conversation.conversation_id,
            turn_id="user-turn-004",
            actor_id="owner-001",
            evidence_ids=("duplicate", "duplicate"),
        )

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert history[0].evidence_ids == ()
    assert history[0].vision is None


def test_update_turn_returns_isolated_snapshot():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(turn_id="user-turn-005"),
    )

    returned = store.update_turn(
        conversation_id=conversation.conversation_id,
        turn_id="user-turn-005",
        actor_id="owner-001",
        evidence_ids=("evidence-001",),
    )

    returned.content = "Mutated returned snapshot"

    history = store.get_history(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    assert history[0].content == "Test message"
    assert history[0].evidence_ids == ("evidence-001",)


def test_update_turn_rejects_non_owner():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(turn_id="user-turn-006"),
    )

    with pytest.raises(ConversationOwnershipError):
        store.update_turn(
            conversation_id=conversation.conversation_id,
            turn_id="user-turn-006",
            actor_id="another-owner",
            evidence_ids=("evidence-001",),
        )


def test_update_turn_rejects_closed_conversation():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(turn_id="user-turn-007"),
    )

    store.close(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
    )

    with pytest.raises(ConversationClosedError):
        store.update_turn(
            conversation_id=conversation.conversation_id,
            turn_id="user-turn-007",
            actor_id="owner-001",
            evidence_ids=("evidence-001",),
        )


def test_update_turn_rejects_unknown_turn():
    store, conversation = create_store_and_conversation()

    with pytest.raises(ConversationNotFoundError):
        store.update_turn(
            conversation_id=conversation.conversation_id,
            turn_id="missing-turn",
            actor_id="owner-001",
            evidence_ids=("evidence-001",),
        )


def test_update_turn_with_no_changes_returns_existing_snapshot():
    store, conversation = create_store_and_conversation()

    store.append_turn(
        conversation_id=conversation.conversation_id,
        actor_id="owner-001",
        turn=make_turn(turn_id="user-turn-008"),
    )

    updated = store.update_turn(
        conversation_id=conversation.conversation_id,
        turn_id="user-turn-008",
        actor_id="owner-001",
    )

    assert updated.turn_id == "user-turn-008"
    assert updated.evidence_ids == ()
    assert updated.vision is None
