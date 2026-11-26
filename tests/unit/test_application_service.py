"""Focused tests for ApplicationService artifact ownership and identity."""

from datetime import UTC, datetime

from smart_waste_multimodal.api.application_service import (
    AnalyzeApplicationService,
)
from smart_waste_multimodal.contracts import ConversationTurn, RequestOptions


def make_turn(
    *,
    turn_id="turn-001",
    conversation_id="conversation-001",
    role="user",
    artifact_id="artifact-001",
):
    return ConversationTurn(
        turn_id=turn_id,
        conversation_id=conversation_id,
        request_id="request-001",
        role=role,
        content="Image submitted for analysis.",
        timestamp=datetime.now(UTC),
        image_artifact_id=artifact_id,
    )


def test_history_authorizes_artifact_referenced_by_user_turn():
    history = (make_turn(),)

    assert AnalyzeApplicationService._history_references_artifact(
        history=history,
        conversation_id="conversation-001",
        artifact_id="artifact-001",
    )


def test_history_rejects_artifact_from_another_conversation():
    history = (make_turn(conversation_id="conversation-002"),)

    assert not AnalyzeApplicationService._history_references_artifact(
        history=history,
        conversation_id="conversation-001",
        artifact_id="artifact-001",
    )


def test_history_rejects_artifact_referenced_only_by_assistant_turn():
    history = (make_turn(role="assistant"),)

    assert not AnalyzeApplicationService._history_references_artifact(
        history=history,
        conversation_id="conversation-001",
        artifact_id="artifact-001",
    )


def test_history_rejects_different_artifact_id():
    history = (make_turn(artifact_id="artifact-002"),)

    assert not AnalyzeApplicationService._history_references_artifact(
        history=history,
        conversation_id="conversation-001",
        artifact_id="artifact-001",
    )


def test_fingerprint_scopes_artifact_identity_to_conversation():
    options = RequestOptions()

    first = AnalyzeApplicationService._fingerprint(
        conversation_id="conversation-001",
        task_type="detection",
        question=None,
        locale="en",
        options=options,
        image_digest=None,
        image_media_type=None,
        image_artifact_id="artifact-001",
    )
    second = AnalyzeApplicationService._fingerprint(
        conversation_id="conversation-002",
        task_type="detection",
        question=None,
        locale="en",
        options=options,
        image_digest=None,
        image_media_type=None,
        image_artifact_id="artifact-001",
    )

    assert first != second


def test_fingerprint_is_stable_for_same_artifact_reference():
    options = RequestOptions()

    first = AnalyzeApplicationService._fingerprint(
        conversation_id="conversation-001",
        task_type="detection",
        question=None,
        locale="en",
        options=options,
        image_digest=None,
        image_media_type=None,
        image_artifact_id="artifact-001",
    )
    second = AnalyzeApplicationService._fingerprint(
        conversation_id="conversation-001",
        task_type="detection",
        question=None,
        locale="en",
        options=options,
        image_digest=None,
        image_media_type=None,
        image_artifact_id="artifact-001",
    )

    assert first == second


def test_fingerprint_distinguishes_different_artifact_references():
    options = RequestOptions()

    first = AnalyzeApplicationService._fingerprint(
        conversation_id="conversation-001",
        task_type="detection",
        question=None,
        locale="en",
        options=options,
        image_digest=None,
        image_media_type=None,
        image_artifact_id="artifact-001",
    )
    second = AnalyzeApplicationService._fingerprint(
        conversation_id="conversation-001",
        task_type="detection",
        question=None,
        locale="en",
        options=options,
        image_digest=None,
        image_media_type=None,
        image_artifact_id="artifact-002",
    )

    assert first != second
