"""Integration tests for successful application-service execution."""

from io import BytesIO

from PIL import Image

from smart_waste_multimodal.adapters.mock_adapters import (
    MockGenerationAdapter,
    MockRetrievalAdapter,
    MockVisionAdapter,
)
from smart_waste_multimodal.api.application_service import (
    AnalyzeApplicationService,
)
from smart_waste_multimodal.contracts import RequestOptions
from smart_waste_multimodal.conversation.in_memory_conversation_store import (
    InMemoryConversationStore,
)
from smart_waste_multimodal.evidence.in_memory_registry import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)
from smart_waste_multimodal.request_execution.in_memory_request_execution_store import (
    InMemoryRequestExecutionStore,
)
from smart_waste_multimodal.vision.in_memory_image_store import (
    InMemoryImageStore,
)


ACTOR_ID = "application-service-integration-actor"


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (80, 40), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_service():
    conversation_store = InMemoryConversationStore()
    image_store = InMemoryImageStore()
    request_execution_store = InMemoryRequestExecutionStore()
    evidence_factory = ConversationScopedInMemoryEvidenceRegistryFactory()

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=MockRetrievalAdapter(),
        generation_stage=MockGenerationAdapter(),
        evidence_registry_factory=evidence_factory,
    )

    service = AnalyzeApplicationService(
        conversation_store=conversation_store,
        image_store=image_store,
        request_execution_store=request_execution_store,
        evidence_registry_factory=evidence_factory,
        orchestrator=orchestrator,
    )

    conversation = service.create_conversation(actor_id=ACTOR_ID)

    return (
        service,
        conversation,
        conversation_store,
        image_store,
        request_execution_store,
    )


def test_text_request_completes_and_persists_conversation_turns():
    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    response = service.execute(
        actor_id=ACTOR_ID,
        conversation_id=conversation.conversation_id,
        request_id="integration-text-request-001",
        task_type="knowledge_question",
        question="How should clean plastic containers be recycled?",
        options=RequestOptions(run_vision=False),
    )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        "integration-text-request-001",
        actor_id=ACTOR_ID,
    )

    assert response.status == "success"
    assert response.request_id == "integration-text-request-001"
    assert response.generation is not None
    assert response.generation.status == "success"

    assert len(history) == 2
    user_turn, assistant_turn = history

    assert user_turn.role == "user"
    assert user_turn.request_id == response.request_id
    assert user_turn.content == "How should clean plastic containers be recycled?"
    assert user_turn.vision is None

    assert assistant_turn.role == "assistant"
    assert assistant_turn.request_id == response.request_id
    assert assistant_turn.content
    assert execution.status == "COMPLETED"
    assert execution.response == response


def test_image_request_registers_artifact_and_persists_vision_result():
    (
        service,
        conversation,
        conversation_store,
        image_store,
        execution_store,
    ) = make_service()

    image_bytes = make_png()

    response = service.execute(
        actor_id=ACTOR_ID,
        conversation_id=conversation.conversation_id,
        request_id="integration-image-request-001",
        task_type="detection",
        question="How should this item be recycled?",
        image_bytes=image_bytes,
        image_media_type="image/png",
    )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        "integration-image-request-001",
        actor_id=ACTOR_ID,
    )

    assert response.status == "success"
    assert response.image_provided is True
    assert response.vision is not None
    assert response.vision.status == "success"
    assert response.vision.image_width == 80
    assert response.vision.image_height == 40

    assert len(history) == 2
    user_turn, assistant_turn = history

    assert user_turn.role == "user"
    assert user_turn.image_artifact_id is not None
    assert user_turn.vision == response.vision

    artifact_id = user_turn.image_artifact_id
    assert image_store.contains(artifact_id)
    assert image_store.get_bytes(artifact_id) == image_bytes
    assert image_store.get_metadata(artifact_id).sha256 is not None

    assert assistant_turn.role == "assistant"
    assert assistant_turn.content
    assert execution.status == "COMPLETED"
    assert execution.response == response

def test_duplicate_request_returns_stored_response():
    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    request = {
        "actor_id": ACTOR_ID,
        "conversation_id": conversation.conversation_id,
        "request_id": "integration-idempotency-request-001",
        "task_type": "knowledge_question",
        "question": "How should clean plastic containers be recycled?",
        "options": RequestOptions(run_vision=False),
    }

    first_response = service.execute(**request)
    history_after_first = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )

    second_response = service.execute(**request)
    history_after_second = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )

    execution = execution_store.get(
        request["request_id"],
        actor_id=ACTOR_ID,
    )

    assert second_response == first_response
    assert len(history_after_first) == 2
    assert len(history_after_second) == 2
    assert [turn.role for turn in history_after_second] == [
        "user",
        "assistant",
    ]
    assert execution.status == "COMPLETED"
    assert execution.response == first_response

def test_execute_rejects_conversation_owned_by_another_actor():
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceError,
    )

    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    other_actor_id = "different-integration-actor"
    request_id = "integration-ownership-request-001"

    with pytest.raises(ApplicationServiceError):
        service.execute(
            actor_id=other_actor_id,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="knowledge_question",
            question="How should clean plastic containers be recycled?",
            options=RequestOptions(run_vision=False),
        )

    owner_history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        request_id,
        actor_id=other_actor_id,
    )

    assert owner_history == ()
    assert execution.status == "FAILED"
    assert execution.response is None
    assert execution.safe_error is not None
    assert execution.safe_error.stage == "input"

def test_image_is_compensated_when_user_turn_append_fails(monkeypatch):
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceError,
    )

    (
        service,
        conversation,
        conversation_store,
        image_store,
        execution_store,
    ) = make_service()

    def fail_append(*args, **kwargs):
        raise RuntimeError("Injected user-turn persistence failure.")

    monkeypatch.setattr(conversation_store, "append_turn", fail_append)

    request_id = "integration-user-append-failure-001"

    with pytest.raises(ApplicationServiceError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="detection",
            question="Identify this waste item.",
            image_bytes=make_png(),
            image_media_type="image/png",
        )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(request_id, actor_id=ACTOR_ID)

    assert history == ()
    assert image_store.count == 0
    assert execution.status == "FAILED"
    assert execution.safe_error is not None


def test_orchestrator_failure_preserves_user_turn_and_image(monkeypatch):
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceError,
    )

    (
        service,
        conversation,
        conversation_store,
        image_store,
        execution_store,
    ) = make_service()

    def fail_run(request, *, conversation, actor_id=None):
        raise RuntimeError("Injected orchestrator failure.")

    monkeypatch.setattr(service._orchestrator, "run", fail_run)

    image_bytes = make_png()
    request_id = "integration-orchestrator-failure-001"

    with pytest.raises(ApplicationServiceError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="detection",
            question="Identify this waste item.",
            image_bytes=image_bytes,
            image_media_type="image/png",
        )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(request_id, actor_id=ACTOR_ID)

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].image_artifact_id is not None
    assert image_store.contains(history[0].image_artifact_id)
    assert image_store.get_bytes(history[0].image_artifact_id) == image_bytes
    assert execution.status == "FAILED"
    assert execution.response is None
    assert execution.safe_error is not None


def test_assistant_turn_append_failure_preserves_updated_user_turn(
    monkeypatch,
):
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceError,
    )

    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    original_append = conversation_store.append_turn

    def fail_assistant_append(conversation_id, turn, *, actor_id):
        if turn.role == "assistant":
            raise RuntimeError("Injected assistant-turn persistence failure.")
        return original_append(
            conversation_id,
            turn,
            actor_id=actor_id,
        )

    monkeypatch.setattr(
        conversation_store,
        "append_turn",
        fail_assistant_append,
    )

    request_id = "integration-assistant-append-failure-001"

    with pytest.raises(ApplicationServiceError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="knowledge_question",
            question="How should clean plastic containers be recycled?",
            options=RequestOptions(run_vision=False),
        )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(request_id, actor_id=ACTOR_ID)

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].evidence_ids
    assert execution.status == "FAILED"
    assert execution.response is None
    assert execution.safe_error is not None


def test_finalization_failure_after_assistant_persistence_requires_reconciliation(
    monkeypatch,
):
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceConsistencyError,
    )

    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    def fail_complete(*args, **kwargs):
        raise RuntimeError("Injected execution finalization failure.")

    monkeypatch.setattr(execution_store, "complete", fail_complete)

    request_id = "integration-finalization-failure-001"

    with pytest.raises(ApplicationServiceConsistencyError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="knowledge_question",
            question="How should clean plastic containers be recycled?",
            options=RequestOptions(run_vision=False),
        )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(request_id, actor_id=ACTOR_ID)

    assert len(history) == 2
    assert [turn.role for turn in history] == ["user", "assistant"]
    assert execution.status == "PROCESSING"
    assert execution.response is None

def test_failed_duplicate_request_is_not_executed_again(monkeypatch):
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceError,
        PreviouslyFailedRequestError,
    )

    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    def fail_run(request, *, conversation, actor_id=None):
        raise RuntimeError("Injected orchestrator failure.")

    monkeypatch.setattr(service._orchestrator, "run", fail_run)

    request = {
        "actor_id": ACTOR_ID,
        "conversation_id": conversation.conversation_id,
        "request_id": "integration-failed-duplicate-001",
        "task_type": "knowledge_question",
        "question": "How should clean plastic containers be recycled?",
        "options": RequestOptions(run_vision=False),
    }

    with pytest.raises(ApplicationServiceError):
        service.execute(**request)

    history_after_failure = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )

    with pytest.raises(PreviouslyFailedRequestError):
        service.execute(**request)

    history_after_duplicate = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        request["request_id"],
        actor_id=ACTOR_ID,
    )

    assert len(history_after_failure) == 1
    assert history_after_duplicate == history_after_failure
    assert execution.status == "FAILED"
    assert execution.response is None


def test_reused_request_id_with_different_input_is_rejected():
    import pytest

    from smart_waste_multimodal.api.application_service import (
        RequestFingerprintConflictError,
    )

    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    request = {
        "actor_id": ACTOR_ID,
        "conversation_id": conversation.conversation_id,
        "request_id": "integration-fingerprint-conflict-001",
        "task_type": "knowledge_question",
        "question": "How should clean plastic containers be recycled?",
        "options": RequestOptions(run_vision=False),
    }

    first_response = service.execute(**request)

    changed_request = {
        **request,
        "question": "How should glass bottles be recycled?",
    }

    with pytest.raises(RequestFingerprintConflictError):
        service.execute(**changed_request)

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        request["request_id"],
        actor_id=ACTOR_ID,
    )

    assert len(history) == 2
    assert execution.status == "COMPLETED"
    assert execution.response == first_response


def test_artifact_from_another_conversation_is_rejected():
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ArtifactOwnershipError,
    )

    (
        service,
        first_conversation,
        conversation_store,
        image_store,
        execution_store,
    ) = make_service()

    first_response = service.execute(
        actor_id=ACTOR_ID,
        conversation_id=first_conversation.conversation_id,
        request_id="integration-cross-conversation-upload-001",
        task_type="detection",
        question="Identify this waste item.",
        image_bytes=make_png(),
        image_media_type="image/png",
    )

    first_history = conversation_store.get_history(
        first_conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    artifact_id = first_history[0].image_artifact_id

    second_conversation = service.create_conversation(
        actor_id=ACTOR_ID,
    )
    second_request_id = "integration-cross-conversation-reuse-001"

    assert first_response.status == "success"
    assert artifact_id is not None
    assert image_store.contains(artifact_id)

    with pytest.raises(ArtifactOwnershipError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=second_conversation.conversation_id,
            request_id=second_request_id,
            task_type="detection",
            question="Analyze the previously uploaded image.",
            image_artifact_id=artifact_id,
        )

    second_history = conversation_store.get_history(
        second_conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        second_request_id,
        actor_id=ACTOR_ID,
    )

    assert second_history == ()
    assert image_store.contains(artifact_id)
    assert execution.status == "FAILED"
    assert execution.response is None


def test_deleted_artifact_with_existing_history_reference_is_rejected():
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ArtifactOwnershipError,
    )

    (
        service,
        conversation,
        conversation_store,
        image_store,
        execution_store,
    ) = make_service()

    service.execute(
        actor_id=ACTOR_ID,
        conversation_id=conversation.conversation_id,
        request_id="integration-deleted-artifact-upload-001",
        task_type="detection",
        question="Identify this waste item.",
        image_bytes=make_png(),
        image_media_type="image/png",
    )

    original_history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    artifact_id = original_history[0].image_artifact_id

    assert artifact_id is not None
    assert image_store.delete(artifact_id) is True
    assert not image_store.contains(artifact_id)

    request_id = "integration-deleted-artifact-reuse-001"

    with pytest.raises(ArtifactOwnershipError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="detection",
            question="Analyze the previously uploaded image.",
            image_artifact_id=artifact_id,
        )

    final_history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        request_id,
        actor_id=ACTOR_ID,
    )

    assert final_history == original_history
    assert not image_store.contains(artifact_id)
    assert execution.status == "FAILED"
    assert execution.response is None


def test_failure_to_persist_failed_state_requires_reconciliation(
    monkeypatch,
):
    import pytest

    from smart_waste_multimodal.api.application_service import (
        ApplicationServiceConsistencyError,
    )

    (
        service,
        conversation,
        conversation_store,
        _,
        execution_store,
    ) = make_service()

    def fail_run(request, *, conversation, actor_id=None):
        raise RuntimeError("Injected orchestrator failure.")

    def fail_execution_update(*args, **kwargs):
        raise RuntimeError("Injected execution failure-state persistence error.")

    monkeypatch.setattr(service._orchestrator, "run", fail_run)
    monkeypatch.setattr(
        execution_store,
        "fail",
        fail_execution_update,
    )

    request_id = "integration-failed-state-persistence-001"

    with pytest.raises(ApplicationServiceConsistencyError):
        service.execute(
            actor_id=ACTOR_ID,
            conversation_id=conversation.conversation_id,
            request_id=request_id,
            task_type="knowledge_question",
            question="How should clean plastic containers be recycled?",
            options=RequestOptions(run_vision=False),
        )

    history = conversation_store.get_history(
        conversation.conversation_id,
        actor_id=ACTOR_ID,
    )
    execution = execution_store.get(
        request_id,
        actor_id=ACTOR_ID,
    )

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].request_id == request_id
    assert execution.status == "PROCESSING"
    assert execution.response is None
