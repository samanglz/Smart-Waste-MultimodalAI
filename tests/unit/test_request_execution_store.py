"""Unit tests for InMemoryRequestExecutionStore."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from smart_waste_multimodal.contracts.common import ErrorInfo
from smart_waste_multimodal.contracts.response import AnalyzeResponse
from smart_waste_multimodal.request_execution.contracts import (
    RequestExecution,
    RequestExecutionStatus,
)
from smart_waste_multimodal.request_execution.errors import (
    IdempotencyConflictError,
    InvalidRequestExecutionError,
    RequestExecutionNotFoundError,
    RequestExecutionStateError,
)
from smart_waste_multimodal.request_execution.in_memory_request_execution_store import (
    InMemoryRequestExecutionStore,
)


FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def make_response(request_id="request-001"):
    """Build a valid pipeline response without useful stage outputs."""
    return AnalyzeResponse(
        request_id=request_id,
        status="failed",
        errors=[
            ErrorInfo(
                code="TEST_PIPELINE_ERROR",
                stage="orchestration",
                message="Synthetic pipeline failure.",
                retryable=False,
            )
        ],
    )


def make_safe_error():
    return ErrorInfo(
        code="TEST_EXECUTION_ERROR",
        stage="orchestration",
        message="The request could not be completed.",
        retryable=False,
    )


def reserve(store, *, actor_id="actor-001",
             request_id="request-001",
             fingerprint=FINGERPRINT_A,
             conversation_id=None):
    return store.reserve(
        actor_id=actor_id,
        request_id=request_id,
        fingerprint=fingerprint,
        conversation_id=conversation_id,
    )


def test_reserve_creates_processing_execution():
    store = InMemoryRequestExecutionStore()

    reservation = reserve(store, conversation_id="conversation-001")

    assert reservation.is_new is True
    assert reservation.execution.actor_id == "actor-001"
    assert reservation.execution.request_id == "request-001"
    assert reservation.execution.conversation_id == "conversation-001"
    assert reservation.execution.fingerprint == FINGERPRINT_A
    assert reservation.execution.status == "PROCESSING"
    assert reservation.execution.response is None
    assert reservation.execution.safe_error is None
    assert reservation.execution.completed_at is None


def test_reserve_same_key_and_fingerprint_returns_existing_execution():
    store = InMemoryRequestExecutionStore()

    first = reserve(store, conversation_id="conversation-001")
    second = reserve(store)

    assert first.is_new is True
    assert second.is_new is False
    assert second.execution == first.execution
    assert second.execution.conversation_id == "conversation-001"


def test_reserve_rejects_same_key_with_different_fingerprint():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    with pytest.raises(IdempotencyConflictError):
        reserve(store, fingerprint=FINGERPRINT_B)

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.fingerprint == FINGERPRINT_A
    assert stored.status == "PROCESSING"


def test_same_request_id_is_isolated_between_actors():
    store = InMemoryRequestExecutionStore()

    first = reserve(store, actor_id="actor-001")
    second = reserve(
        store,
        actor_id="actor-002",
        fingerprint=FINGERPRINT_B,
    )

    assert first.is_new is True
    assert second.is_new is True
    assert store.get(
        "request-001", actor_id="actor-001"
    ).fingerprint == FINGERPRINT_A
    assert store.get(
        "request-001", actor_id="actor-002"
    ).fingerprint == FINGERPRINT_B


def test_get_does_not_expose_another_actors_execution():
    store = InMemoryRequestExecutionStore()
    reserve(store, actor_id="actor-001")

    with pytest.raises(RequestExecutionNotFoundError):
        store.get("request-001", actor_id="actor-002")


def test_get_unknown_request_raises_not_found():
    store = InMemoryRequestExecutionStore()

    with pytest.raises(RequestExecutionNotFoundError):
        store.get("missing-request", actor_id="actor-001")


def test_complete_transitions_processing_to_completed():
    store = InMemoryRequestExecutionStore()
    reserve(store, conversation_id="conversation-001")

    response = make_response()
    completed = store.complete(
        "request-001",
        actor_id="actor-001",
        response=response,
        turn_id="turn-001",
    )

    assert completed.status == "COMPLETED"
    assert completed.response == response
    assert completed.safe_error is None
    assert completed.internal_error_type is None
    assert completed.turn_id == "turn-001"
    assert completed.conversation_id == "conversation-001"
    assert completed.completed_at is not None
    assert completed.completed_at >= completed.created_at

    stored = store.get("request-001", actor_id="actor-001")
    assert stored == completed


def test_complete_can_set_conversation_id_when_reservation_had_none():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    completed = store.complete(
        "request-001",
        actor_id="actor-001",
        response=make_response(),
        conversation_id="conversation-created-later",
    )

    assert completed.conversation_id == "conversation-created-later"


def test_complete_accepts_the_same_existing_conversation_id():
    store = InMemoryRequestExecutionStore()
    reserve(store, conversation_id="conversation-001")

    completed = store.complete(
        "request-001",
        actor_id="actor-001",
        response=make_response(),
        conversation_id="conversation-001",
    )

    assert completed.conversation_id == "conversation-001"


def test_complete_rejects_changing_conversation_id_without_wrapping_state_error():
    store = InMemoryRequestExecutionStore()
    reserve(store, conversation_id="conversation-001")

    with pytest.raises(RequestExecutionStateError):
        store.complete(
            "request-001",
            actor_id="actor-001",
            response=make_response(),
            conversation_id="conversation-002",
        )

    assert store.get(
        "request-001", actor_id="actor-001"
    ).status == "PROCESSING"


def test_fail_transitions_processing_to_failed():
    store = InMemoryRequestExecutionStore()
    reserve(store, conversation_id="conversation-001")

    failed = store.fail(
        "request-001",
        actor_id="actor-001",
        safe_error=make_safe_error(),
        internal_error_type="RuntimeError",
        turn_id="turn-001",
    )

    assert failed.status == "FAILED"
    assert failed.response is None
    assert failed.safe_error == make_safe_error()
    assert failed.internal_error_type == "RuntimeError"
    assert failed.turn_id == "turn-001"
    assert failed.conversation_id == "conversation-001"
    assert failed.completed_at is not None
    assert failed.completed_at >= failed.created_at
    assert store.get("request-001", actor_id="actor-001") == failed


def test_fail_can_set_conversation_id_when_reservation_had_none():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    failed = store.fail(
        "request-001",
        actor_id="actor-001",
        safe_error=make_safe_error(),
        internal_error_type="RuntimeError",
        conversation_id="conversation-created-later",
    )

    assert failed.conversation_id == "conversation-created-later"


def test_fail_rejects_changing_conversation_id_without_wrapping_state_error():
    store = InMemoryRequestExecutionStore()
    reserve(store, conversation_id="conversation-001")

    with pytest.raises(RequestExecutionStateError):
        store.fail(
            "request-001",
            actor_id="actor-001",
            safe_error=make_safe_error(),
            internal_error_type="RuntimeError",
            conversation_id="conversation-002",
        )

    assert store.get(
        "request-001", actor_id="actor-001"
    ).status == "PROCESSING"


def test_completed_execution_cannot_be_completed_again():
    store = InMemoryRequestExecutionStore()
    reserve(store)
    store.complete(
        "request-001",
        actor_id="actor-001",
        response=make_response(),
    )

    with pytest.raises(RequestExecutionStateError):
        store.complete(
            "request-001",
            actor_id="actor-001",
            response=make_response(),
        )


def test_completed_execution_cannot_be_failed():
    store = InMemoryRequestExecutionStore()
    reserve(store)
    store.complete(
        "request-001",
        actor_id="actor-001",
        response=make_response(),
    )

    with pytest.raises(RequestExecutionStateError):
        store.fail(
            "request-001",
            actor_id="actor-001",
            safe_error=make_safe_error(),
            internal_error_type="RuntimeError",
        )


def test_failed_execution_cannot_be_completed_or_failed_again():
    store = InMemoryRequestExecutionStore()
    reserve(store)
    store.fail(
        "request-001",
        actor_id="actor-001",
        safe_error=make_safe_error(),
        internal_error_type="RuntimeError",
    )

    with pytest.raises(RequestExecutionStateError):
        store.complete(
            "request-001",
            actor_id="actor-001",
            response=make_response(),
        )

    with pytest.raises(RequestExecutionStateError):
        store.fail(
            "request-001",
            actor_id="actor-001",
            safe_error=make_safe_error(),
            internal_error_type="RuntimeError",
        )


def test_duplicate_reservation_returns_completed_result():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    completed = store.complete(
        "request-001",
        actor_id="actor-001",
        response=make_response(),
    )

    repeated = reserve(store)

    assert repeated.is_new is False
    assert repeated.execution == completed
    assert repeated.execution.status == "COMPLETED"


def test_duplicate_reservation_returns_failed_execution():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    failed = store.fail(
        "request-001",
        actor_id="actor-001",
        safe_error=make_safe_error(),
        internal_error_type="RuntimeError",
    )

    repeated = reserve(store)

    assert repeated.is_new is False
    assert repeated.execution == failed
    assert repeated.execution.status == "FAILED"


def test_reservation_snapshot_is_independent_from_internal_state():
    store = InMemoryRequestExecutionStore()
    reservation = reserve(store)

    reservation.execution.conversation_id = "external-mutation"
    reservation.execution.turn_id = "external-turn"

    stored = store.get("request-001", actor_id="actor-001")

    assert stored.conversation_id is None
    assert stored.turn_id is None


def test_get_snapshot_is_independent_from_internal_state():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    snapshot = store.get("request-001", actor_id="actor-001")
    snapshot.turn_id = "external-turn"

    assert store.get(
        "request-001", actor_id="actor-001"
    ).turn_id is None


def test_complete_stores_an_independent_response_snapshot():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    response = make_response()
    completed = store.complete(
        "request-001",
        actor_id="actor-001",
        response=response,
    )

    response.warnings.append("external mutation")

    stored = store.get("request-001", actor_id="actor-001")
    assert "external mutation" not in stored.response.warnings
    assert completed.response.warnings == []


def test_complete_return_value_is_independent_from_stored_snapshot():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    completed = store.complete(
        "request-001",
        actor_id="actor-001",
        response=make_response(),
    )
    completed.response.warnings.append("external mutation")

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.response.warnings == []


def test_fail_stores_an_independent_error_snapshot():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    error = make_safe_error()
    store.fail(
        "request-001",
        actor_id="actor-001",
        safe_error=error,
        internal_error_type="RuntimeError",
    )

    error.message = "mutated externally"

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.safe_error.message == "The request could not be completed."


@pytest.mark.parametrize(
    "actor_id,request_id",
    [
        ("", "request-001"),
        (" actor-001", "request-001"),
        ("actor-001 ", "request-001"),
        ("actor-001", ""),
        ("actor-001", " request-001"),
        ("actor-001", "request-001 "),
        ("a" * 201, "request-001"),
        ("actor-001", "r" * 201),
        (None, "request-001"),
        ("actor-001", None),
        (123, "request-001"),
        ("actor-001", 123),
    ],
)
def test_reserve_rejects_invalid_identity(actor_id, request_id):
    store = InMemoryRequestExecutionStore()

    with pytest.raises(InvalidRequestExecutionError):
        store.reserve(
            actor_id=actor_id,
            request_id=request_id,
            fingerprint=FINGERPRINT_A,
        )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "g" * 64,
        None,
    ],
)
def test_reserve_rejects_invalid_fingerprint(fingerprint):
    store = InMemoryRequestExecutionStore()

    with pytest.raises(InvalidRequestExecutionError):
        reserve(store, fingerprint=fingerprint)


def test_complete_rejects_non_analyze_response():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    with pytest.raises(InvalidRequestExecutionError):
        store.complete(
            "request-001",
            actor_id="actor-001",
            response={"request_id": "request-001"},
        )


def test_fail_rejects_non_error_info():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    with pytest.raises(InvalidRequestExecutionError):
        store.fail(
            "request-001",
            actor_id="actor-001",
            safe_error={"code": "ERROR"},
            internal_error_type="RuntimeError",
        )


@pytest.mark.parametrize(
    "internal_error_type",
    [
        "",
        " ",
        " RuntimeError",
        "RuntimeError ",
        "x" * 201,
        None,
        123,
    ],
)
def test_fail_rejects_invalid_internal_error_type(internal_error_type):
    store = InMemoryRequestExecutionStore()
    reserve(store)

    with pytest.raises(InvalidRequestExecutionError):
        store.fail(
            "request-001",
            actor_id="actor-001",
            safe_error=make_safe_error(),
            internal_error_type=internal_error_type,
        )


def test_complete_unknown_request_raises_not_found():
    store = InMemoryRequestExecutionStore()

    with pytest.raises(RequestExecutionNotFoundError):
        store.complete(
            "missing-request",
            actor_id="actor-001",
            response=make_response(),
        )


def test_fail_unknown_request_raises_not_found():
    store = InMemoryRequestExecutionStore()

    with pytest.raises(RequestExecutionNotFoundError):
        store.fail(
            "missing-request",
            actor_id="actor-001",
            safe_error=make_safe_error(),
            internal_error_type="RuntimeError",
        )


def test_concurrent_reservations_accept_exactly_one_new_execution():
    store = InMemoryRequestExecutionStore()
    worker_count = 16
    barrier = Barrier(worker_count)

    def worker():
        barrier.wait()
        return reserve(store)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _: worker(), range(worker_count)))

    assert sum(result.is_new for result in results) == 1
    assert all(
        result.execution.status == "PROCESSING"
        for result in results
    )
    assert len({
        result.execution.created_at
        for result in results
    }) >= 1

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.status == "PROCESSING"
    assert stored.fingerprint == FINGERPRINT_A


def test_concurrent_reservations_with_conflicting_fingerprints_are_serialized():
    store = InMemoryRequestExecutionStore()
    barrier = Barrier(2)

    def worker(fingerprint):
        barrier.wait()
        try:
            result = reserve(store, fingerprint=fingerprint)
            return ("reserved", result.is_new)
        except IdempotencyConflictError:
            return ("conflict", False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            worker,
            [FINGERPRINT_A, FINGERPRINT_B],
        ))

    assert sum(kind == "reserved" for kind, _ in results) == 1
    assert sum(kind == "conflict" for kind, _ in results) == 1

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.fingerprint in {FINGERPRINT_A, FINGERPRINT_B}
    assert stored.status == "PROCESSING"


def test_request_execution_contract_rejects_invalid_fingerprint():
    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint="invalid",
            created_at=datetime.now(UTC),
        )


def test_request_execution_contract_rejects_naive_created_at():
    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            created_at=datetime.now(),
        )


def test_request_execution_contract_rejects_completion_before_creation():
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    completed_at = datetime(2025, 12, 31, tzinfo=UTC)

    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            status="COMPLETED",
            response=make_response(),
            created_at=created_at,
            completed_at=completed_at,
        )


def test_request_execution_contract_rejects_processing_with_completion_data():
    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            status="PROCESSING",
            response=make_response(),
            created_at=datetime.now(UTC),
        )


def test_request_execution_contract_rejects_completed_without_response():
    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            status="COMPLETED",
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


def test_request_execution_contract_rejects_failed_without_error_details():
    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            status="FAILED",
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

# ---------------------------------------------------------------------------
# Additional boundary and concurrency tests
# ---------------------------------------------------------------------------

def test_complete_rejects_response_with_different_request_id():
    store = InMemoryRequestExecutionStore()
    reserve(store)

    with pytest.raises(InvalidRequestExecutionError):
        store.complete(
            "request-001",
            actor_id="actor-001",
            response=make_response(request_id="request-002"),
        )

    assert store.get(
        "request-001", actor_id="actor-001"
    ).status == "PROCESSING"


def test_request_execution_contract_rejects_response_request_id_mismatch():
    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            status="COMPLETED",
            response=make_response(request_id="request-002"),
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


def test_complete_and_fail_race_allows_exactly_one_terminal_transition():
    store = InMemoryRequestExecutionStore()
    reserve(store)
    barrier = Barrier(2)

    def complete_worker():
        barrier.wait()
        try:
            result = store.complete(
                "request-001",
                actor_id="actor-001",
                response=make_response(),
            )
            return ("completed", result.status)
        except RequestExecutionStateError:
            return ("state_error", None)

    def fail_worker():
        barrier.wait()
        try:
            result = store.fail(
                "request-001",
                actor_id="actor-001",
                safe_error=make_safe_error(),
                internal_error_type="RuntimeError",
            )
            return ("failed", result.status)
        except RequestExecutionStateError:
            return ("state_error", None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(complete_worker),
            executor.submit(fail_worker),
        ]
        results = [future.result() for future in futures]

    successful = [
        result for result in results
        if result[0] in {"completed", "failed"}
    ]
    rejected = [
        result for result in results
        if result[0] == "state_error"
    ]

    assert len(successful) == 1
    assert len(rejected) == 1

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.status in {"COMPLETED", "FAILED"}
    assert stored.status == successful[0][1]


@pytest.mark.parametrize(
    ("method_name", "field_name", "invalid_value"),
    [
        ("complete", "conversation_id", ""),
        ("complete", "conversation_id", "x" * 201),
        ("complete", "turn_id", ""),
        ("complete", "turn_id", "x" * 201),
        ("fail", "conversation_id", ""),
        ("fail", "conversation_id", "x" * 201),
        ("fail", "turn_id", ""),
        ("fail", "turn_id", "x" * 201),
    ],
)
def test_terminal_transition_rejects_invalid_optional_identifiers(
    method_name,
    field_name,
    invalid_value,
):
    store = InMemoryRequestExecutionStore()
    reserve(store)

    kwargs = {
        "actor_id": "actor-001",
        field_name: invalid_value,
    }

    if method_name == "complete":
        kwargs["response"] = make_response()
    else:
        kwargs["safe_error"] = make_safe_error()
        kwargs["internal_error_type"] = "RuntimeError"

    with pytest.raises(InvalidRequestExecutionError):
        getattr(store, method_name)("request-001", **kwargs)

    stored = store.get("request-001", actor_id="actor-001")
    assert stored.status == "PROCESSING"


def test_request_execution_contract_rejects_naive_completed_at():
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError):
        RequestExecution(
            actor_id="actor-001",
            request_id="request-001",
            fingerprint=FINGERPRINT_A,
            status="COMPLETED",
            response=make_response(),
            created_at=created_at,
            completed_at=datetime(2026, 1, 1, 1, 0),
        )
