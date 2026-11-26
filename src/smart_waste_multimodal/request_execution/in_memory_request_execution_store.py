"""Thread-safe in-memory RequestExecutionStore implementation."""

from datetime import UTC, datetime
from threading import RLock

from smart_waste_multimodal.contracts.common import ErrorInfo
from smart_waste_multimodal.contracts.response import AnalyzeResponse

from .contracts import RequestExecution, RequestReservation
from .errors import (
    IdempotencyConflictError,
    InvalidRequestExecutionError,
    RequestExecutionNotFoundError,
    RequestExecutionStateError,
)


class InMemoryRequestExecutionStore:
    """Store request execution snapshots in process memory."""

    def __init__(self) -> None:
        self._executions: dict[tuple[str, str], RequestExecution] = {}
        self._lock = RLock()

    def reserve(
        self,
        *,
        actor_id: str,
        request_id: str,
        fingerprint: str,
        conversation_id: str | None = None,
    ) -> RequestReservation:
        self._validate_identity(actor_id, request_id)

        try:
            candidate = RequestExecution(
                actor_id=actor_id,
                request_id=request_id,
                conversation_id=conversation_id,
                fingerprint=fingerprint,
                status="PROCESSING",
                created_at=datetime.now(UTC),
            )
        except Exception as exc:
            raise InvalidRequestExecutionError(
                "Request execution reservation data is invalid."
            ) from exc

        key = (actor_id, request_id)

        with self._lock:
            existing = self._executions.get(key)

            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "The request ID was previously used with different input."
                    )
                return RequestReservation(
                    execution=existing.model_copy(deep=True),
                    is_new=False,
                )

            stored = candidate.model_copy(deep=True)
            self._executions[key] = stored

            return RequestReservation(
                execution=stored.model_copy(deep=True),
                is_new=True,
            )

    def get(
        self,
        request_id: str,
        *,
        actor_id: str,
    ) -> RequestExecution:
        self._validate_identity(actor_id, request_id)

        with self._lock:
            execution = self._executions.get((actor_id, request_id))

            if execution is None:
                raise RequestExecutionNotFoundError(
                    "Request execution was not found."
                )

            return execution.model_copy(deep=True)

    def complete(
        self,
        request_id: str,
        *,
        actor_id: str,
        response: AnalyzeResponse,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> RequestExecution:
        self._validate_identity(actor_id, request_id)

        if not isinstance(response, AnalyzeResponse):
            raise InvalidRequestExecutionError(
                "response must be an AnalyzeResponse instance."
            )

        with self._lock:
            current = self._get_processing(actor_id, request_id)

            if response.request_id != request_id:
                raise InvalidRequestExecutionError(
                    "response request_id must match the execution request_id."
                )

            now = max(datetime.now(UTC), current.created_at)
            resolved_conversation_id = self._resolve_conversation_id(
                current,
                conversation_id,
            )

            try:
                updated = RequestExecution.model_validate({
                    **current.model_dump(mode="python"),
                    "conversation_id": resolved_conversation_id,
                    "turn_id": turn_id if turn_id is not None else current.turn_id,
                    "status": "COMPLETED",
                    "response": response.model_copy(deep=True),
                    "safe_error": None,
                    "internal_error_type": None,
                    "completed_at": now,
                })
            except Exception as exc:
                raise InvalidRequestExecutionError(
                    "Completion data is invalid."
                ) from exc

            self._executions[(actor_id, request_id)] = updated
            return updated.model_copy(deep=True)

    def fail(
        self,
        request_id: str,
        *,
        actor_id: str,
        safe_error: ErrorInfo,
        internal_error_type: str,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> RequestExecution:
        self._validate_identity(actor_id, request_id)

        if not isinstance(safe_error, ErrorInfo):
            raise InvalidRequestExecutionError(
                "safe_error must be an ErrorInfo instance."
            )

        if (
            not isinstance(internal_error_type, str)
            or not internal_error_type
            or internal_error_type != internal_error_type.strip()
            or len(internal_error_type) > 200
        ):
            raise InvalidRequestExecutionError(
                "internal_error_type must be a valid non-empty string."
            )

        with self._lock:
            current = self._get_processing(actor_id, request_id)
            now = max(datetime.now(UTC), current.created_at)
            resolved_conversation_id = self._resolve_conversation_id(
                current,
                conversation_id,
            )

            try:
                updated = RequestExecution.model_validate({
                    **current.model_dump(mode="python"),
                    "conversation_id": resolved_conversation_id,
                    "turn_id": turn_id if turn_id is not None else current.turn_id,
                    "status": "FAILED",
                    "response": None,
                    "safe_error": safe_error.model_copy(deep=True),
                    "internal_error_type": internal_error_type,
                    "completed_at": now,
                })
            except Exception as exc:
                raise InvalidRequestExecutionError(
                    "Failure data is invalid."
                ) from exc

            self._executions[(actor_id, request_id)] = updated
            return updated.model_copy(deep=True)

    def _get_processing(
        self,
        actor_id: str,
        request_id: str,
    ) -> RequestExecution:
        current = self._executions.get((actor_id, request_id))

        if current is None:
            raise RequestExecutionNotFoundError(
                "Request execution was not found."
            )

        if current.status != "PROCESSING":
            raise RequestExecutionStateError(
                f"Cannot change an execution in state {current.status}."
            )

        return current

    @staticmethod
    def _resolve_conversation_id(
        current: RequestExecution,
        requested: str | None,
    ) -> str | None:
        if requested is None:
            return current.conversation_id

        if (
            current.conversation_id is not None
            and requested != current.conversation_id
        ):
            raise RequestExecutionStateError(
                "A request execution cannot change its conversation ID."
            )

        return requested

    @staticmethod
    def _validate_identity(actor_id: str, request_id: str) -> None:
        for label, value in (
            ("actor_id", actor_id),
            ("request_id", request_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or len(value) > 200
            ):
                raise InvalidRequestExecutionError(
                    f"A valid {label} is required."
                )
