"""Protocol defining request execution storage operations."""

from typing import Protocol

from smart_waste_multimodal.contracts.common import ErrorInfo
from smart_waste_multimodal.contracts.response import AnalyzeResponse

from .contracts import RequestExecution, RequestReservation


class RequestExecutionStore(Protocol):
    """Storage interface for idempotent request execution tracking."""

    def reserve(
        self,
        *,
        actor_id: str,
        request_id: str,
        fingerprint: str,
        conversation_id: str | None = None,
    ) -> RequestReservation:
        """Atomically reserve a request ID or return its existing execution."""

    def get(
        self,
        request_id: str,
        *,
        actor_id: str,
    ) -> RequestExecution:
        """Retrieve an execution after verifying ownership."""

    def complete(
        self,
        request_id: str,
        *,
        actor_id: str,
        response: AnalyzeResponse,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> RequestExecution:
        """Complete a processing execution with its pipeline response."""

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
        """Mark a processing execution as failed."""
