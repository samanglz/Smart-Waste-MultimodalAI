"""Application-level orchestration for conversation analysis requests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    Conversation,
    ConversationTurn,
    ImageArtifact,
    RequestOptions,
)
from smart_waste_multimodal.contracts.common import ErrorInfo
from smart_waste_multimodal.conversation.conversation_store import (
    ConversationStore,
)
from smart_waste_multimodal.evidence.interfaces import (
    ConversationEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)
from smart_waste_multimodal.request_execution.errors import (
    IdempotencyConflictError,
)
from smart_waste_multimodal.request_execution.request_execution_store import (
    RequestExecutionStore,
)
from smart_waste_multimodal.vision.image_store import (
    ImageStore,
    SupportedImageMediaType,
)
from smart_waste_multimodal.vision.image_store_errors import (
    ImageNotFoundError,
)


class ApplicationServiceError(RuntimeError):
    """Base exception for application-service failures."""


class RequestAlreadyProcessingError(ApplicationServiceError):
    """Raised when the same request is already being processed."""


class PreviouslyFailedRequestError(ApplicationServiceError):
    """Raised when a request ID refers to a previously failed execution."""


class RequestFingerprintConflictError(ApplicationServiceError):
    """Raised when a request ID is reused with different input."""


class ArtifactOwnershipError(ApplicationServiceError):
    """Raised when an image artifact is not owned by the conversation."""


class ApplicationServiceConsistencyError(ApplicationServiceError):
    """Raised when application-level persistence becomes inconsistent."""


class AnalyzeApplicationService:
    """Coordinate conversation, image, execution, and pipeline services.

    actor_id must come from a trusted backend identity context.
    It must never be copied directly from an untrusted request payload.

    New image uploads receive newly generated artifact IDs. An artifact ID
    is a permanent identity and must never be reused for different content,
    even after the corresponding image has been deleted.
    """

    def __init__(
        self,
        *,
        conversation_store: ConversationStore,
        image_store: ImageStore,
        request_execution_store: RequestExecutionStore,
        evidence_registry_factory: ConversationEvidenceRegistryFactory,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        dependencies = {
            "conversation_store": conversation_store,
            "image_store": image_store,
            "request_execution_store": request_execution_store,
            "evidence_registry_factory": evidence_registry_factory,
            "orchestrator": orchestrator,
        }

        for name, dependency in dependencies.items():
            if dependency is None:
                raise TypeError(f"{name} must not be None.")

        self._conversation_store = conversation_store
        self._image_store = image_store
        self._request_execution_store = request_execution_store
        self._evidence_registry_factory = evidence_registry_factory
        self._orchestrator = orchestrator


    def create_conversation(self, *, actor_id: str) -> Conversation:
        """Create a conversation for the authenticated actor."""

        self._validate_actor_id(actor_id)
        return self._conversation_store.create(actor_id=actor_id)

    def execute(
        self,
        *,
        actor_id: str,
        conversation_id: str,
        request_id: str,
        task_type: Literal["detection", "knowledge_question"],
        question: str | None = None,
        locale: Literal["en", "fa"] = "en",
        options: RequestOptions | None = None,
        image_bytes: bytes | bytearray | memoryview | None = None,
        image_media_type: SupportedImageMediaType | None = None,
        image_artifact_id: str | None = None,
    ) -> AnalyzeResponse:
        """Execute one analysis request and persist its conversation turns.

        A new image upload and an existing artifact reference are mutually
        exclusive. Existing artifact ownership is verified against the
        authorized conversation history after request reservation.
        """

        self._validate_actor_id(actor_id)
        self._validate_request_id(request_id)

        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id must not be empty.")

        if task_type not in ("detection", "knowledge_question"):
            raise ValueError("Unsupported task_type.")

        if image_bytes is not None and image_artifact_id is not None:
            raise ValueError(
                "Provide image_bytes or image_artifact_id, not both."
            )

        if image_bytes is None and image_media_type is not None:
            raise ValueError(
                "image_media_type requires image_bytes."
            )

        if image_bytes is not None and image_media_type is None:
            raise ValueError(
                "image_media_type is required when image_bytes is provided."
            )

        if image_artifact_id is not None and (
            not isinstance(image_artifact_id, str)
            or not image_artifact_id.strip()
        ):
            raise ValueError("image_artifact_id must not be empty.")

        request_options = options or RequestOptions()

        # Conversation validation is intentionally deferred until after
        # idempotency reservation and duplicate-request resolution.

        # Build the idempotency identity without resolving an existing
        # artifact. Duplicate requests must be handled before accessing
        # Conversation or ImageStore state.
        uploaded_bytes = None
        image_digest = None

        if image_bytes is not None:
            uploaded_bytes = bytes(image_bytes)
            image_digest = hashlib.sha256(uploaded_bytes).hexdigest()

        fingerprint = self._fingerprint(
            conversation_id=conversation_id,
            task_type=task_type,
            question=question,
            locale=locale,
            options=request_options,
            image_digest=image_digest,
            image_media_type=image_media_type,
            image_artifact_id=image_artifact_id,
        )

        try:
            reservation = self._request_execution_store.reserve(
                actor_id=actor_id,
                request_id=request_id,
                fingerprint=fingerprint,
                conversation_id=conversation_id,
            )
        except IdempotencyConflictError as exc:
            raise RequestFingerprintConflictError(
                "This request_id has already been used with different input."
            ) from exc

        execution = reservation.execution

        if not reservation.is_new:
            if execution.fingerprint != fingerprint:
                raise RequestFingerprintConflictError(
                    "This request_id has already been used with different input."
                )

            if execution.status == "COMPLETED":
                if execution.response is None:
                    raise ApplicationServiceConsistencyError(
                        "Completed execution has no stored response."
                    )
                return execution.response

            if execution.status == "PROCESSING":
                raise RequestAlreadyProcessingError(
                    "This request is already being processed."
                )

            if execution.status == "FAILED":
                raise PreviouslyFailedRequestError(
                    execution.safe_error.message
                    if execution.safe_error is not None
                    else "This request previously failed."
                )

            raise ApplicationServiceConsistencyError(
                "Request execution has an unsupported status."
            )

        turn_id = str(uuid4())
        now = datetime.now(UTC)

        registered_artifact = None
        user_turn_appended = False
        assistant_turn_persisted = False

        try:
            # Only a newly reserved execution reaches this point.
            # Validate Conversation ownership and active state before
            # registering an image or appending a User Turn.
            conversation = self._conversation_store.get(
                conversation_id,
                actor_id=actor_id,
            )
            previous_history = self._conversation_store.get_history(
                conversation_id,
                actor_id=actor_id,
            )

            if image_artifact_id is not None:
                if not self._history_references_artifact(
                    history=previous_history,
                    conversation_id=conversation_id,
                    artifact_id=image_artifact_id,
                ):
                    raise ArtifactOwnershipError(
                        "The image artifact does not belong to this conversation."
                    )

                try:
                    existing_artifact = self._image_store.get_metadata(
                        image_artifact_id
                    )
                except ImageNotFoundError as exc:
                    raise ArtifactOwnershipError(
                        "The image artifact is no longer available."
                    ) from exc

            if uploaded_bytes is not None:
                artifact_id = str(uuid4())
                registered_artifact = self._image_store.register(
                    artifact_id,
                    uploaded_bytes,
                    image_media_type,
                )

            elif image_artifact_id is not None:
                registered_artifact = existing_artifact

            user_content = (
                question.strip()
                if isinstance(question, str) and question.strip()
                else (
                    "Image submitted for analysis."
                    if registered_artifact is not None
                    else "Analysis request submitted."
                )
            )

            user_turn = ConversationTurn(
                turn_id=turn_id,
                conversation_id=conversation_id,
                request_id=request_id,
                role="user",
                content=user_content,
                timestamp=now,
                image_artifact_id=(
                    registered_artifact.artifact_id
                    if registered_artifact is not None
                    else None
                ),
            )

            self._conversation_store.append_turn(
                conversation_id,
                user_turn,
                actor_id=actor_id,
            )
            user_turn_appended = True

            analyze_request = AnalyzeRequest(
                request_id=request_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                turn_timestamp=now,
                task_type=task_type,
                image=registered_artifact,
                conversation_history=list(previous_history),
                question=question,
                locale=locale,
                options=request_options,
            )

            response = self._orchestrator.run(
                analyze_request,
                conversation=conversation,
                actor_id=actor_id,
            )

            if not isinstance(response, AnalyzeResponse):
                raise TypeError(
                    "PipelineOrchestrator.run must return AnalyzeResponse."
                )

            # Retrieve every piece of evidence generated for this turn,
            # not merely the subset selected for the final answer.
            registry = self._evidence_registry_factory.get(
                conversation_id,
                actor_id=actor_id,
            )

            turn_evidence = registry.list_by_turn(
                turn_id,
                actor_id=actor_id,
            )

            evidence_ids = tuple(
                item.evidence_id for item in turn_evidence
            )

            # Update the existing user turn before creating an assistant turn.
            self._conversation_store.update_turn(
                conversation_id,
                turn_id,
                actor_id=actor_id,
                vision=response.vision,
                evidence_ids=evidence_ids,
            )

            if response.status in (
                "success",
                "partial_success",
                "needs_clarification",
            ):
                assistant_content = self._assistant_content(response)

                assistant_turn = ConversationTurn(
                    turn_id=str(uuid4()),
                    conversation_id=conversation_id,
                    request_id=request_id,
                    role="assistant",
                    content=assistant_content,
                    timestamp=datetime.now(UTC),
                    evidence_ids=self._assistant_evidence_ids(response),
                )

                self._conversation_store.append_turn(
                    conversation_id,
                    assistant_turn,
                    actor_id=actor_id,
                )
                assistant_turn_persisted = True

            # A valid AnalyzeResponse with status="failed" is still a
            # completed pipeline execution, not an infrastructure failure.
            self._request_execution_store.complete(
                request_id,
                actor_id=actor_id,
                response=response,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )

            return response

        except Exception as exc:
            # Once the Assistant Turn is persisted, do not mark the execution
            # FAILED blindly: the conversation already contains a response.
            if assistant_turn_persisted:
                raise ApplicationServiceConsistencyError(
                    "The Assistant Turn was persisted, but RequestExecution "
                    "finalization failed. Reconciliation is required."
                ) from exc

            # Preserve the user turn, image, and evidence after pipeline or
            # persistence failures. Do not fabricate an assistant response.
            try:
                self._request_execution_store.fail(
                    request_id,
                    actor_id=actor_id,
                    safe_error=self._make_safe_error(exc),
                    internal_error_type=type(exc).__name__,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
            except Exception as persistence_exc:
                raise ApplicationServiceConsistencyError(
                    "The request failed and its execution status could not "
                    "be persisted."
                ) from persistence_exc

            # If appending the user turn itself failed, the newly registered
            # image is unreferenced and can be compensated safely.
            if not user_turn_appended and registered_artifact is not None:
                if uploaded_bytes is not None:
                    try:
                        self._image_store.delete(
                            registered_artifact.artifact_id
                        )
                    except Exception:
                        # Do not hide the original request failure.
                        pass

            if isinstance(exc, ArtifactOwnershipError):
                raise

            raise ApplicationServiceError(
                "The analysis request could not be completed."
            ) from exc

    @staticmethod
    def _validate_actor_id(actor_id: str) -> None:
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("A trusted actor_id is required.")

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must not be empty.")

        if request_id != request_id.strip():
            raise ValueError(
                "request_id must not have leading or trailing whitespace."
            )

    @staticmethod
    def _history_references_artifact(
        *,
        history: tuple[ConversationTurn, ...],
        conversation_id: str,
        artifact_id: str,
    ) -> bool:
        """Check whether a user turn in this conversation references an artifact."""
        return any(
            turn.role == "user"
            and turn.conversation_id == conversation_id
            and turn.image_artifact_id == artifact_id
            for turn in history
        )

    @staticmethod
    def _fingerprint(
        *,
        conversation_id: str,
        task_type: str,
        question: str | None,
        locale: str,
        options: RequestOptions,
        image_digest: str | None,
        image_media_type: str | None,
        image_artifact_id: str | None,
    ) -> str:
        payload = {
            "conversation_id": conversation_id,
            "task_type": task_type,
            "question": question,
            "locale": locale,
            "options": options.model_dump(mode="json"),
            "image_sha256": image_digest,
            "image_media_type": image_media_type,
            "image_artifact_id": image_artifact_id,
        }

        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _assistant_evidence_ids(response: AnalyzeResponse) -> tuple[str, ...]:
        """Return only unambiguously cited, authorized retrieval evidence IDs."""
        generation = response.generation
        final_context = response.final_evidence_context

        if (
            generation is None
            or generation.status != "success"
            or final_context is None
            or not generation.citation_chunk_ids
        ):
            return ()

        cited_chunks = set(generation.citation_chunk_ids)
        matches = {}

        for evidence in final_context.evidence:
            source_type = getattr(
                evidence.source_type, "value", evidence.source_type
            )

            if source_type != "retrieval_chunk":
                continue

            content = evidence.content
            if not isinstance(content, dict):
                continue

            chunk_id = content.get("chunk_id")
            if (
                not isinstance(chunk_id, str)
                or chunk_id != evidence.provenance.source_id
                or chunk_id not in cited_chunks
            ):
                continue

            matches.setdefault(chunk_id, []).append(evidence.evidence_id)

        result = []
        for chunk_id in generation.citation_chunk_ids:
            evidence_ids = matches.get(chunk_id, [])

            if len(evidence_ids) == 1:
                evidence_id = evidence_ids[0]
                if evidence_id not in result:
                    result.append(evidence_id)

        return tuple(result)

    @staticmethod
    def _assistant_content(response: AnalyzeResponse) -> str:
        if response.status == "needs_clarification":
            message = response.clarification_message
            if message and message.strip():
                return message.strip()

            return "Additional clarification is required."

        generation = response.generation
        answer = None

        if generation is not None:
            if (
                generation.status in ("success", "insufficient_evidence")
                and generation.answer
                and generation.answer.strip()
            ):
                answer = generation.answer.strip()

        if response.status != "partial_success":
            return (
                answer
                or "The request was processed, but no answer text was generated."
            )

        details = []

        for error in response.errors:
            details.append(f"{error.stage}: {error.message}")

        for warning in response.warnings:
            message = warning.strip()
            if message and message not in details:
                details.append(message)

        if answer is None:
            answer = "The request was partially processed; no answer was generated."

        if not details:
            return answer

        report = "\n".join(f"- {item}" for item in details)
        return f"{answer}\n\nExecution details:\n{report}"

    @staticmethod
    def _make_safe_error(exc: Exception) -> ErrorInfo:
        """Create a sanitized ErrorInfo without exposing internal details."""

        candidates = {
            "code": "APPLICATION_EXECUTION_FAILED",
            "stage": "input",
            "message": "The request could not be completed.",
            "retryable": False,
            "error_code": "APPLICATION_EXECUTION_FAILED",
            "description": "The request could not be completed.",
        }

        fields = getattr(ErrorInfo, "model_fields", {})
        payload = {
            name: candidates[name]
            for name in fields
            if name in candidates
        }

        return ErrorInfo.model_validate(payload)
