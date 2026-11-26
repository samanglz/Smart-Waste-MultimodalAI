"""Build the evidence package authorized by Final Decision."""

from pydantic import ValidationError

from ..contracts.evidence import Evidence
from ..contracts.final_decision import FinalDecisionResult
from ..contracts.final_evidence_context import FinalEvidenceContext
from .interfaces import ConversationEvidenceRegistry


class FinalEvidenceContextBuildError(Exception):
    """Base exception for final evidence context construction."""


class EvidenceContextUnavailableError(FinalEvidenceContextBuildError):
    """Raised when the decision or registry cannot support construction."""


class EvidenceContextIntegrityError(FinalEvidenceContextBuildError):
    """Raised when retrieved evidence violates the authorized package."""


class FinalEvidenceContextBuilder:
    """Retrieve and package exactly the evidence authorized by Final Decision."""

    def build(
        self,
        *,
        request_id: str,
        conversation_id: str,
        final_decision: FinalDecisionResult,
        evidence_registry: ConversationEvidenceRegistry,
        actor_id: str | None = None,
    ) -> FinalEvidenceContext:
        if not isinstance(request_id, str) or not request_id.strip():
            raise EvidenceContextIntegrityError("request_id must not be blank.")

        if (
            final_decision.execution_status != "DECIDED"
            or final_decision.action != "DIRECT"
        ):
            raise EvidenceContextUnavailableError(
                "Final Evidence Context requires a DECIDED/DIRECT result."
            )

        evidence_ids = list(final_decision.evidence_ids)

        if not evidence_ids:
            raise EvidenceContextIntegrityError(
                "DIRECT decision contains no evidence IDs."
            )

        if any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
            raise EvidenceContextIntegrityError(
                "Decision contains a blank or invalid evidence ID."
            )

        if len(evidence_ids) != len(set(evidence_ids)):
            raise EvidenceContextIntegrityError(
                "Decision contains duplicate evidence IDs."
            )

        if evidence_registry.conversation_id != conversation_id:
            raise EvidenceContextIntegrityError(
                "Registry conversation_id does not match the current Conversation."
            )

        try:
            records = evidence_registry.get_many(
                evidence_ids,
                actor_id=actor_id or evidence_registry.owner_id,
            )
        except Exception as exc:
            raise EvidenceContextUnavailableError(
                "Evidence registry retrieval failed."
            ) from exc

        if not isinstance(records, list):
            raise EvidenceContextIntegrityError("Registry get_many must return a list.")

        if len(records) != len(evidence_ids):
            raise EvidenceContextIntegrityError(
                "One or more authorized evidence records are missing."
            )

        for expected_id, item in zip(evidence_ids, records):
            if not isinstance(item, Evidence):
                raise EvidenceContextIntegrityError(
                    "Registry returned an invalid evidence record."
                )

            if item.evidence_id != expected_id:
                raise EvidenceContextIntegrityError(
                    "Retrieved evidence order or identity does not match "
                    "the authorized evidence IDs."
                )

            if item.conversation_id != conversation_id:
                raise EvidenceContextIntegrityError(
                    "Retrieved evidence belongs to another Conversation."
                )

            if not item.provenance.source_id or not item.provenance.source_id.strip():
                raise EvidenceContextIntegrityError(
                    "Evidence provenance source_id must not be blank."
                )

        try:
            return FinalEvidenceContext(
                request_id=request_id,
                conversation_id=conversation_id,
                evidence_ids=tuple(evidence_ids),
                evidence=tuple(records),
            )
        except ValidationError as exc:
            raise EvidenceContextIntegrityError(
                "Retrieved evidence does not satisfy the context contract."
            ) from exc
