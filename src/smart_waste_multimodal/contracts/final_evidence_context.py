from pydantic import BaseModel, ConfigDict, model_validator

from smart_waste_multimodal.contracts.evidence import Evidence


class FinalEvidenceContext(BaseModel):
    """Immutable package of evidence authorized by Final Decision."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    conversation_id: str
    evidence_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]

    @model_validator(mode="after")
    def validate_evidence_alignment(self):
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be blank")

        if not self.evidence:
            raise ValueError("FinalEvidenceContext must contain evidence")

        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids must not contain blank values")

        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")

        if len(self.evidence_ids) != len(self.evidence):
            raise ValueError("evidence_ids and evidence must have equal lengths")

        for evidence_id, item in zip(self.evidence_ids, self.evidence):
            if item.evidence_id != evidence_id:
                raise ValueError("Evidence order must match evidence_ids order")

            if item.conversation_id != self.conversation_id:
                raise ValueError("All evidence must belong to the same Conversation")

        return self
