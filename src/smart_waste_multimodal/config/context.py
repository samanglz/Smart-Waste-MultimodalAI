"""Configuration for conversation-context handling."""

from pydantic import Field

from smart_waste_multimodal.contracts.common import ContractModel


class ContextConfig(ContractModel):
    """Settings for deterministic conversation-context handling."""

    max_age_seconds: int = Field(default=1800, gt=0)
    max_history_turns: int = Field(default=20, gt=0)
    max_recovery_attempts: int = Field(default=2, ge=1, le=5)
