"""Evidence registration abstractions and implementations."""

from .final_evidence_context_builder import (
    EvidenceContextIntegrityError,
    EvidenceContextUnavailableError,
    FinalEvidenceContextBuilder,
    FinalEvidenceContextBuildError,
)
from .in_memory_registry import (
    ConversationScopedInMemoryEvidenceRegistry,
    ConversationScopedInMemoryEvidenceRegistryFactory,
    InMemoryEvidenceRegistry,
    InMemoryEvidenceRegistryFactory,
)
from .interfaces import (
    ConversationClosedError,
    ConversationEvidenceRegistry,
    ConversationEvidenceRegistryFactory,
    ConversationNotFoundError,
    ConversationOwnershipError,
    DuplicateEvidenceError,
    EvidenceConversationMismatchError,
    EvidenceRegistry,
    EvidenceRegistryError,
    EvidenceRegistryFactory,
    EvidenceRequestMismatchError,
)

__all__ = [
    "ConversationClosedError",
    "ConversationEvidenceRegistry",
    "ConversationEvidenceRegistryFactory",
    "ConversationNotFoundError",
    "ConversationOwnershipError",
    "ConversationScopedInMemoryEvidenceRegistry",
    "ConversationScopedInMemoryEvidenceRegistryFactory",
    "DuplicateEvidenceError",
    "EvidenceContextIntegrityError",
    "EvidenceContextUnavailableError",
    "EvidenceConversationMismatchError",
    "EvidenceRegistry",
    "EvidenceRegistryError",
    "EvidenceRegistryFactory",
    "EvidenceRequestMismatchError",
    "FinalEvidenceContextBuildError",
    "FinalEvidenceContextBuilder",
    "InMemoryEvidenceRegistry",
    "InMemoryEvidenceRegistryFactory",
]
