"""Architecture conformance and decision-boundary integration tests."""

import ast
from pathlib import Path

from smart_waste_multimodal.adapters.mock_adapters import (
    MockGenerationAdapter,
    MockRetrievalAdapter,
    MockVisionAdapter,
)
from smart_waste_multimodal.contracts import FinalEvidenceContext
from smart_waste_multimodal.contracts.evidence import EvidenceSourceType
from smart_waste_multimodal.evidence import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.final_decision import (
    FinalDecisionEngine,
)
from smart_waste_multimodal.orchestration.orchestrator import PipelineOrchestrator
from tests.integration.test_pipeline_e2e import make_e2e_request


class RecordingRegistryFactory:
    def __init__(self):
        self.delegate = ConversationScopedInMemoryEvidenceRegistryFactory()
        self.registries = {}

    def create(self, conversation, *, actor_id):
        registry = self.delegate.create(conversation, actor_id=actor_id)
        self.registries[conversation.conversation_id] = registry
        return registry


class GenerationInputProbe:
    """Records the current Generation input boundary."""

    def __init__(self):
        self.received_type = None
        self.delegate = MockGenerationAdapter()

    def __call__(self, request, final_evidence_context):
        self.received_type = type(final_evidence_context)
        return self.delegate(request, final_evidence_context)


class RecordingFinalDecisionEngine:
    """Records invocation timing while preserving real decision behavior."""

    def __init__(self):
        self.delegate = FinalDecisionEngine()
        self.calls = []

    def decide(self, **kwargs):
        registry = kwargs["evidence_registry"]
        records_before_decision = registry.list_all(actor_id=registry.owner_id)
        result = self.delegate.decide(**kwargs)

        self.calls.append(
            {
                "request_id": kwargs["request"].request_id,
                "evidence_count": len(records_before_decision),
                "result": result,
            }
        )
        return result


def _read_package_sources(package_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(package_root)): path.read_text(encoding="utf-8")
        for path in package_root.rglob("*.py")
    }


def _parse_sources(sources: dict[str, str]) -> dict[str, ast.Module]:
    return {name: ast.parse(source, filename=name) for name, source in sources.items()}


def _calls_method_on_attribute(
    tree: ast.AST,
    owner_attribute: str,
    method_name: str,
) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr != method_name:
            continue

        owner = function.value
        if isinstance(owner, ast.Attribute) and owner.attr == owner_attribute:
            return True

    return False


def _contains_executable_string(
    tree: ast.AST,
    value: str,
) -> bool:
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == value
        for node in ast.walk(tree)
    )


def _declared_symbols(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    }


def test_pipeline_registers_vision_and_retrieval_evidence():
    factory = RecordingRegistryFactory()
    generation = GenerationInputProbe()

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=MockRetrievalAdapter(),
        generation_stage=generation,
        evidence_registry_factory=factory,
    )

    request = make_e2e_request()
    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert response.generation is not None
    assert response.final_decision is not None

    registry = factory.registries[request.conversation_id]
    evidence = registry.list_all(actor_id=registry.owner_id)
    source_types = {item.source_type for item in evidence}

    assert EvidenceSourceType.VISION_SNAPSHOT in source_types
    assert EvidenceSourceType.VISION_DETECTION in source_types
    assert EvidenceSourceType.RETRIEVAL_CHUNK in source_types

    # Generation must consume only the Final Evidence Context.
    assert generation.received_type is FinalEvidenceContext
    assert response.final_evidence_context is not None
    assert response.final_evidence_context.evidence_ids == tuple(
        response.final_decision.evidence_ids
    )

    assert all(item.request_id == request.request_id for item in evidence)


def test_final_decision_is_called_once_after_evidence_registration():
    factory = RecordingRegistryFactory()
    decision_engine = RecordingFinalDecisionEngine()

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=MockRetrievalAdapter(),
        generation_stage=MockGenerationAdapter(),
        evidence_registry_factory=factory,
        final_decision_engine=decision_engine,
    )

    request = make_e2e_request()
    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert len(decision_engine.calls) == 1

    call = decision_engine.calls[0]
    assert call["request_id"] == request.request_id
    assert call["evidence_count"] > 0

    assert response.final_decision is not None
    assert response.final_decision == call["result"]
    assert response.final_decision.execution_status == "DECIDED"


def test_architecture_conformance_report():
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "src" / "smart_waste_multimodal"

    sources = _read_package_sources(package_root)
    trees = _parse_sources(sources)

    orchestrator_tree = trees["orchestration/orchestrator.py"]
    retrieval_tree = trees["contracts/retrieval.py"]
    response_tree = trees["contracts/response.py"]

    rewrite_trees = {
        name: tree
        for name, tree in trees.items()
        if name.startswith("query_rewriting/")
    }

    adapter_source = sources["retrieval/hybrid_rag_adapter.py"]
    bootstrap_source = sources["retrieval/bootstrap.py"]
    orchestrator_source = sources["orchestration/orchestrator.py"]

    rewrite_connected = (
        "rewrite_for_retrieval" in adapter_source
        and "self.retrieval_stage" in orchestrator_source
        and "context_assessment" in orchestrator_source
        and 'rewrite_mode: RewriteMode = "selective"' in bootstrap_source
    )

    context_connected = (
        "context_analyzer" in orchestrator_source
        and "context_validator" in orchestrator_source
    )

    recovery_connected = _calls_method_on_attribute(
        orchestrator_tree,
        "context_recovery_orchestrator",
        "resolve",
    )

    final_decision_connected = _calls_method_on_attribute(
        orchestrator_tree,
        "final_decision_engine",
        "decide",
    )

    generation_receives_raw_retrieval = (
        "self.generation_stage,\n                request,\n                vision,\n                retrieval,"
        in orchestrator_source
    )
    generation_receives_final_context = (
        "self.generation_stage,\n                request,\n                final_evidence_context,"
        in orchestrator_source
    )

    # This is a diagnostic signal only. Behavioral tests must prove the gate.
    generation_gate_present = (
        "final_decision.action" in orchestrator_source
        and '"DIRECT"' in orchestrator_source
    )

    human_review_leaks = _contains_executable_string(
        retrieval_tree, "HUMAN_REVIEW"
    ) or any(
        _contains_executable_string(tree, "HUMAN_REVIEW")
        for tree in rewrite_trees.values()
    )

    response_validates_final_decision = (
        "from .final_decision import FinalDecisionResult"
        in sources["contracts/response.py"]
        and _contains_executable_string(response_tree, "HUMAN_REVIEW")
        and _contains_executable_string(response_tree, "needs_clarification")
    )

    all_symbols = set()
    for tree in trees.values():
        all_symbols.update(_declared_symbols(tree))

    candidate_stages_present = any(
        "CandidateGeneration" in symbol
        or "CandidateRanking" in symbol
        or "candidate_generation" in symbol.lower()
        or "candidate_ranking" in symbol.lower()
        for symbol in all_symbols
    )

    statuses = [
        (
            "Context Analyzer + Validator",
            "PASS" if context_connected else "FAIL",
            "Context analysis and validation are connected to the orchestrator.",
        ),
        (
            "Bounded Context Recovery",
            "PASS" if recovery_connected else "FAIL",
            "Orchestrator calls ContextRecoveryOrchestrator.resolve().",
        ),
        (
            "Selective Query Rewriting",
            "PASS" if rewrite_connected else "FAIL",
            "Selective rewriting is connected through the retrieval adapter.",
        ),
        (
            "Final Decision integration",
            "PASS" if final_decision_connected else "FAIL",
            "Orchestrator calls the injected FinalDecisionEngine.decide().",
        ),
        (
            "HUMAN_REVIEW ownership",
            "FAIL" if human_review_leaks else "PASS",
            (
                "No executable HUMAN_REVIEW decision literal in Retrieval "
                "or Query Rewriting."
                if not human_review_leaks
                else "HUMAN_REVIEW decision logic exists outside Final Decision."
            ),
        ),
        (
            "Response decision consistency",
            "PASS" if response_validates_final_decision else "FAIL",
            "Response validates HUMAN_REVIEW against FinalDecisionResult.",
        ),
        (
            "Candidate Generation / Ranking removal",
            "FAIL" if candidate_stages_present else "PASS",
            (
                "No candidate-generation or candidate-ranking symbols found."
                if not candidate_stages_present
                else "A prohibited candidate stage symbol remains."
            ),
        ),
        (
            "Final Evidence Context",
            (
                "FAIL"
                if generation_receives_raw_retrieval
                or not generation_receives_final_context
                else "PASS"
            ),
            (
                "Generation must consume FinalEvidenceContext only."
                if generation_receives_raw_retrieval
                or not generation_receives_final_context
                else "Generation receives the authorized FinalEvidenceContext."
            ),
        ),
        (
            "Final Decision Generation Gate",
            "GAP" if not generation_gate_present else "REVIEW",
            (
                "Generation is not visibly gated by a DIRECT final decision."
                if not generation_gate_present
                else "A possible gate pattern exists; behavioral verification remains required."
            ),
        ),
    ]

    print("\n" + "=" * 100)
    print("ARCHITECTURE VERIFICATION & DECISION LEAKAGE AUDIT")
    print("=" * 100)

    counts = {}
    for name, status, detail in statuses:
        print(f"[{status:8}] {name}")
        print(f"          {detail}")
        counts[status] = counts.get(status, 0) + 1

    print("-" * 100)
    for status in ("PASS", "GAP", "REVIEW", "FAIL"):
        print(f"{status}: {counts.get(status, 0)}")
    print("=" * 100)

    assert context_connected
    assert recovery_connected
    assert rewrite_connected
    assert final_decision_connected
    assert not human_review_leaks
    assert response_validates_final_decision
    assert not candidate_stages_present


def _test_conversation_for_request(request):
    """Build a valid test Conversation for the supplied AnalyzeRequest."""
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.conversation import Conversation

    now = datetime.now(UTC)
    return Conversation(
        conversation_id=request.conversation_id,
        owner_id="test-owner",
        created_at=now,
        updated_at=now,
    )
