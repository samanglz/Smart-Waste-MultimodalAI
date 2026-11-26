from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from smart_waste_multimodal.contracts.context import ContextAnalysisOutput
from smart_waste_multimodal.contracts.evidence import EvidenceSourceType
from smart_waste_multimodal.contracts.conversation import ConversationTurn
from smart_waste_multimodal.query_rewriting.context_analyzer import (
    ContextAnalyzer,
)

NOW = datetime(2030, 5, 10, 12, 0, tzinfo=UTC)


def make_turn(
    turn_id="turn-1",
    role="user",
    content="I have a glass bottle.",
):
    return ConversationTurn(
        turn_id=turn_id,
        role=role,
        content=content,
        timestamp=NOW,
    )


def resolved_output():
    return ContextAnalysisOutput(
        status="resolved",
        referenced_turn_ids=["turn-1"],
        evidence_ids=["analyzer-test-conversation-text-evidence"],
        required_evidence_types=[
            EvidenceSourceType.CONVERSATION_TEXT
        ],
        resolved_context="The item is a glass bottle.",
        signals=["object_reference_resolved"],
        reason="The user identified the item.",
    )


def independent_output():
    return ContextAnalysisOutput(
        status="independent",
        reason="The query is self-contained.",
        signals=["self_contained_query"],
    )


class FakeModels:
    def __init__(self, responses=None, errors=None):
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        if self.errors:
            error = self.errors.pop(0)
            raise error

        if not self.responses:
            raise RuntimeError("No fake response configured.")

        return self.responses.pop(0)


class FakeClient:
    def __init__(self, models):
        self.models = models


def fake_response(output):
    return SimpleNamespace(
        parsed=output,
        text=output.model_dump_json(),
    )


def test_constructor_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        ContextAnalyzer(timeout_seconds=0)


def test_constructor_rejects_invalid_attempt_count():
    with pytest.raises(ValueError):
        ContextAnalyzer(max_attempts=0)


def test_constructor_rejects_negative_retry_delay():
    with pytest.raises(ValueError):
        ContextAnalyzer(retry_delay_seconds=-0.1)


def test_empty_query_is_rejected_without_model_call():
    models = FakeModels()
    analyzer = ContextAnalyzer(client=FakeClient(models))

    with pytest.raises(ValueError, match="non-empty query"):
        analyzer.analyze("   ", [make_turn()])

    assert models.calls == []


def test_empty_history_returns_independent_without_model_call():
    models = FakeModels()
    analyzer = ContextAnalyzer(client=FakeClient(models))

    result = analyzer.analyze("What is glass?", [])

    assert result.status == "independent"
    assert "empty_conversation_history" in result.signals
    assert models.calls == []


def test_valid_resolved_output_is_returned():
    output = resolved_output()
    models = FakeModels(responses=[fake_response(output)])
    analyzer = ContextAnalyzer(client=FakeClient(models))

    result = analyzer.analyze(
        "What about this bottle?",
        [make_turn()],
    )

    assert result == output
    assert len(models.calls) == 1
    assert models.calls[0]["model"] == analyzer.model
    assert "What about this bottle?" in models.calls[0]["contents"]


def test_independent_output_is_returned():
    output = independent_output()
    models = FakeModels(responses=[fake_response(output)])
    analyzer = ContextAnalyzer(client=FakeClient(models))

    result = analyzer.analyze("What is glass?", [make_turn()])

    assert result.status == "independent"


@pytest.mark.parametrize(
    "status",
    ["insufficient", "ambiguous", "conflict"],
)
def test_unresolved_statuses_are_preserved(status):
    output = ContextAnalysisOutput(
        status=status,
        referenced_turn_ids=["turn-1"],
        reason="The reference cannot be safely resolved.",
    )
    models = FakeModels(responses=[fake_response(output)])
    analyzer = ContextAnalyzer(client=FakeClient(models))

    result = analyzer.analyze("What about it?", [make_turn()])

    assert result.status == status


def test_history_is_limited_to_configured_maximum():
    from smart_waste_multimodal.config.context import ContextConfig

    turns = [
        make_turn(turn_id=f"turn-{index}")
        for index in range(4)
    ]
    output = independent_output()
    models = FakeModels(responses=[fake_response(output)])

    analyzer = ContextAnalyzer(
        client=FakeClient(models),
        config=ContextConfig(max_history_turns=2),
    )

    analyzer.analyze("What is glass?", turns)

    prompt = models.calls[0]["contents"]

    assert '"turn_id":"turn-2"' in prompt
    assert '"turn_id":"turn-3"' in prompt
    assert '"turn_id":"turn-0"' not in prompt
    assert '"turn_id":"turn-1"' not in prompt


def test_vision_snapshot_is_included_in_its_source_turn():
    from smart_waste_multimodal.contracts.vision import (
        Detection,
        VisionResult,
    )

    vision = VisionResult(
        status="success",
        detections=[
            Detection(
                class_name="glass",
                confidence=0.95,
                bbox_xyxy=(1, 1, 20, 20),
            )
        ],
        image_width=100,
        image_height=100,
        artifact_id="image-1",
    )

    turn = ConversationTurn(
        turn_id="turn-vision",
        role="user",
        content="Identify this item.",
        timestamp=NOW,
        vision=vision,
    )

    analyzer = ContextAnalyzer(
        client=FakeClient(
            FakeModels(responses=[fake_response(independent_output())])
        )
    )

    analyzer.analyze("What is this?", [turn])

    prompt = analyzer._build_prompt("What is this?", [turn])

    assert '"turn_id":"turn-vision"' in prompt
    assert '"source_type"' not in prompt
    assert '"class_name":"glass"' in prompt


def test_retry_succeeds_after_transient_failure():
    models = FakeModels(
        errors=[RuntimeError("Temporary provider failure")],
        responses=[fake_response(independent_output())],
    )
    analyzer = ContextAnalyzer(
        client=FakeClient(models),
        max_attempts=2,
        retry_delay_seconds=0,
    )

    result = analyzer.analyze("What is glass?", [make_turn()])

    assert result.status == "independent"
    assert len(models.calls) == 2


def test_failure_after_max_attempts_raises_runtime_error():
    models = FakeModels(
        errors=[
            RuntimeError("Failure one"),
            RuntimeError("Failure two"),
        ]
    )
    analyzer = ContextAnalyzer(
        client=FakeClient(models),
        max_attempts=2,
        retry_delay_seconds=0,
    )

    with pytest.raises(
        RuntimeError,
        match="failed after 2 attempt",
    ):
        analyzer.analyze("What about it?", [make_turn()])

    assert len(models.calls) == 2


def test_invalid_model_json_is_retried():
    invalid_response = SimpleNamespace(
        parsed=None,
        text='{"status": "unknown"}',
    )
    models = FakeModels(
        responses=[
            invalid_response,
            fake_response(independent_output()),
        ]
    )
    analyzer = ContextAnalyzer(
        client=FakeClient(models),
        max_attempts=2,
        retry_delay_seconds=0,
    )

    result = analyzer.analyze("What is glass?", [make_turn()])

    assert result.status == "independent"
    assert len(models.calls) == 2


def test_prompt_contains_evidence_and_uncertainty_rules():
    analyzer = ContextAnalyzer(client=FakeClient(FakeModels()))

    prompt = analyzer._build_prompt(
        "What about this?",
        [make_turn()],
    )

    assert "Never invent missing facts" in prompt
    assert "Assistant messages can help interpret" in prompt
    assert "Do not return a status named stale" in prompt
    assert "CURRENT QUERY:" in prompt
    assert "CONVERSATION HISTORY JSON:" in prompt


@pytest.mark.parametrize(
    "query",
    [
        "What is glass?",
        "What is the capital of Japan?",
    ],
)
def test_standalone_query_with_nonempty_unrelated_history_does_not_require_gemini(
    monkeypatch,
    query,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fail_if_client_requested(self):
        raise AssertionError("Gemini client must not be requested.")

    monkeypatch.setattr(
        ContextAnalyzer,
        "_get_client",
        fail_if_client_requested,
    )

    analyzer = ContextAnalyzer()
    history = [
        make_turn(
            turn_id="turn-unrelated",
            content="Yesterday I asked about recycling a glass bottle.",
        )
    ]

    result = analyzer.analyze(query, history)

    assert result.status == "independent"
    assert result.resolved_context is None
    assert result.evidence_ids == []
    assert result.required_evidence_types == []
    assert "deterministic_standalone_query" in result.signals


def test_explicit_reference_without_gemini_is_not_marked_independent(
    monkeypatch,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fail_if_client_requested(self):
        raise AssertionError("Gemini client must not be requested.")

    monkeypatch.setattr(
        ContextAnalyzer,
        "_get_client",
        fail_if_client_requested,
    )

    analyzer = ContextAnalyzer()
    history = [
        make_turn(
            turn_id="turn-bottle",
            content="I have a glass bottle.",
        )
    ]

    result = analyzer.analyze("What about this bottle?", history)

    assert result.status == "ambiguous"
    assert result.resolved_context is None
    assert result.evidence_ids == []
    assert result.required_evidence_types == []
    assert "context_dependency_not_ruled_out" in result.signals


def test_ambiguous_reference_without_gemini_fails_closed(
    monkeypatch,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fail_if_client_requested(self):
        raise AssertionError("Gemini client must not be requested.")

    monkeypatch.setattr(
        ContextAnalyzer,
        "_get_client",
        fail_if_client_requested,
    )

    analyzer = ContextAnalyzer()
    history = [
        make_turn(
            turn_id="turn-previous",
            content="I have a glass bottle.",
        )
    ]

    result = analyzer.analyze("What about it?", history)

    assert result.status == "ambiguous"
    assert result.resolved_context is None
    assert result.evidence_ids == []
    assert result.required_evidence_types == []
    assert result.reason.strip()
