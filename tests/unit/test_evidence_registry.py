from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.evidence import (
    DuplicateEvidenceError,
    EvidenceRequestMismatchError,
    InMemoryEvidenceRegistry,
)


def make_evidence(
    evidence_id="evidence-001",
    request_id="request-001",
    content=None,
):
    return Evidence(
        evidence_id=evidence_id,
        request_id=request_id,
        source_type=EvidenceSourceType.VISION_DETECTION,
        provenance=EvidenceProvenance(
            source_id="artifact-001",
            source_version="model-v1",
            source_locator={"detection_index": "0"},
        ),
        content=content if content is not None else {
            "class_name": "glass",
            "confidence": 0.93,
            "bbox_xyxy": [10, 20, 80, 100],
        },
        created_at=datetime.now(UTC),
        metadata={"origin": "unit-test"},
               conversation_id="conversation-001",
               created_by_request_id=request_id,
               origin_turn_id="turn-001",
    )


def test_register_and_roundtrip_preserves_structured_json():
    registry = InMemoryEvidenceRegistry("request-001")
    evidence = make_evidence()

    registry.register(evidence)
    restored = registry.get("evidence-001")

    assert restored is not None
    assert restored.content == evidence.content
    assert restored.provenance.source_id == "artifact-001"
    assert restored.provenance.source_locator == {
        "detection_index": "0"
    }


def test_registry_snapshot_isolated_from_input_mutation():
    registry = InMemoryEvidenceRegistry("request-001")
    evidence = make_evidence()

    registry.register(evidence)
    evidence.content["class_name"] = "modified"
    evidence.metadata["origin"] = "modified"

    restored = registry.get("evidence-001")
    assert restored is not None
    assert restored.content["class_name"] == "glass"
    assert restored.metadata["origin"] == "unit-test"


def test_returned_nested_mutation_does_not_change_registry_snapshot():
    registry = InMemoryEvidenceRegistry("request-001")
    registry.register(make_evidence())

    returned = registry.get("evidence-001")
    assert returned is not None
    returned.content["class_name"] = "modified"

    again = registry.get("evidence-001")
    assert again is not None
    assert again.content["class_name"] == "glass"


def test_evidence_top_level_fields_are_frozen():
    evidence = make_evidence()

    with pytest.raises(ValidationError):
        evidence.evidence_id = "changed"


def test_registry_rejects_evidence_from_another_request():
    registry = InMemoryEvidenceRegistry("request-001")

    with pytest.raises(EvidenceRequestMismatchError):
        registry.register(make_evidence(request_id="request-002"))


def test_registry_rejects_duplicate_identifiers_without_overwrite():
    registry = InMemoryEvidenceRegistry("request-001")
    registry.register(make_evidence())

    with pytest.raises(DuplicateEvidenceError):
        registry.register(
            make_evidence(content={"class_name": "metal"})
        )

    assert registry.get("evidence-001").content["class_name"] == "glass"


def test_get_many_preserves_requested_order_and_omits_missing_ids():
    registry = InMemoryEvidenceRegistry("request-001")
    registry.register(make_evidence(evidence_id="e-002"))
    registry.register(make_evidence(evidence_id="e-001"))

    results = registry.get_many(["e-002", "missing", "e-001"])

    assert [item.evidence_id for item in results] == ["e-002", "e-001"]


def test_list_all_uses_deterministic_identifier_order():
    registry = InMemoryEvidenceRegistry("request-001")
    registry.register(make_evidence(evidence_id="e-003"))
    registry.register(make_evidence(evidence_id="e-001"))
    registry.register(make_evidence(evidence_id="e-002"))

    assert [
        item.evidence_id for item in registry.list_all()
    ] == ["e-001", "e-002", "e-003"]


def test_registry_is_request_scoped():
    first = InMemoryEvidenceRegistry("request-001")
    second = InMemoryEvidenceRegistry("request-002")

    first.register(make_evidence())

    assert first.get("evidence-001") is not None
    assert second.get("evidence-001") is None


@pytest.mark.parametrize(
    "invalid_content",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": object()},
        {1: "non-string-key"},
    ],
)
def test_evidence_rejects_non_json_compatible_content(invalid_content):
    with pytest.raises(ValidationError):
        make_evidence(content=invalid_content)


def test_evidence_requires_timezone_aware_created_at():
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="e-1",
            request_id="request-001",
            source_type=EvidenceSourceType.CONVERSATION_TEXT,
            provenance=EvidenceProvenance(source_id="turn-1"),
            content="hello",
            created_at=datetime(2026, 1, 1),
            conversation_id="conversation-001",
            created_by_request_id="request-001",
            origin_turn_id="turn-001",
        )


def test_evidence_schema_version_defaults_to_two():
    evidence = make_evidence()

    assert evidence.schema_version == 2


def test_old_evidence_payload_defaults_to_schema_version_two():
    evidence = make_evidence()
    payload = evidence.model_dump()
    payload.pop("schema_version")

    restored = Evidence.model_validate(payload)

    assert restored.schema_version == 2


def test_evidence_accepts_future_positive_schema_versions():
    evidence = make_evidence()
    payload = evidence.model_dump()
    payload["schema_version"] = 2

    restored = Evidence.model_validate(payload)

    assert restored.schema_version == 2


@pytest.mark.parametrize("invalid_version", [0, -1, True, "2"])
def test_evidence_rejects_invalid_schema_versions(invalid_version):
    evidence = make_evidence()
    payload = evidence.model_dump()
    payload["schema_version"] = invalid_version

    with pytest.raises(ValidationError):
        Evidence.model_validate(payload)


def test_registry_roundtrip_preserves_schema_version():
    registry = InMemoryEvidenceRegistry("request-001")
    evidence = make_evidence()
    payload = evidence.model_dump()
    payload["schema_version"] = 2
    versioned_evidence = Evidence.model_validate(payload)

    registry.register(versioned_evidence)
    restored = registry.get(versioned_evidence.evidence_id)

    assert restored is not None
    assert restored.schema_version == 2

def test_evidence_requires_new_scope_identifiers():
    evidence = make_evidence()
    payload = evidence.model_dump()

    for field in (
        "conversation_id",
        "created_by_request_id",
        "origin_turn_id",
    ):
        invalid_payload = dict(payload)
        invalid_payload.pop(field)
        with pytest.raises(ValidationError):
            Evidence.model_validate(invalid_payload)


@pytest.mark.parametrize(
    "field",
    ["conversation_id", "created_by_request_id", "origin_turn_id"],
)
@pytest.mark.parametrize("invalid_value", ["", "   "])
def test_evidence_rejects_empty_scope_identifiers(field, invalid_value):
    payload = make_evidence().model_dump()
    payload[field] = invalid_value

    with pytest.raises(ValidationError):
        Evidence.model_validate(payload)


def test_evidence_allows_request_creator_mismatch():
    payload = make_evidence().model_dump()
    payload["created_by_request_id"] = "original-request"

    evidence = Evidence.model_validate(payload)

    assert evidence.request_id == "request-001"
    assert evidence.created_by_request_id == "original-request"


def test_evidence_roundtrip_preserves_conversation_and_turn_scope():
    evidence = make_evidence()
    restored = Evidence.model_validate_json(evidence.model_dump_json())

    assert restored.conversation_id == evidence.conversation_id
    assert restored.created_by_request_id == evidence.created_by_request_id
    assert restored.origin_turn_id == evidence.origin_turn_id
