import json

import pytest

from smart_waste_multimodal.query_rewriting.validation import (
    RewriteResult,
    validate_rewrite_response,
)


def test_valid_response_is_parsed():
    raw = json.dumps({
        "rewritten_query": "glass recycling broken glass safety",
        "preserved_aspects": [
            "glass recycling",
            "broken glass safety",
        ],
    })

    result = validate_rewrite_response(raw)

    assert result == RewriteResult(
        rewritten_query="glass recycling broken glass safety",
        preserved_aspects=("glass recycling", "broken glass safety"),
    )


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "null"])
def test_invalid_json_or_empty_response_is_rejected(raw):
    with pytest.raises(ValueError):
        validate_rewrite_response(raw)


def test_missing_field_is_rejected():
    raw = json.dumps({"rewritten_query": "glass recycling"})

    with pytest.raises(ValueError):
        validate_rewrite_response(raw)


def test_unexpected_field_is_rejected():
    raw = json.dumps({
        "rewritten_query": "glass recycling",
        "preserved_aspects": [],
        "answer": "unsupported extra field",
    })

    with pytest.raises(ValueError):
        validate_rewrite_response(raw)


def test_empty_rewrite_is_rejected():
    raw = json.dumps({
        "rewritten_query": " ",
        "preserved_aspects": ["glass recycling"],
    })

    with pytest.raises(ValueError):
        validate_rewrite_response(raw)


def test_invalid_preserved_aspects_are_rejected():
    raw = json.dumps({
        "rewritten_query": "glass recycling",
        "preserved_aspects": ["glass recycling", 123],
    })

    with pytest.raises(ValueError):
        validate_rewrite_response(raw)


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        validate_rewrite_response(None)
