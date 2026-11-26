from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import smart_waste_multimodal.contracts.request as request_module
from smart_waste_multimodal.contracts.conversation import ConversationTurn
from smart_waste_multimodal.contracts.request import AnalyzeRequest

FIXED_NOW = datetime(2030, 5, 10, 12, 0, 0, tzinfo=UTC)


def make_request(timestamp):
    turn = ConversationTurn(
        turn_id="turn-1",
        role="user",
        content="What material is this?",
        timestamp=timestamp,
    )

    return AnalyzeRequest(
        request_id="request-1",
        turn_timestamp=FIXED_NOW,
        conversation_id="test-conversation-test_future_timestamps-22",
        turn_id="test-turn-test_future_timestamps-22",
        image={
            "artifact_id": "image-1",
            "media_type": "image/jpeg",
        },
        question="What material is this?",
        conversation_history=[turn],
    )


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(
        request_module,
        "_utc_now",
        lambda: FIXED_NOW,
    )


def test_rejects_future_timestamp():
    future = FIXED_NOW + timedelta(microseconds=1)

    with pytest.raises(
        ValidationError,
        match="must not be in the future",
    ):
        make_request(future)


def test_accepts_timestamp_equal_to_now():
    request = make_request(FIXED_NOW)

    assert len(request.conversation_history) == 1


def test_accepts_past_timestamp():
    past = FIXED_NOW - timedelta(days=1)

    request = make_request(past)

    assert len(request.conversation_history) == 1


def test_future_timestamp_is_rejected_across_timezones():
    offset = timezone(timedelta(hours=5))
    future_in_offset = (
        FIXED_NOW + timedelta(microseconds=1)
    ).astimezone(offset)

    with pytest.raises(
        ValidationError,
        match="must not be in the future",
    ):
        make_request(future_in_offset)


def test_equivalent_instant_in_another_timezone_is_accepted():
    offset = timezone(timedelta(hours=5))
    same_instant = FIXED_NOW.astimezone(offset)

    request = make_request(same_instant)

    assert len(request.conversation_history) == 1
