import pytest
from pydantic import ValidationError

from smart_waste_multimodal.config import ContextConfig


def test_context_config_defaults():
    config = ContextConfig()

    assert config.max_age_seconds == 1800
    assert config.max_history_turns == 20


def test_context_config_custom_values():
    config = ContextConfig(
        max_age_seconds=900,
        max_history_turns=10,
    )

    assert config.max_age_seconds == 900
    assert config.max_history_turns == 10


@pytest.mark.parametrize("value", [0, -1])
def test_context_config_rejects_invalid_max_age(value):
    with pytest.raises(ValidationError):
        ContextConfig(max_age_seconds=value)


@pytest.mark.parametrize("value", [0, -1])
def test_context_config_rejects_invalid_history_limit(value):
    with pytest.raises(ValidationError):
        ContextConfig(max_history_turns=value)
