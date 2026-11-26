"""Gemini Interactions API query rewriter with a bounded HTTP timeout."""

from __future__ import annotations

import os
from collections.abc import Callable

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_MS = 30_000


def build_gemini_rewriter(
    *,
    model: str | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Callable[[list[dict[str, str]]], str]:
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in the environment."
        )

    selected_model = (
        model
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK is unavailable. Install the google-genai package."
        ) from exc

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retryOptions=types.HttpRetryOptions(attempts=1),
        ),
    )

    def rewrite(messages: list[dict[str, str]]) -> str:
        system_instruction = next(
            (
                item["content"]
                for item in messages
                if item.get("role") == "system"
            ),
            "",
        )
        user_messages = [
            item["content"]
            for item in messages
            if item.get("role") == "user"
        ]

        if not user_messages:
            raise ValueError("At least one user message is required.")

        interaction = client.interactions.create(
            model=selected_model,
            input="\n\n".join(user_messages),
            system_instruction=system_instruction or None,
            response_format={
                "type": "text",
                "mime_type": "application/json",
            },
        )

        output = interaction.output_text
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("Gemini returned an empty response.")

        return output.strip()

    return rewrite
