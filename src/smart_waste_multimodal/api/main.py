"""FastAPI transport layer for the local single-user deployment."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from smart_waste_multimodal.contracts import (
    AnalyzeResponse,
    RequestOptions,
)
from smart_waste_multimodal.deployment import (
    ApplicationServices,
    build_application_services,
)


MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_application_services()
    yield
    app.state.services = None


app = FastAPI(
    title="Smart Waste Multimodal AI",
    version="0.1.0",
    description="Local multimodal waste detection and recycling knowledge API.",
    lifespan=lifespan,
)


def _services() -> ApplicationServices:
    services = getattr(app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=503,
            detail="Application services are not ready.",
        )
    return services


def _actor_id(services: ApplicationServices) -> str:
    # This identity comes from trusted backend configuration, never from
    # request JSON, query parameters, or multipart form fields.
    return services.actor_id


def _is_gemini_configured() -> bool:
    import os
    return bool(
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "smart-waste-multimodal-ai"}


@app.post("/api/v1/conversations", status_code=201)
def create_conversation():
    services = _services()
    conversation = services.application_service.create_conversation(
        actor_id=_actor_id(services),
    )
    return conversation.model_dump(mode="json")


@app.post(
    "/api/v1/conversations/{conversation_id}/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    conversation_id: str,
    request_id: str = Form(..., min_length=1, max_length=200),
    task_type: Literal["detection", "knowledge_question"] = Form(
        "knowledge_question"
    ),
    question: str | None = Form(None, max_length=4000),
    locale: Literal["en", "fa"] = Form("en"),
    options: str = Form("{}"),
    image: UploadFile | None = File(None),
):
    services = _services()
    actor_id = _actor_id(services)

    try:
        request_options = RequestOptions.model_validate_json(options)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="The options JSON does not match the request contract.",
        ) from exc

    if request_options.generate_answer and not _is_gemini_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Answer generation requires GEMINI_API_KEY or GOOGLE_API_KEY. "
                "Configure a key or set options.generate_answer to false."
            ),
        )

    image_bytes = None
    image_media_type = None

    if image is not None:
        image_media_type = image.content_type
        if image_media_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=415,
                detail="Only JPEG and PNG images are supported.",
            )

        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 15 MiB upload limit.",
            )

    try:
        return await run_in_threadpool(
            services.application_service.execute,
            actor_id=actor_id,
            conversation_id=conversation_id,
            request_id=request_id,
            task_type=task_type,
            question=question,
            locale=locale,
            options=request_options,
            image_bytes=image_bytes,
            image_media_type=image_media_type,
        )
    except Exception as exc:
        name = type(exc).__name__

        if name in {
            "RequestFingerprintConflictError",
            "RequestAlreadyProcessingError",
            "PreviouslyFailedRequestError",
        }:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        if name in {
            "ArtifactOwnershipError",
            "ConversationOwnershipError",
        }:
            raise HTTPException(status_code=403, detail="Access denied.") from exc

        if name in {
            "ConversationNotFoundError",
            "ConversationClosedError",
        }:
            raise HTTPException(
                status_code=404 if name == "ConversationNotFoundError" else 409,
                detail="Conversation is unavailable.",
            ) from exc

        if name in {
            "InvalidImageError",
            "UnsupportedImageFormatError",
            "ImageSizeLimitError",
            "DuplicateImageArtifactError",
            "ImageStoreCapacityError",
        }:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if name in {"ValueError", "ValidationError"}:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        raise HTTPException(
            status_code=500,
            detail="The analysis request could not be completed.",
        ) from exc


@app.post("/api/v1/conversations/{conversation_id}/close")
def close_conversation(conversation_id: str):
    services = _services()
    actor_id = _actor_id(services)

    try:
        closed = services.conversation_store.close(
            conversation_id,
            actor_id=actor_id,
        )
    except Exception as exc:
        name = type(exc).__name__
        if name == "ConversationOwnershipError":
            raise HTTPException(status_code=403, detail="Access denied.") from exc
        if name == "ConversationNotFoundError":
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        if name == "ConversationClosedError":
            raise HTTPException(status_code=409, detail="Conversation is already closed.") from exc
        raise

    # A conversation can be closed before its first analysis, in which case
    # no evidence registry exists yet.
    try:
        services.evidence_registry_factory.close_conversation(
            conversation_id,
            actor_id=actor_id,
            closed_at=closed.closed_at,
        )
    except Exception as exc:
        if type(exc).__name__ != "ConversationNotFoundError":
            raise HTTPException(
                status_code=500,
                detail="Conversation was closed but evidence cleanup failed.",
            ) from exc

    return closed.model_dump(mode="json")


@app.delete(
    "/api/v1/conversations/{conversation_id}",
    status_code=204,
)
def delete_conversation(conversation_id: str):
    services = _services()
    actor_id = _actor_id(services)

    # Registry deletion validates ownership when a registry exists.
    try:
        services.evidence_registry_factory.delete_conversation(
            conversation_id,
            actor_id=actor_id,
        )
    except Exception as exc:
        if type(exc).__name__ != "ConversationNotFoundError":
            if type(exc).__name__ == "ConversationOwnershipError":
                raise HTTPException(status_code=403, detail="Access denied.") from exc
            raise HTTPException(
                status_code=500,
                detail="Evidence cleanup failed.",
            ) from exc

    try:
        deleted = services.conversation_store.delete(
            conversation_id,
            actor_id=actor_id,
        )
    except Exception as exc:
        if type(exc).__name__ == "ConversationOwnershipError":
            raise HTTPException(status_code=403, detail="Access denied.") from exc
        raise

    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return Response(status_code=204)
