# Local Docker Deployment

## Requirements

- Docker Engine with Docker Compose
- The ONNX model at `exports/best.onnx`
- The retrieval index files under `data/retrieval_index/v4_expanded/`

## Configure

Copy `.env.example` to `.env`.

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to enable answer generation and
Gemini-backed context analysis. The application can start without a Gemini key,
but requests that ask for answer generation are rejected until a key is configured.

## Build and run

```bash
docker compose up --build -d
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

API documentation:

- http://127.0.0.1:8000/docs

## API flow

1. `POST /api/v1/conversations`
2. `POST /api/v1/conversations/{conversation_id}/analyze`
3. `POST /api/v1/conversations/{conversation_id}/close`
4. `DELETE /api/v1/conversations/{conversation_id}`

The analyze endpoint accepts multipart form fields:
`request_id`, `task_type`, `question`, `locale`, `options` (JSON), and optional
`image` (JPEG or PNG, maximum 15 MiB).

Example options for image detection without answer generation:

```json
{
  "run_vision": true,
  "run_retrieval": false,
  "generate_answer": false,
  "retrieval_query_source": "question"
}
```

Example options for a knowledge question without answer generation:

```json
{
  "run_vision": false,
  "run_retrieval": true,
  "generate_answer": false,
  "retrieval_query_source": "question"
}
```

## Runtime data lifecycle

Conversation, image, request-execution, and evidence stores are in memory.
They are shared across requests handled by this application process and are
lost when the container process restarts or the container is recreated.
No automatic conversation expiration is implemented. Conversations remain
active until explicitly closed or deleted.

This is a single-user local deployment. The actor identity is configured by
`SMART_WASTE_ACTOR_ID` on the backend and is never accepted from request data.
The API is bound to localhost by the default Compose port mapping. Do not
expose it to an untrusted network without adding real authentication and
transport security.
