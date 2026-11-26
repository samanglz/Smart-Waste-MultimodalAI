# Smart Waste Multimodal AI

An evidence-based multimodal system for waste detection and recycling knowledge retrieval. It combines YOLO-based image detection, hybrid information retrieval, evidence validation, and optional LLM-based answer generation in a modular pipeline exposed through a FastAPI service.

The system supports three primary workflows: **Standalone Detection**, **Standalone Knowledge Retrieval**, and **Detection-Based Retrieval**. Answer generation is an optional stage that can use validated evidence when the request and final decision permit it.

## Core Workflows

### 1. Standalone Detection

Detects waste materials in an input image without running knowledge retrieval.

- Accepts an image.
- Applies image preprocessing.
- Runs ONNX-based object detection.
- Returns detected material classes, confidence scores, and bounding boxes.

This workflow does not require a retrieval query or a knowledge-generation step.

### 2. Standalone Knowledge Retrieval

Retrieves recycling-related information directly from a user's question, without running image detection.

- Accepts a natural-language question.
- Uses the question as the retrieval query.
- Can optionally use API-assisted query rewriting to improve the retrieval query.
- Retrieves relevant knowledge using dense and lexical search.
- Combines rankings with Reciprocal Rank Fusion (RRF).
- Passes retrieved evidence through the evidence and final-decision stages.

This workflow does not require an image.

### 3. Detection-Based Retrieval

Uses image detection results to retrieve recycling knowledge about the detected materials.

- Accepts an image.
- Runs the vision pipeline.
- Extracts the detected material classes.
- Builds a retrieval query from those classes.
- Retrieves supporting knowledge using hybrid retrieval.
- Validates evidence and applies the final decision process.

For example, an image containing plastic and glass objects can produce a retrieval query based on the detected classes.

This workflow connects visual recognition to knowledge retrieval without requiring the user to manually provide the material names.

Detection-derived queries bypass question-based query rewriting. Retrieval is not executed in this workflow if Vision produces no detections.

## Query Rewriting

The question-based retrieval pipeline supports optional API-assisted query rewriting.

When a query rewriting service is configured, it can use a language-model API to improve the original question for knowledge retrieval. The purpose is to produce a retrieval-oriented query while preserving the user's original question as the source of truth.

The rewriting process is selective rather than mandatory:

- The original question is used directly when rewriting is not configured or is bypassed.
- A rewritten query can be used when the rewriting service returns a valid result.
- If rewriting fails, the system can fall back to the original question.
- The rewriting outcome is recorded through a query rewrite trace.

Query rewriting applies to question-based retrieval. Detection-derived queries bypass this service because they are constructed directly from the detected material classes.

## Optional Answer Generation

The system can optionally generate a natural-language answer using the validated evidence produced by the pipeline.

Generation is not required for the three core workflows. It is executed only when:

- Answer generation is enabled in the request.
- Context and evidence validation do not block execution.
- The final decision is `DIRECT`.
- A final evidence context is available.
- A Gemini API key is configured.

The generation stage uses the final evidence context rather than bypassing the evidence-based decision process.

If generation is requested but the required Gemini API key is unavailable, the API rejects the request.

## Architecture Overview

```mermaid
flowchart TD
    A[API Request] --> B{Requested Workflow}

    B -->|Standalone Detection| C[Image Preprocessing]
    C --> D[ONNX Vision Inference]
    D --> E[Detection Results]

    B -->|Standalone Knowledge Retrieval| F[User Question]
    F --> G[Question Context Analysis]
    G --> H[Optional API-Assisted Query Rewriting]

    B -->|Detection-Based Retrieval| I[Image Preprocessing]
    I --> J[ONNX Vision Inference]
    J --> K[Detected Material Classes]
    K --> L[Detection-Derived Query]

    H --> M[Hybrid Retrieval]
    L --> M

    M --> N[FAISS Dense Retrieval]
    M --> O[BM25 Lexical Retrieval]
    N --> P[Reciprocal Rank Fusion]
    O --> P

    P --> Q[Retrieved Evidence]
    Q --> R[Evidence Validation]
    R --> S[Final Decision]

    E --> T[Pipeline Response]
    S --> T

    S --> U{DIRECT and Generation Enabled?}
    U -->|Yes| V[Optional Gemini Generation]
    V --> W[Generated Answer]
    W --> T
```

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI, Uvicorn |
| Object Detection | YOLO-compatible ONNX model |
| Inference | ONNX Runtime |
| Image Processing | OpenCV, NumPy, Pillow |
| Text Embeddings | Sentence Transformers |
| Embedding Model | `intfloat/multilingual-e5-small` |
| Dense Retrieval | FAISS |
| Lexical Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| LLM API Integration | Optional API-assisted query rewriting and Gemini-based answer generation |
| Validation | Pydantic |
| Testing | pytest |
| Code Quality | Ruff |
| Deployment | Docker, Docker Compose |

## Retrieval Pipeline

The knowledge retrieval component combines two complementary retrieval methods:

- **Dense retrieval:** multilingual text embeddings generated with `intfloat/multilingual-e5-small` and searched through FAISS.
- **Lexical retrieval:** BM25-based keyword matching.

The two ranked result lists are combined using Reciprocal Rank Fusion.

The E5 encoder uses the expected prefixes:

- `query:` for search queries.
- `passage:` for indexed documents.

Embeddings are normalized before indexing and search.

The retrieval adapter supports both question-based queries and queries derived from detected material classes. Detection-derived queries bypass question-based query rewriting.

## Evidence-Based Decision Process

Retrieved and vision-derived information is handled through explicit evidence contracts and validation stages.

The final decision process supports three actions:

- `DIRECT` — sufficient validated evidence is available for a direct response.
- `INSUFFICIENT_EVIDENCE` — the available evidence does not support a direct response.
- `HUMAN_REVIEW` — additional clarification or human review is required.

Optional answer generation is gated by the final decision and receives the final evidence context.

## API

The application exposes a FastAPI service.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/conversations` | Create a conversation |
| POST | `/api/v1/conversations/{conversation_id}/analyze` | Analyze a request |
| POST | `/api/v1/conversations/{conversation_id}/close` | Close a conversation |
| DELETE | `/api/v1/conversations/{conversation_id}` | Delete a conversation |

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### Analyze Request

The analyze endpoint accepts multipart form data:

| Field | Description |
|---|---|
| `request_id` | Request identifier |
| `task_type` | `detection` or `knowledge_question` |
| `question` | Optional natural-language question |
| `locale` | Response locale: `en` or `fa` |
| `options` | JSON-encoded request options |
| `image` | Optional JPEG or PNG image, up to 15 MiB |

The request options control which pipeline stages are executed.

For example, the deployment configuration documents these two independent modes:

**Standalone Detection**

```json
{
  "run_vision": true,
  "run_retrieval": false,
  "generate_answer": false,
  "retrieval_query_source": "question"
}
```

**Standalone Knowledge Retrieval**

```json
{
  "run_vision": false,
  "run_retrieval": true,
  "generate_answer": false,
  "retrieval_query_source": "question"
}
```

Detection-Based Retrieval uses both Vision and Retrieval, with the retrieval query derived from the vision results.

## Getting Started

### Requirements

- Python 3.11 or newer for local development.
- Docker Engine and Docker Compose for container deployment.
- The ONNX model at `exports/best.onnx`.
- The retrieval index and chunk data under `data/retrieval_index/v4_expanded/`.

### Local Installation

Clone the repository:

```bash
git clone https://github.com/samanglz/Smart-Waste-MultimodalAI.git
cd Smart-Waste-MultimodalAI
```

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install development and optional generation dependencies:

```bash
python -m pip install -e ".[dev,generation]"
```

### Environment Configuration

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Available environment variables:

| Variable | Purpose |
|---|---|
| `SMART_WASTE_ACTOR_ID` | Backend-configured actor identity |
| `GEMINI_API_KEY` | Optional Gemini API key |
| `GOOGLE_API_KEY` | Alternative Google API key |
| `GEMINI_MODEL` | Optional Gemini model override |

The application can start without a Gemini key. However, requests that require Gemini-backed functionality need the appropriate API configuration.

### Docker Deployment

Configure `.env`, then build and start the application:

```bash
docker compose up --build -d
```

Check the service:

```bash
curl http://127.0.0.1:8000/health
```

Open the API documentation:

`http://127.0.0.1:8000/docs`

The Compose configuration binds the service to `127.0.0.1:8000` by default.

For further deployment details, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Testing and Code Quality

Run the test suite:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

The repository includes unit and integration tests covering pipeline orchestration, adapters, evidence handling, retrieval, vision processing, API workflows, and failure scenarios.

Test results depend on the current code and environment; consult the latest test run rather than assuming a historical result represents the current state.

## Project Structure

```text
.
├── data/
│   └── retrieval_index/
├── docs/
├── experiments/
│   └── evaluation/
├── exports/
├── src/
│   └── smart_waste_multimodal/
│       ├── adapters/
│       ├── api/
│       ├── contracts/
│       ├── embeddings/
│       ├── evidence/
│       ├── generation/
│       ├── indexing/
│       ├── orchestration/
│       ├── retrieval/
│       └── vision/
├── tests/
│   ├── integration/
│   └── unit/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── DEPLOYMENT.md
└── pyproject.toml
```

## Runtime and Security Notes

- Conversation and related runtime state are stored in memory.
- Runtime state is lost when the application process or container restarts.
- Conversations remain active until explicitly closed or deleted; there is no automatic expiration.
- The default Docker Compose configuration binds to localhost.
- The configured actor identity is supplied by the backend environment, not by the client request.
- The default deployment is intended for local, single-user use. Do not expose the service to an untrusted network without adding appropriate authentication, authorization, and deployment safeguards.

## Project Scope

The project brings together three independently selectable capabilities—visual detection, knowledge retrieval, and detection-based retrieval—under a shared orchestration and evidence-validation architecture.

API-assisted query rewriting can improve question-based retrieval, while answer generation remains an optional downstream capability rather than a prerequisite for detection or retrieval.

The repository also contains evaluation experiments and supporting documentation for the system's development.
