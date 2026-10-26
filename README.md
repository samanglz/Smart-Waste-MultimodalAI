# Smart Waste Multimodal AI

A modular, contract-driven multimodal AI pipeline designed to combine
waste-object detection with evidence-based recycling knowledge retrieval
and grounded answer generation.

## Project goals

- Integrate Vision and RAG components through explicit, versioned contracts.
- Keep stages independently testable and dependency-injected.
- Handle stage failures without unnecessarily stopping independent stages.
- Make pipeline outcomes, errors, and warnings explicit.
- Support reproducible experiments and traceable artifacts.
- Enforce automated quality checks from the beginning.

## Current architecture

The Python package lives under `src/smart_waste_multimodal/`.

- `contracts/`: validated request, response, Vision, Retrieval, and Generation
  data models.
- `adapters/`: Protocol interfaces for injecting the three pipeline stages.
- `orchestration/`: dependency-injected pipeline orchestration, stage-result
  type checks, safe failure handling, and conditional stage dependencies.
- `observability/`: explicit logging configuration that preserves existing
  root logger handlers.
- `config/`, `evaluation/`, `generation/`, `retrieval/`, `vision/`, and `api/`:
  package areas reserved for their corresponding capabilities as implementation
  progresses.

## Pipeline behavior

- At least one of Vision, Retrieval, or Generation must be requested.
- Retrieval can use an independent question or Vision detections as its query
  source.
- Question-based Retrieval can run concurrently with Vision when both are
  requested.
- Detection-based Retrieval depends on successful Vision output containing
  detections.
- A stage failure is represented as a structured error; independent stages
  continue where possible.
- Warnings can produce a `partial_success` response even when useful stage
  outputs are available.
- Stage outputs are checked against their expected contract types.

## Development and testing

Python 3.11 or newer is required.

Install the package and runtime dependencies:

```bash
python -m pip install -e .
```

Install development dependencies as well:

```bash
python -m pip install -e ".[dev]"
```

Run quality checks and tests:

```bash
ruff check .
python -m pytest tests/unit tests/integration -q
```

GitHub Actions runs linting and both unit and integration tests.

## Development status

The contract models, adapter interfaces, logging configuration, orchestrator,
and mock-based unit and integration tests are implemented.

Real model and service integrations remain future work. In particular, this
repository does not yet claim to run the production YOLO/ONNX detector or the
existing recycling RAG system end to end. Those integrations will be added
incrementally behind the defined interfaces and validated with dedicated tests.

## Related projects

This repository is independent of the existing `smart-waste-detection` and
`waste-detection-rag` repositories. Work here should not modify those projects.
