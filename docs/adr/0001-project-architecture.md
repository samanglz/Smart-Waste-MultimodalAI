# ADR-0001: Independent, Modular Project Architecture

- Status: Accepted
- Date: 2026-10-24

## Context

The system will integrate an existing computer-vision project and an
existing recycling-knowledge RAG project. Both repositories must remain
independent and must not be modified as a side effect of integration.

## Decision

Create a separate repository named `smart-waste-multimodal-ai`.

Use explicit contracts and adapters to connect external capabilities.
Keep vision, retrieval, orchestration, generation, API, evaluation,
configuration, and observability responsibilities separated.

Implement modules only when needed by a verified milestone.

## Consequences

- Existing repositories remain independent.
- Components can be tested and replaced separately.
- Contract changes must be reviewed for compatibility.
- Integration must use documented interfaces, not assumptions about
  another repository's internal implementation.
