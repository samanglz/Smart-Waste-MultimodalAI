# Query Rewriting Prompt Contract — v1

Status: Initial implementation
Prompt version: `query-rewriting-v1`

## Objective

Rewrite a natural-language user query to improve retrieval clarity while
preserving its complete information need.

## Required behavior

- Preserve every distinct sub-question.
- Preserve entities, material names, constraints, conditions and safety concerns.
- Do not add unsupported facts, assumptions, locations or answers.
- Avoid keyword stuffing.
- Use only the original query as input.
- Do not use target chunk IDs, target chunk text, relevance labels or ground truth.
- Return structured JSON containing `rewritten_query` and `preserved_aspects`.

## Important limitation

A syntactically valid response and a list of preserved aspects do not prove
semantic equivalence or better retrieval. These are separate evaluation tasks.

## Runtime policy

The natural query remains the default. This prompt contract does not itself
activate rewriting or define a selective-fallback threshold.

## Evaluation policy

Evaluate Natural-only, Rewrite-always and Selective Fallback separately using
a frozen evaluation set. Report MRR, Coverage@5, Coverage@10, Coverage@20,
latency, cost when available, and regressions.

Ground-truth chunk information is permitted for offline evaluation only.
It must not be supplied to the rewriter or used as a runtime selection signal.
