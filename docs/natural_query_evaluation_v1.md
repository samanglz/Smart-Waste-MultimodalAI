# Natural Query Evaluation v1

## Purpose

This dataset records natural-language queries before generating rewritten
queries or running a new retrieval baseline. It is intended to compare
retrieval behavior, not to prove that rewriting is universally beneficial.

## Scope

- 28 queries across 7 recycling topics.
- Four query styles per topic: direct, indirect, real-world scenario,
  and multi-part.
- English query text, matching the language of the existing retrieval corpus.

## Provenance and limitations

The queries were authored with awareness of the recycling domain. They were
not derived from target chunk IDs or retrieval rankings, but the set is not
blind or independently authored. Relevance labels must be checked against
the source passages before final metrics are treated as reliable.

## Leakage controls

- Do not add rewrites or target chunk IDs to the frozen query file.
- Keep target relevance labels in a separate evaluation file.
- Do not pass labels, target IDs, target text, or retrieval rankings to the
  runtime query rewriter.
- Save each rewriting strategy's output and results separately.
- Do not change the query set after observing its retrieval results. If a
  correction is necessary, create a new dataset version and document why.

## Interpretation

A comparison on this set is exploratory. Because the set is small and
domain-aware, it cannot by itself establish generalization to unseen users
or domains.
