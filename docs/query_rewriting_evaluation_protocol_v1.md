# Query Rewriting Evaluation Protocol v1

## 1. Purpose and scope

This protocol defines how to evaluate query rewriting in
`smart-waste-multimodal-ai`.

It separates two questions:

1. **Rewrite quality:** Does the rewritten query preserve the user's intent
   and all important details?
2. **Retrieval impact:** Does using the rewritten query retrieve evidence
   that is more relevant to the original question?

Valid JSON or a successful schema check is necessary for an output to be
usable, but it is not evidence that the rewrite is semantically good or
improves retrieval.

This protocol is an exploratory evaluation plan, not a claim of production
readiness or statistical significance.

## 2. Frozen evaluation dataset

- Dataset: `experiments/evaluation/natural_queries_eval_v2_frozen.json`
- Size: 28 natural-language queries
- Difficulty: 12 easy, 10 medium, 6 challenging
- Dataset SHA-256 is recorded in the accompanying protocol manifest.

The frozen dataset must not be edited in place. If a substantive defect is
discovered, record it and create a new version rather than silently changing
this benchmark.

The fields describing evidence and required aspects are evaluation material.
They must not be passed to the rewrite model at runtime.

## 3. Experimental conditions

Run the following conditions against the same frozen query set.

### A. Natural-query baseline

Send the original user query directly to the retriever.

### B. Rewrite-always

Send every original query to the rewrite model. Validate its output, then
send the rewritten query to the same retriever. If rewriting or validation
fails, record the failure and use the original query as a safe fallback.
Report fallback frequency separately.

### C. Selective rewriting (later phase)

Use a documented decision policy to rewrite only selected queries and retain
the original for the rest. Do not tune the policy and report its performance
on the same small set as if it were an unbiased final test. Treat threshold
exploration as exploratory; confirm a chosen policy on a separate set before
making production claims.

For fair comparisons, all conditions must use the same corpus/index snapshot,
retriever configuration, retrieval depth k, filtering rules, and evaluation
judgments. Record any unavoidable differences.

## 4. Runtime information boundaries

The rewrite model may receive:
- The original natural-language query.
- The versioned rewrite prompt and output schema.
- Any explicitly documented runtime context permitted by the design.

The rewrite model must not receive:
- Ground-truth labels or relevant chunk IDs.
- Evaluation-only `must_preserve` annotations.
- Candidate-chunk review decisions.
- Retrieval results from the evaluation run, unless a separately specified
  experiment explicitly studies retrieval-aware rewriting.

The frozen dataset and its evaluation annotations are for offline evaluation,
not hidden runtime inputs.

## 5. Rewrite-quality assessment

Assess each rewrite against the original query. Use a 0–2 score for each
dimension:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Intent preservation | Intent changed | Partly preserved or ambiguous | Same intent |
| Constraint and subquestion preservation | Important details omitted | Minor omission or weakening | All important details retained |
| Unsupported additions | Material invented details | Potentially misleading addition | No unsupported addition |
| Retrieval clarity | Less clear or misleading | Little improvement | Clearer without changing meaning |
| Readability | Confusing or malformed | Understandable with issues | Clear and natural |

Maximum: 10 points per query.

These scores are ordinal rubric scores, not calibrated probabilities. Record
the individual dimension scores and short justifications; do not report only
the total. A critical intent change, material unsupported claim, or omission
of a required constraint must be flagged even if the total score is high.

Use the frozen query's evaluation annotations for offline human review only.
Do not expose them to the rewrite model. For a small benchmark, manually review
all 28 outputs. If feasible, have a second reviewer independently assess a
subset and resolve disagreements explicitly.

## 6. Structural and safety checks

For each model response, record:
- Whether parsing succeeded.
- Whether the output schema passed validation.
- Whether required fields are present and non-empty.
- Whether the response was rejected, retried, or fell back to the original.
- Whether any unsupported entity, location, material, or instruction was added.

Structural validity and semantic quality must be reported separately.

## 7. Retrieval evaluation

Retrieval effectiveness cannot be scored until relevance judgments are
available and reviewed. The presence of an evidence reference in the dataset
does not automatically make every chunk retrieved from that source relevant
to every query.

Create offline relevance judgments against the actual indexed corpus.
Preserve the distinction between:
- Evidence that supports the answer.
- Chunks that are merely topically related.
- Chunks that are irrelevant or contradictory.

For a fixed retrieval depth k, report:

- **HitRate@k:** fraction of queries with at least one relevant result in the
  top k.
- **MRR@k:** mean reciprocal rank of the first relevant result, using zero
  when none appears in the top k.
- **Relevant-result coverage:** number of judged relevant evidence items
  retrieved in the top k divided by the number of judged relevant evidence
  items for that query, where the denominator is non-zero and judgments
  define a meaningful complete set.
- **No-evidence cases:** report separately; do not automatically score them
  as retrieval failures if the question is genuinely unanswerable from the
  corpus.

Use the same relevance judgments for natural and rewritten queries. Report
per-query differences as well as aggregate results, including regressions.
Do not claim improvement merely because a rewritten query retrieves more
chunks or uses more specific words.

If judgments are incomplete, label the affected metrics as provisional and
do not interpret unjudged results as definitely irrelevant.

## 8. Latency, cost, and failures

Record separately:
- Rewrite latency.
- Retrieval latency.
- End-to-end latency.
- Token usage and estimated model cost when available.
- Retry, parsing, validation, and fallback rates.

Compare rewrite overhead with any retrieval-quality gain. If provider usage
or cost data are unavailable, mark the field unavailable instead of estimating
it without evidence.

## 9. Reporting

The evaluation report should include:
1. Dataset ID and SHA-256.
2. Prompt version, model/provider identifier, model parameters, and run date.
3. Retriever/index snapshot and retrieval configuration.
4. Number of successful, failed, and fallback rewrites.
5. Rewrite-quality scores by dimension and difficulty.
6. Retrieval metrics for each condition, once judgments are available.
7. Per-query gains and regressions.
8. Latency and cost.
9. Limitations, unjudged evidence, and any manual review decisions.

Keep raw model responses and normalized results as separate artifacts. Each
run should have a unique ID and a manifest recording the inputs and versions.

## 10. Decision rules and limitations

- Do not select a policy based only on total rewrite-quality score.
- Do not accept a material intent change just because retrieval metrics rise.
- Do not claim production readiness from this 28-query exploratory benchmark.
- Do not tune and validate a selective-rewrite threshold on the same queries
  and then describe the result as an independent test.
- Preserve prior experiment files; use new versioned output names.
- Never modify the legacy `waste-detection-rag` project as part of this
  protocol-writing step.

## 11. Current status

This document defines the evaluation method only. It does not connect an LLM,
assign ground truth, run retrieval, change the frozen dataset, or establish
that rewriting improves performance.
