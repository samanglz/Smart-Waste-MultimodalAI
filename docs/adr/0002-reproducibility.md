# ADR-0002: Reproducible Experiments and Provenance

- Status: Accepted
- Date: 2026-10-24

## Decision

Track the configuration and provenance of meaningful experiments.

Record, where applicable:
- Git commit and experiment identifier
- Python and dependency versions
- Model identifier, version, and file hash where practical
- Retrieval index and embedding model versions
- Chunking and retrieval configuration
- Random seeds
- Evaluation metrics and runtime
- Input data provenance and artifact locations

A JSON manifest is the initial registry format. Adopt a larger experiment
tracking system only when a concrete requirement justifies it.

## Limitations

A fixed random seed does not guarantee bit-for-bit reproducibility across
different hardware, libraries, or nondeterministic operations. Record
relevant environment details and limitations.
