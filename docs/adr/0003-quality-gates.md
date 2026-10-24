# ADR-0003: Automated Quality Gates

- Status: Accepted
- Date: 2026-10-24

## Decision

Introduce quality checks incrementally:
1. Import and syntax checks
2. Unit tests
3. Integration tests
4. Linting
5. End-to-end tests when the full pipeline exists
6. GitHub Actions for repeatable checks on pushes and pull requests

A change is not considered verified until its relevant checks have run
and their actual results have been reviewed.

## Initial CI

The first workflow uses Python 3.11, installs the development package,
runs Ruff, and runs the unit test suite. Its first successful run is a
milestone to verify, not an assumed outcome.
