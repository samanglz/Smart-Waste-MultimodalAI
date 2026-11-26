# ADR 0004: Image Store Architecture

- **Status:** Accepted
- **Date:** 2026-11-21

## Context

The multimodal pipeline accepts images through `ImageArtifact`, which
intentionally exposes metadata and an artifact identifier rather than
filesystem paths or arbitrary URLs.

The `ImageLoader` protocol resolves an `ImageArtifact` into an OpenCV-
compatible BGR image. `ONNXVisionAdapter` consumes this interface and
does not own image storage or image-file access.

The project needs a storage mechanism suitable for its initial Google
Colab notebook workflow. It must support image reuse across conversation
turns during a runtime session without coupling image storage to the
Evidence Registry.

## Decision

The first implementation will use an independent in-memory Image Store.

The store will retain the original encoded image bytes and resolve them
using a unique `artifact_id`.

### Storage and identity

- Store original encoded image bytes rather than preprocessed tensors.
- Use `artifact_id` as the lookup key.
- Reject duplicate artifact identifiers; never silently overwrite an
  existing image.
- Keep the Image Store independent from the Evidence Registry.

### Integrity and validation

- Compute SHA-256 when an image is registered.
- Verify image integrity when retrieving it.
- Accept only supported JPEG and PNG media types.
- Validate that the actual image content matches the declared media type.
- Validate image dimensions against the supplied artifact metadata.
- Apply explicit limits to encoded file size and decoded image dimensions
  before expensive processing.

### Loading and inference

- The ImageLoader implementation will retrieve bytes from the Image Store.
- Image decoding will happen in the ImageLoader.
- The loader will return a non-empty NumPy array with `uint8` dtype and
  BGR channel order, shaped as `(height, width, 3)`.
- Preprocessing, ONNX inference, and postprocessing remain responsibilities
  of the existing vision pipeline.

### Lifecycle

- Image retention and cleanup are managed independently from Evidence
  Registry retention.
- Removing an Evidence Registry entry must not implicitly delete the
  corresponding stored image.
- Removing an image must not implicitly remove conversation evidence.
- Images may be reused by later turns while they remain available under
  the configured retention policy.
- The initial in-memory implementation does not survive a Colab runtime
  restart or process termination.

## Consequences

### Positive

- The initial notebook workflow does not depend on persistent filesystem
  paths.
- Image storage remains decoupled from pipeline contracts and evidence
  management.
- Original image bytes can be decoded consistently whenever needed.
- A future storage backend can be introduced without exposing storage
  paths through `ImageArtifact` or changing the vision-stage interface.

### Trade-offs

- Images are lost when the runtime restarts.
- Memory usage must be bounded.
- The implementation must define explicit deletion and retention behavior.
- The in-memory store is scoped to the lifetime of its owning runtime
  process and is not a distributed storage solution.

## Implementation Scope

This ADR records the architectural decision only.

It does not implement the Image Store, ImageLoader, notebook upload
handler, retention policy, or persistent storage backend. Those changes
must be implemented and validated in separately approved stages.
