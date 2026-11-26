# Image Store — Technical Design

- Project: `smart-waste-multimodal-ai`
- Document version: 1.0
- Date: 2026-11-21
- Status: Design approved
- Related decision: ADR 0004 — Image Store Architecture

## 1. Purpose and Scope

This document defines the Version 1 technical design for an independent
Image Store that registers, validates, temporarily stores, and retrieves
images for the project's vision pipeline.

Development environment:
- Google Colab with Google Drive.
- Final deployment: Docker on a local Host.

The Version 1 backend is in-memory in both environments. Images are not
persistent and are lost when the Colab runtime or Docker container restarts.

This component is independent of the Evidence Registry.

## 2. Architecture

```text
Notebook / Local Application
          |
          v
    ImageArtifact
          |
          v
      ImageStore
          |
          v
  InMemoryImageStore
          |
          | get_image()
          v
 InMemoryImageLoader
          |
          | verify + decode
          v
   BGR uint8 ndarray
          |
          v
  ONNXVisionAdapter
          |
          v
 Preprocessing / Inference
 / Postprocessing
```

Responsibilities:

- `ImageStore`: backend-independent storage contract.
- `InMemoryImageStore`: Version 1 in-memory implementation.
- `InMemoryImageLoader`: adapter that retrieves and decodes stored images.
- `ONNXVisionAdapter`: consumes the decoded image through the existing
  ImageLoader contract. It does not depend directly on ImageStore.

Image Store and Evidence Registry have independent storage and lifecycle
management. Their connection is through the existing image artifact
contract and the vision pipeline.

## 3. Resource Limits and Supported Formats

| Constraint | Version 1 limit |
|---|---:|
| Maximum encoded size per image | 15 MiB |
| Total encoded bytes in one Store instance | 1024 MiB |
| Maximum width | 8192 pixels |
| Maximum height | 8192 pixels |
| Maximum pixel count | 25,000,000 |
| Supported formats | JPEG and PNG |

Capacity accounting uses the original encoded image bytes. It does not
include decoded NumPy arrays or Python object overhead.

No automatic eviction or LRU policy is used. A registration that would
exceed the total capacity is rejected without modifying the Store.

## 4. ImageStore Contract

The backend-independent `ImageStore` Protocol exposes the following
operations:

| Operation | Result |
|---|---|
| `register(...)` | Registered `ImageArtifact` |
| `get_bytes(artifact_id)` | Original encoded `bytes` |
| `get_metadata(artifact_id)` | Independent `ImageArtifact` metadata |
| `get_image(artifact_id)` | Consistent snapshot of bytes and metadata |
| `contains(artifact_id)` | `bool` |
| `delete(artifact_id)` | `bool` |
| `clear()` | Number of deleted records |
| `total_bytes` | Total encoded bytes |
| `count` | Number of stored images |

`get_image` is the primary retrieval operation for the loader. It returns
the encoded bytes and corresponding metadata from one consistent snapshot.

Stored bytes are immutable. Bytes-like input is copied into a `bytes`
object before storage. Returned metadata must not expose mutable internal
state.

The internal record is conceptually:

    {
        artifact_id: {
            image_bytes: bytes,
            metadata: ImageArtifact
        }
    }

## 5. Registration Workflow

Registration follows this sequence:

1. Validate the input type and artifact identifier.
2. Convert the input into an independent immutable `bytes` object.
3. Enforce the 15 MiB per-image encoded-size limit.
4. Detect the actual JPEG or PNG content signature.
5. Verify that the declared media type matches the actual content.
6. Validate the image structure with Pillow.
7. Validate width, height, and total pixel count.
8. Perform a complete image decode to reject truncated or invalid data.
9. Calculate SHA-256 from the original encoded bytes.
10. Construct the complete `ImageArtifact` metadata.
11. Under the Store lock, check duplicate identifier and total capacity.
12. Atomically insert the bytes and metadata.
13. Return the registered metadata.

If any step fails, no partial record is retained.

Duplicate identifiers are rejected. Existing records are never silently
overwritten.

Validation and expensive decoding should happen outside the critical
section. Duplicate checking, capacity checking, and final insertion must
be atomic under the lock.

## 6. Concurrency and Lifecycle

The in-memory implementation uses `threading.RLock`.

- Registration's final checks and insertion are atomic.
- Duplicate identifier and capacity checks are performed under the lock.
- Retrieval snapshots, deletion, and clearing are synchronized.
- Expensive validation and initial decoding happen outside the lock.
- A failed registration must not change the Store's record count or
  encoded-byte accounting.

`delete` affects only the Image Store. `clear` affects only the Image Store.
Neither operation implicitly deletes or modifies Evidence Registry records.

## 7. ImageLoader Contract

Construction:

    InMemoryImageLoader(image_store)

Loading:

    load(artifact: ImageArtifact) -> np.ndarray

The loader performs these operations:

1. Retrieve a consistent bytes-and-metadata snapshot.
2. Compare the supplied artifact against the stored metadata.
3. Compare all relevant metadata fields:
   `artifact_id`, `media_type`, `sha256`, `width`, and `height`.
4. Reject any mismatch with `ImageArtifactMismatchError`.
5. Verify SHA-256 against the stored digest.
6. Decode the image using Pillow.
7. Convert RGB pixels to BGR.
8. Return a NumPy array with `uint8` dtype.
9. Validate the resulting shape: `(height, width, 3)`.

The loader does not perform model preprocessing, inference, or
postprocessing.

The ImageLoader output remains compatible with the existing vision
pipeline contract: a BGR `uint8` array.

## 8. Error Contract

| Error | Meaning |
|---|---|
| `ImageStoreError` | Base Image Store exception |
| `DuplicateImageArtifactError` | Artifact identifier already exists |
| `ImageNotFoundError` | Requested artifact does not exist |
| `InvalidImageError` | Invalid, corrupt, or incomplete image |
| `UnsupportedImageFormatError` | Unsupported image format |
| `ImageIntegrityError` | SHA-256 integrity verification failed |
| `ImageSizeLimitError` | Per-image size or dimension limit exceeded |
| `ImageStoreCapacityError` | Total Store capacity would be exceeded |
| `ImageArtifactMismatchError` | Supplied and stored metadata differ |

All specialized errors derive from `ImageStoreError`.

## 9. Proposed File Organization

    src/smart_waste_multimodal/
    └── vision/
        ├── image_loader.py
        ├── image_store.py
        ├── image_store_errors.py
        ├── in_memory_image_store.py
        ├── in_memory_image_loader.py
        ├── preprocessing.py
        ├── postprocessing.py
        └── onnx_adapter.py

Existing files are not to be modified unless a concrete integration
requirement is identified and separately approved.

## 10. Docker and Local Host Deployment

- The Store is created at application scope, not separately for every
  request.
- Requests handled by the same process share the same Store instance.
- Container restart destroys all stored images.
- Version 1 does not require a Docker volume for image storage.
- The implementation must not depend on Colab-specific paths.
- The 1024 MiB capacity applies independently to each Store instance.
- Multiple application worker processes have separate in-memory stores.
  Worker count and resulting memory consumption must be considered during
  Docker deployment design.

## 11. Required Verification Coverage

Implementation and tests must cover:

- Successful JPEG registration.
- Successful PNG registration.
- Rejection of content whose actual format does not match its declaration.
- Rejection of corrupt or truncated images.
- Rejection of unsupported formats.
- Enforcement of encoded-size and dimension limits.
- Duplicate identifier rejection without overwrite.
- Capacity rejection without eviction or partial insertion.
- Exact retrieval of original encoded bytes.
- SHA-256 integrity verification.
- Metadata mismatch detection.
- Correct BGR output, dtype, and dimensions.
- Correct synchronization under concurrent operations.
- Independent Image Store and Evidence Registry lifecycles.
- Correct deletion and clearing behavior.

## 12. Open Design Detail

The current artifact contract models one image rather than a sequence of
frames. Animated or multi-frame image handling therefore remains an
explicit implementation detail to settle before finalizing validation
behavior. It must not be silently assumed to be supported.

## 13. Implementation Sequence

1. Define Image Store exceptions and the `ImageStore` Protocol.
2. Implement `InMemoryImageStore`.
3. Implement `InMemoryImageLoader`.
4. Integrate the loader with the existing Vision Pipeline.
5. Run component-level tests.
6. Run end-to-end integration tests through the ONNX vision path.

Implementation, integration, and testing are separate approved stages.
This document records the approved design; it does not authorize changes
to executable code.
