"""Persistence and validation for FAISS indexes and chunk metadata."""

import json
from pathlib import Path

from smart_waste_multimodal.indexing.faiss_index import FaissIndex


class IndexStore:
    """Persist a FAISS index and its corresponding chunk metadata."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, index: FaissIndex, chunks: list[dict[str, str]]) -> None:
        """Save the vector index and chunk metadata."""
        if index.size != len(chunks):
            raise ValueError("Index size must match the number of chunks")
        if not chunks:
            raise ValueError("chunks must not be empty")

        required_keys = {"chunk_id", "source", "text"}

        for position, chunk in enumerate(chunks):
            if not required_keys.issubset(chunk):
                raise ValueError(
                    f"Chunk at position {position} is missing required keys"
                )
            if not all(
                isinstance(chunk[key], str) and chunk[key].strip()
                for key in required_keys
            ):
                raise ValueError(
                    f"Chunk at position {position} has invalid metadata"
                )

        self.directory.mkdir(parents=True, exist_ok=True)
        index.save(self.directory / "index.faiss")

        metadata_path = self.directory / "chunks.json"
        metadata_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> tuple[FaissIndex, list[dict[str, str]]]:
        """Load the index and its corresponding chunk metadata."""
        index_path = self.directory / "index.faiss"
        metadata_path = self.directory / "chunks.json"

        if not index_path.is_file():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = FaissIndex.load(index_path)

        try:
            chunks = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid chunk metadata JSON") from exc

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Chunk metadata must be a non-empty list")

        required_keys = {"chunk_id", "source", "text"}

        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Chunk at position {position} must be an object"
                )
            if not required_keys.issubset(chunk):
                raise ValueError(
                    f"Chunk at position {position} is missing required keys"
                )
            if not all(
                isinstance(chunk[key], str) and chunk[key].strip()
                for key in required_keys
            ):
                raise ValueError(
                    f"Chunk at position {position} has invalid metadata"
                )

        if index.size != len(chunks):
            raise ValueError(
                "Index size does not match the number of chunks"
            )

        return index, chunks
