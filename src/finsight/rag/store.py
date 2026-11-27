"""Persistent Chroma collection for 10-K chunks.

- We compute embeddings ourselves and pass vectors in (embedding_function=None): the embedder is
  a plain object we control and can fake in tests, and Chroma can never silently download its
  own default model.
- anonymized_telemetry=False: Chroma sends usage telemetry by default; we keep everything local.
- Cosine distance, matching our normalized embeddings. score = 1 - distance.
- The collection records which embedding model built it; querying with a different model would
  compare incompatible vectors, so that's refused.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

COLLECTION = "sec_filings"


class EmbeddingModelMismatchError(RuntimeError):
    pass


def open_collection(path: str | Path, embedder_name: str, name: str = COLLECTION):
    Path(path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": embedder_name},
    )
    built_with = (col.metadata or {}).get("embedding_model")
    if built_with and built_with != embedder_name:
        raise EmbeddingModelMismatchError(
            f"collection {name!r} was built with {built_with!r}, not {embedder_name!r}; re-ingest into a new path"
        )
    return col
