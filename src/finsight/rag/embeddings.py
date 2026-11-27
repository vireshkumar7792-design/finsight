"""Local embeddings (no API, no cost).

WHAT AN EMBEDDING IS: a model maps a text to a fixed-length vector (384 numbers here). It is
trained so that texts with similar meaning land close together (high cosine similarity), even
with different words: "supply constraints" ends up near "shortages of components". Retrieval
= embed the question, then find the stored chunk vectors nearest to it.

MODEL: BAAI/bge-small-en-v1.5 rather than all-MiniLM-L6-v2. Both are small (CPU-friendly),
384-dim and free, but MiniLM reads only 256 tokens (our ~300-token chunks would be cut off)
while bge-small reads 512 and is trained specifically for retrieval (question -> passage).
bge expects a short instruction prefix on QUERIES only, so queries and passages are
embedded slightly differently.

Tests use tests/fakes.py::HashEmbedder, which has the same two methods.
"""

from __future__ import annotations

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """sentence-transformers model, loaded lazily (first use downloads ~130 MB once, then cached)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import: only when needed

            self._model = SentenceTransformer(self.name, device="cpu")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 200)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True)[0].tolist()
