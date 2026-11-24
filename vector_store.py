"""Vector store: keep chunk embeddings and find the closest ones.

Two backends behind one interface:

* **ChromaDB** (``use_chroma=True`` and the ``chromadb`` package installed):
  a persistent collection in ``./chroma_db`` — chunks survive restarts.
* **In-memory** (default, and the automatic fallback when ChromaDB is not
  installed): chunks live in a list and search is a cosine-similarity
  ranking. Nothing is persisted, which is fine for learning, tests and
  the evaluation script.

The Streamlit app tries ChromaDB first and falls back silently to memory;
``qa.py`` and the tests use the in-memory store so they run anywhere.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from config import CHROMA_DB_DIR, COLLECTION_NAME
from embeddings import cosine_similarity
from ingest import DocumentChunk


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float  # cosine similarity, 0..1 (higher = more relevant)


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_DB_DIR, use_chroma: bool = False):
        self.persist_dir = persist_dir
        self.use_chroma = use_chroma
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []
        self._collection = None
        self.backend = "in-memory"
        if use_chroma:
            self._try_init_chroma()

    def _try_init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError:
            return  # ChromaDB not installed -> stay in-memory
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
            self.backend = "chromadb"
        except Exception:
            self._collection = None
            self.backend = "in-memory"

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks together with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)
        if self._collection is not None:
            self._collection.add(
                ids=[f"{c.source}-{c.chunk_index}-{uuid.uuid4().hex[:8]}" for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[{"source": c.source, "page": c.page,
                            "chunk_index": c.chunk_index} for c in chunks],
            )

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks, best first."""
        if not self._chunks:
            return []
        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_embedding, emb))
            for chunk, emb in zip(self._chunks, self._embeddings)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(top_k, 1)]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()
        if self._collection is not None:
            try:
                self._collection.delete(where={})
            except Exception:
                pass
