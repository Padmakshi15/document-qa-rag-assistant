"""Embeddings: turn text into numbers a computer can compare.

Two providers, chosen honestly at runtime:

* **OpenAI** (``OPENAI_API_KEY`` set *and* the ``openai`` package
  installed): ``text-embedding-3-small`` embeddings.
* **Local hash embedding** (default, offline): each word is hashed into one
  of 384 buckets and counted. Words that appear in both a question and a
  chunk push their vectors in the same direction, so cosine similarity
  still finds the relevant chunk. It is a deliberately simple fallback —
  it has no semantic knowledge (it cannot tell that "car" and "automobile"
  mean the same thing) — and the app labels it as such wherever it is used.
  It exists so the whole pipeline runs and can be learned offline, with
  no key and no network.
"""

from __future__ import annotations

import hashlib
import math
import os
import re

from config import LOCAL_EMBEDDING_DIM, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common words carry almost no topic signal and would make unrelated texts
# look similar, so the local fallback ignores them.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "in", "of", "on", "to", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "there", "here", "what", "which", "who", "whom",
    "how", "why", "can", "could", "should", "would", "will", "shall",
    "do", "does", "did", "have", "has", "had", "not", "no", "yes",
    "please", "tell", "me", "about", "your", "its", "our",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


def hash_embed(text: str, dim: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    """Deterministic bag-of-words embedding: hash each token to a bucket.

    The vector is L2-normalised, so cosine similarity is a plain dot
    product. Identical wording scores 1.0; unrelated wording scores ~0.
    """
    vec = [0.0] * dim
    for token in _tokens(text):
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two (already normalised) vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingProvider:
    """Common interface: ``name`` is shown in the UI so the mode is honest."""

    name: str = "base"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Offline fallback — no key, no network, no semantic knowledge."""

    name = "local-hash (offline fallback)"

    def __init__(self, dim: int = LOCAL_EMBEDDING_DIM):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embed(t, self.dim) for t in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings, used only when a key is configured."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = OPENAI_EMBEDDING_MODEL):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("OpenAI embeddings need: pip install openai") from exc
        key = api_key or OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [list(item.embedding) for item in response.data]


def get_embedding_provider() -> EmbeddingProvider:
    """OpenAI when a key is set and the package is available, else local.

    Returns the local hash provider (and never raises) when OpenAI is
    unavailable, so the app always runs.
    """
    if OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingProvider()
        except Exception:
            pass  # fall through to the offline provider
    return LocalHashEmbeddingProvider()
