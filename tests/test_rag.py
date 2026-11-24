"""Tests for the Document Q&A Assistant (RAG).

Run:  python -m pytest
All tests run offline with the local hash embedding fallback — no API
key, no network, no ChromaDB needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from embeddings import LocalHashEmbeddingProvider, cosine_similarity, hash_embed  # noqa: E402
from ingest import chunk_text, load_document  # noqa: E402
from qa import FALLBACK_MESSAGE, RAGAssistant  # noqa: E402
from vector_store import VectorStore  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def make_assistant() -> RAGAssistant:
    return RAGAssistant(provider=LocalHashEmbeddingProvider(), store=VectorStore())


def test_config_defaults():
    """Tested defaults kept after the chunk-size / overlap / top-k tuning."""
    assert config.CHUNK_SIZE == 1000
    assert config.CHUNK_OVERLAP == 200
    assert config.TOP_K == 4


def test_chunking_overlap():
    """The tail of one chunk repeats at the start of the next."""
    text = "abcdefghij" * 60  # 600 chars of repeating pattern
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20, source="t.txt")
    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        # Last 20 chars of the first chunk appear at the start of the next.
        assert first.text[-20:] in second.text or second.text.startswith(first.text[-20:])
    # No chunk exceeds the configured size.
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunking_short_text_is_one_chunk():
    assert len(chunk_text("hello world", chunk_size=1000, chunk_overlap=200)) == 1
    assert chunk_text("", chunk_size=1000, chunk_overlap=200) == []


def test_retrieval_returns_relevant_chunk():
    """A question about retention retrieves the policy chunk, not the guide."""
    assistant = make_assistant()
    assistant.ingest_file(os.path.join(SAMPLES, "sample_policy.txt"))
    assistant.ingest_file(os.path.join(SAMPLES, "sample_guide.txt"))
    results = assistant.retrieve("How long are uploaded documents retained?")
    assert results, "expected at least one retrieved chunk"
    assert "90 days" in results[0].chunk.text
    assert results[0].chunk.source == "sample_policy.txt"


def test_fallback_when_no_context():
    """An empty index answers with the fallback, never an invented answer."""
    assistant = make_assistant()
    answer = assistant.ask("What is the default chunk size?")
    assert not answer.grounded
    assert answer.text == FALLBACK_MESSAGE
    assert answer.sources == []


def test_fallback_when_unrelated_question():
    """A question with no supporting text gets the fallback even with docs indexed."""
    assistant = make_assistant()
    assistant.ingest_file(os.path.join(SAMPLES, "sample_guide.txt"))
    answer = assistant.ask("What is the refund policy for cancelled subscriptions?")
    assert not answer.grounded
    assert "couldn't find supporting information" in answer.text


def test_citation_includes_page_and_source():
    """A grounded answer cites the source filename and page number."""
    assistant = make_assistant()
    assistant.ingest_file(os.path.join(SAMPLES, "sample_policy.txt"))
    answer = assistant.ask("How long are uploaded documents retained?")
    assert answer.grounded
    assert answer.sources
    assert answer.sources[0].source == "sample_policy.txt"
    assert answer.sources[0].page >= 1
    assert "sample_policy.txt, p. 1" in answer.citations()
    assert "sample_policy.txt" in answer.text  # citation shown in the answer


def test_pdf_extraction_keeps_page_numbers():
    """The sample PDF extracts text and its chunks carry page 1."""
    chunks = load_document(os.path.join(SAMPLES, "sample_handbook.pdf"))
    assert chunks
    assert any("mentor" in c.text.lower() for c in chunks)
    assert all(c.page >= 1 for c in chunks)
    assert all(c.source == "sample_handbook.pdf" for c in chunks)


def test_hash_embedding_similarity_is_sensible():
    """Same wording scores ~1.0, unrelated wording scores near 0."""
    assert cosine_similarity(hash_embed("retention policy ninety days"),
                             hash_embed("retention policy ninety days")) > 0.99
    assert cosine_similarity(hash_embed("retention policy ninety days"),
                             hash_embed("quantum banana spaceship")) < 0.2
