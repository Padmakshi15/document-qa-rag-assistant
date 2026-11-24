"""Question answering: retrieve, then answer only from the retrieved text.

Flow for one question (the R in RAG):

1. Embed the question with the same provider used for the documents.
2. Retrieve the TOP_K most similar chunks from the vector store.
3. If nothing is stored, or the best chunk is below MIN_SIMILARITY,
   return the fallback — the assistant says it does not know instead of
   inventing an answer.
4. Otherwise build a grounded prompt ("answer only from this context")
   and either call the LLM (OpenAI key set) or compose an extractive
   answer from the retrieved chunks (offline mode). Either way every
   answer carries its source citations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from config import MIN_SIMILARITY, OPENAI_API_KEY, OPENAI_CHAT_MODEL, TOP_K
from embeddings import EmbeddingProvider, get_embedding_provider
from ingest import DocumentChunk, load_document, load_text
from vector_store import SearchResult, VectorStore

FALLBACK_MESSAGE = (
    "I couldn't find supporting information in the uploaded documents "
    "to answer that question. Please try rephrasing it, or upload a "
    "document that covers the topic."
)

GROUNDED_PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: {fallback}

Context:
{context}

Question: {question}

Answer (with the source of each fact):"""


def _citations(sources: list[DocumentChunk]) -> list[str]:
    seen: list[str] = []
    for chunk in sources:
        cite = chunk.citation()
        if cite not in seen:
            seen.append(cite)
    return seen


@dataclass
class Answer:
    text: str
    sources: list[DocumentChunk] = field(default_factory=list)
    grounded: bool = False  # True when the answer comes from retrieved text
    mode: str = "offline-extractive"  # or "openai"

    def citations(self) -> list[str]:
        """Unique ``source, p. N`` citations, in retrieval order."""
        return _citations(self.sources)


class RAGAssistant:
    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ):
        self.provider = provider or get_embedding_provider()
        self.store = store or VectorStore(use_chroma=False)
        self.top_k = top_k
        self.min_similarity = min_similarity

    # -- Ingestion ---------------------------------------------------------
    def ingest_file(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        chunks = load_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return self._add(chunks)

    def ingest_text(self, text: str, source: str, chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> int:
        chunks = load_text(text, source=source, chunk_size=chunk_size,
                           chunk_overlap=chunk_overlap)
        return self._add(chunks)

    def _add(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.provider.embed([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        return len(chunks)

    # -- Question answering --------------------------------------------------
    def retrieve(self, question: str) -> list[SearchResult]:
        query_embedding = self.provider.embed([question])[0]
        return self.store.search(query_embedding, top_k=self.top_k)

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            return Answer(text=FALLBACK_MESSAGE, grounded=False, mode=self._mode())
        results = self.retrieve(question)
        if not results or results[0].score < self.min_similarity:
            return Answer(text=FALLBACK_MESSAGE, grounded=False, mode=self._mode())
        if self.provider.name.startswith("local-hash"):
            # Hash embeddings can score above the threshold on pure bucket
            # collisions even when no question word appears in the chunk
            # (see tests/test_rag.py). Require at least one exact shared
            # word in offline mode; the OpenAI provider is semantic, so
            # this gate does not apply to it.
            from embeddings import _tokens

            q_tokens = set(_tokens(question))
            shared = q_tokens & set(_tokens(results[0].chunk.text))
            # A single shared word (e.g. "policy" in "refund policy ...")
            # is not evidence the chunk answers the question. Require at
            # least 2 shared words for longer questions, 1 for very short
            # ones; the OpenAI provider is semantic, so this gate does
            # not apply to it.
            needed = 2 if len(q_tokens) >= 3 else 1
            if len(shared) < needed:
                return Answer(text=FALLBACK_MESSAGE, grounded=False, mode=self._mode())
        sources = [r.chunk for r in results]
        context = "\n\n".join(
            f"[{c.citation()}] {c.text}" for c in sources
        )
        text = self._generate(question, context, sources)
        return Answer(text=text, sources=sources, grounded=True, mode=self._mode())

    def _mode(self) -> str:
        return "openai" if self.provider.name == "openai" else "offline-extractive"

    def _generate(self, question: str, context: str, sources: list[DocumentChunk]) -> str:
        """LLM answer when OpenAI is configured, else an extractive answer."""
        if self.provider.name == "openai" and (OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")):
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=OPENAI_API_KEY or os.environ["OPENAI_API_KEY"])
                prompt = GROUNDED_PROMPT_TEMPLATE.format(
                    fallback=FALLBACK_MESSAGE, context=context, question=question
                )
                resp = client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                answer = (resp.choices[0].message.content or "").strip()
                if answer:
                    return f"{answer}\n\nSources: {', '.join(_citations(sources))}"
            except Exception:
                pass  # fall through to the offline extractive answer
        # Offline extractive mode: quote the most relevant retrieved chunk.
        # Labelled honestly — this is not a generated summary, it is the
        # source text itself, so it cannot hallucinate beyond the documents.
        best = sources[0]
        cites = ", ".join(_citations(sources))
        return (
            "Based on the uploaded documents:\n\n"
            f"{best.text.strip()}\n\n"
            f"Sources: {cites}"
        )
