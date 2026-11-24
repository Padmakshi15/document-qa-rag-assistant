# Document Q&A Assistant — RAG

Ask questions about your own documents and get answers **only from those
documents**, with the source page shown for every answer. If the documents
do not contain the answer, the assistant says so instead of guessing.

Built with Python, Streamlit, ChromaDB and an LLM API, with an offline
fallback so the whole pipeline also runs with no API key.

**Author:** Padmakshi Bhonagiri

## What it does

- Upload PDF (`.pdf`) or text (`.txt`, `.md`) documents.
- Ask a question in plain language.
- Get a grounded answer plus citations such as `sample_policy.txt, p. 1`.
- Get a clear fallback message when no supporting text is found:
  > "I couldn't find supporting information in the uploaded documents
  > to answer that question."

## Architecture

```
Upload PDF / TXT
      │  ingest.py — text extraction (PyPDF / plain text)
      ▼
Chunking (1000 chars, 200 overlap)
      │  ingest.py — chunk_text()
      ▼
Embeddings
      │  embeddings.py — OpenAI, or local hash embedding offline
      ▼
ChromaDB vector store (./chroma_db, persistent)
      │  vector_store.py — add / cosine-similarity search
      ▼
Retrieve top-4 chunks
      │  qa.py — retrieve()
      ▼
Grounded answer
         qa.py — grounded prompt ("answer only from this context")
                 + fallback when no supporting text is found
```

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI: upload, settings, ask, sources |
| `ingest.py` | PDF/TXT extraction, chunking with overlap |
| `embeddings.py` | OpenAI embeddings, or offline hash fallback |
| `vector_store.py` | ChromaDB persistent store, in-memory fallback |
| `qa.py` | Retrieval, grounded prompt, fallback, citations |
| `config.py` | Tested defaults (chunk size / overlap / top-k) |
| `eval.py` | Fixed-question evaluation, grounded-answer rate |
| `tests/test_rag.py` | pytest suite (offline) |

## How to run locally

```bash
git clone https://github.com/PadhuPadmakshi/document-qa-rag-assistant.git
cd document-qa-rag-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Then open the address Streamlit prints (usually http://localhost:8501).

## How to configure LLM / embeddings keys

No key is needed to run the app. Without a key it uses:

- **Embeddings:** local hash embedding (offline, labelled in the sidebar).
  It matches words, not meanings — it cannot tell that "car" and
  "automobile" mean the same thing. Good for learning and testing, not
  for production-quality semantic search.
- **Answers:** extractive — the answer quotes the most relevant retrieved
  chunk, so it cannot hallucinate beyond your documents.

With an OpenAI key, the app uses OpenAI embeddings and a grounded LLM answer:

```bash
export OPENAI_API_KEY="sk-..."
# Optional overrides:
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
streamlit run app.py
```

Never commit a key: `.env` and `.streamlit/secrets.toml` are git-ignored.
On Streamlit Cloud, set `OPENAI_API_KEY` in the app's Secrets instead.

## How to upload docs and ask questions

1. In the sidebar, keep the defaults or change chunk size / overlap / top-k.
2. Upload one or more PDF or TXT files and click **Index uploaded documents**.
3. Type a question and click **Ask**.
4. Read the answer, then open **Sources** to see each retrieved chunk with
   its source file and page number, and check the answer against it.

Try the sample documents first — they are already in `samples/`:

- `samples/sample_guide.txt` — how the assistant works
- `samples/sample_policy.txt` — document retention policy (90 days)
- `samples/sample_handbook.pdf` — onboarding handbook (first week, mentor)

## Tested settings

Different chunk sizes, overlap values and retrieval counts were tested
against the fixed set of sample questions in `samples/eval_questions.json`
(see `eval.py`). The settings below gave the most accurate grounded answers.

| Setting | Value | Why |
| --- | --- | --- |
| Chunk size | 1000 chars | Keeps a paragraph of context together; a retrieved chunk is mostly relevant |
| Chunk overlap | 200 chars | An answer straddling a boundary is not cut in half |
| Top-k retrieval | 4 chunks | Enough context to answer; more added noise without better answers |
| Min. similarity | 0.10 | Below this, the assistant uses the fallback instead of guessing (plus an exact-word overlap check in offline mode, to rule out hash collisions) |

Change them in the sidebar, or with environment variables
`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`.

## Evaluation

```bash
python eval.py
```

Builds a fresh assistant from `samples/`, asks the 6 fixed questions in
`samples/eval_questions.json` (4 answerable, 2 unanswerable), and prints a
per-question PASS/FAIL plus the grounded-answer rate. The 2 unanswerable
questions must return the fallback — a correct "I don't know" counts as a
pass.

## Tests

```bash
python -m pytest
```

Runs offline (local hash embeddings, in-memory store — no key, no network,
no ChromaDB needed). Covers chunking overlap, retrieval relevance, the
fallback with no context and with an unrelated question, citations
(source + page), PDF extraction, config defaults, and embedding similarity.

## Learn RAG with this project

See [`LEARN.md`](LEARN.md) — a beginner-friendly guide that explains RAG
step by step, what embeddings and vector search mean in plain words, each
file's role, the flow of one question, and three hands-on exercises.

## License

MIT — see [LICENSE](LICENSE).
