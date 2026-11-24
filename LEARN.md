# Learn RAG with the Document Q&A Assistant

A beginner-friendly guide. No prior knowledge of RAG is assumed — only
basic Python. Everything here refers to the real code in this repo, so
you can read a section and then open the file it talks about.

## 1. What is RAG, in plain words?

**RAG = Retrieval-Augmented Generation.**

An ordinary chatbot answers from what it memorised during training. That
has two problems: it may not know *your* documents, and it may invent a
confident-sounding answer when it does not know.

RAG fixes both by looking things up first:

1. **Retrieve** — find the passages in *your* documents that are most
   relevant to the question.
2. **Generate** — give those passages to a language model and tell it:
   "answer only from this context."

The model is no longer answering from memory; it is answering from the
passages in front of it. And if no passage supports an answer, the
assistant says so — that is the fallback in `qa.py`.

Think of it like an open-book exam: the model may use the book, must cite
the page, and must leave a question blank if the book does not cover it.

## 2. What are embeddings?

An **embedding** is a list of numbers that represents a piece of text —
its "coordinates" in a meaning-space. Texts with similar meanings end up
with similar numbers, so similarity becomes arithmetic: compare two lists
of numbers and you get a score.

- In this project the real embeddings come from OpenAI
  (`text-embedding-3-small`) when `OPENAI_API_KEY` is set.
- Without a key, `embeddings.py` uses a **local hash embedding**: each word
  is hashed into one of 384 buckets and counted. It is honest about its
  limits — it matches *words*, not meanings. "Car" and "automobile" land
  in different buckets, so it cannot see they are related. It exists so
  you can run and learn the whole pipeline offline.

**Vector search** is the lookup step: the question is embedded the same
way as the documents, and the store returns the chunks whose embeddings
are closest to the question's embedding. "Closest" is measured with
**cosine similarity** — a score from 0 (unrelated) to 1 (identical).
See `cosine_similarity()` in `embeddings.py`.

## 3. Each file's role

| File | What it does | Key function |
| --- | --- | --- |
| `config.py` | The tested defaults | `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`, `TOP_K = 4` |
| `ingest.py` | Reads PDFs/TXT and splits text into chunks | `load_document()`, `chunk_text()` |
| `embeddings.py` | Turns text into numbers | `get_embedding_provider()`, `hash_embed()` |
| `vector_store.py` | Stores embeddings, finds the closest chunks | `VectorStore.add()`, `VectorStore.search()` |
| `qa.py` | The whole ask-a-question flow | `RAGAssistant.ask()` |
| `app.py` | The Streamlit web UI | `main()` |
| `eval.py` | Scores the assistant on fixed questions | `main()` |

## 4. The flow of one question

Trace this question through the code:
*"How long are uploaded documents retained?"*

1. **You ask** — `app.py` calls `assistant.ask(question)`.
2. **Embed the question** — `qa.py` → `provider.embed([question])`.
   The question becomes a list of numbers.
3. **Retrieve** — `vector_store.search()` compares that list with every
   stored chunk embedding (cosine similarity) and returns the top 4.
   The best match is a chunk from `sample_policy.txt` containing
   "retained for 90 days".
4. **Check for evidence** — if the store is empty, or the best score is
   below `MIN_SIMILARITY` (0.10), `ask()` returns the fallback message.
   No guessing.
5. **Answer** — with a key, the grounded prompt in `qa.py`
   (`GROUNDED_PROMPT_TEMPLATE`) sends the retrieved chunks and the
   question to the LLM with the instruction "answer only from this
   context". Without a key, the offline mode quotes the best chunk.
6. **Cite** — the answer lists its sources, e.g.
   `sample_policy.txt, p. 1`, and the UI shows each source chunk under
   **Sources** so you can verify the answer yourself.

## 5. Why chunking, and why these numbers?

A whole document is too long to embed as one piece: the embedding would
blur many topics together, and the model would receive far more text than
it needs. So `chunk_text()` in `ingest.py` splits text into overlapping
pieces with a sliding window:

- **Chunk size 1000** — about a short paragraph. Big enough to keep
  context together, small enough that a retrieved chunk is mostly relevant.
- **Overlap 200** — the last 200 characters of one chunk repeat at the
  start of the next, so a sentence split across a boundary survives.
- **Top-k 4** — four chunks give enough context; more added noise without
  better answers in the evaluation.

These were not guessed: different values were tested against the fixed
questions in `samples/eval_questions.json` and these gave the most
accurate grounded answers. Run `python eval.py` to see the score.

## 6. Exercises

### Exercise 1 — Try a different chunk size

Edit `config.py`: set `CHUNK_SIZE = 300` and `CHUNK_OVERLAP = 50`, then
run `python eval.py`. Compare the grounded-answer rate with the default
(1000 / 200). Then try `CHUNK_SIZE = 3000`. What changes, and why?
Small chunks lose context; very large chunks dilute relevance. Restore
the defaults when you are done.

### Exercise 2 — Add a new document

Write a short `.txt` file about a topic only you know — for example, a
favourite recipe with a specific ingredient amount. Put it in `samples/`,
run `python eval.py` (it indexes every file in `samples/`), then start
the app with `streamlit run app.py`, upload your file, and ask about the
ingredient amount. Check the citation: does it point at your file and
the right page?

### Exercise 3 — Ask an unanswerable question

With the sample documents indexed, ask: *"What is the refund policy for
cancelled subscriptions?"* No sample document covers refunds, so you
should get the fallback message, not an invented policy. Then ask a
question the documents *do* answer and compare the two. The difference
— a grounded answer with citations versus an honest "I don't know" —
is the whole point of RAG.

## 7. Where to go next

- Read `tests/test_rag.py` — each test isolates one behaviour (overlap,
  retrieval, fallback, citations) and is a good model for testing RAG code.
- Swap the local hash embedding for OpenAI embeddings (set
  `OPENAI_API_KEY`) and re-run `eval.py` — watch which questions improve.
- Try a larger `TOP_K` in the sidebar and see when extra chunks stop helping.
- As a stretch goal, add a `.docx` loader to `ingest.py` following the
  same pattern as `extract_text_from_txt()`.
