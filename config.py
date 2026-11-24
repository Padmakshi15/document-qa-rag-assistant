"""Configuration for the Document Q&A Assistant (RAG).

These are the settings kept after testing different chunk sizes, overlap
values and retrieval counts against a fixed set of sample questions
(see ``samples/eval_questions.json`` and ``eval.py``).
"""

import os

# --- Chunking (tested defaults) -------------------------------------------
# CHUNK_SIZE=1000 characters: large enough to keep a paragraph of context
# together, small enough that a retrieved chunk is mostly relevant.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
# CHUNK_OVERLAP=200 characters: the tail of one chunk repeats at the start
# of the next, so an answer that straddles a boundary is not cut in half.
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
# TOP_K=4: retrieve the four most similar chunks per question. Fewer missed
# supporting text; more added noise and cost without better answers.
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

# --- Retrieval --------------------------------------------------------------
# Minimum cosine similarity for a chunk to count as supporting evidence.
# Below this the assistant answers with the fallback instead of guessing.
MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", "0.10"))

# --- Embeddings / LLM -------------------------------------------------------
# OpenAI is used only when an API key is set. Without one the app runs
# fully offline with the local hash embedding fallback (see embeddings.py).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Local fallback embedding width (hash embedding, see embeddings.py).
LOCAL_EMBEDDING_DIM = 384

# --- Storage ----------------------------------------------------------------
CHROMA_DB_DIR = os.environ.get("RAG_CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "document_qa"

# --- App --------------------------------------------------------------------
APP_TITLE = "Document Q&A Assistant"
MAX_UPLOAD_MB = 20
