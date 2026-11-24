"""Evaluate the RAG assistant against a fixed set of sample questions.

Builds a fresh in-memory assistant from ``samples/``, asks every question
in ``samples/eval_questions.json``, and reports the grounded-answer rate:
the share of answerable questions answered from retrieved text with at
least one expected keyword present, plus the share of unanswerable
questions that correctly got the fallback.

Run:  python eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K  # noqa: E402
from qa import FALLBACK_MESSAGE, RAGAssistant  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))


def build_assistant() -> RAGAssistant:
    assistant = RAGAssistant(top_k=TOP_K)
    for name in sorted(os.listdir(os.path.join(BASE, "samples"))):
        if name.endswith((".txt", ".md", ".pdf")):
            assistant.ingest_file(
                os.path.join(BASE, "samples", name),
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            )
    return assistant


def main() -> int:
    with open(os.path.join(BASE, "samples", "eval_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)

    assistant = build_assistant()
    print(f"Indexed {assistant.store.count()} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, top_k={TOP_K})")
    print(f"Embedding mode: {assistant.provider.name}\n")

    answerable_ok = answerable_total = 0
    fallback_ok = fallback_total = 0
    for item in questions:
        q = item["question"]
        answer = assistant.ask(q)
        if item.get("should_answer", True):
            answerable_total += 1
            text_lower = answer.text.lower()
            hits = [kw for kw in item.get("expected_keywords", [])
                    if kw.lower() in text_lower]
            ok = answer.grounded and len(hits) == len(item.get("expected_keywords", []))
            answerable_ok += ok
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] Q: {q}")
            print(f"        keywords found: {hits} | grounded={answer.grounded} "
                  f"| citations={answer.citations()}")
        else:
            fallback_total += 1
            ok = (not answer.grounded) and FALLBACK_MESSAGE[:40] in answer.text
            fallback_ok += ok
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] Q: {q}")
            print(f"        expected fallback, got fallback={not answer.grounded}")

    total = answerable_total + fallback_total
    passed = answerable_ok + fallback_ok
    rate = (passed / total * 100) if total else 0.0
    print(f"\nGrounded-answer rate: {passed}/{total} = {rate:.0f}% "
          f"(answerable {answerable_ok}/{answerable_total}, "
          f"fallback {fallback_ok}/{fallback_total})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
