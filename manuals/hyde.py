"""HyDE (rung 6): search with a fake answer instead of the question (Gao et al., 2022, "Precise Zero-Shot Dense
Retrieval without Relevance Labels").

A question ("my pump won't shut off") and a manual entry ("RUN ON WON'T SHUT DOWN | FLASHING GREEN ... verify
precharge") are written very differently, so their vectors sit apart. HyDE asks the LLM to WRITE a short manual-style
entry that answers the question, then embeds that: a fake answer looks more like the real answer than the question
does. The facts in it may be wrong (the LLM has never read these manuals); only its shape and vocabulary matter.

Only DENSE search uses it. BM25 / SPLADE keep the original question, so an exact code ("A13", "F-03") is never
replaced by whatever code the LLM guessed.

Fake passages are cached on disk per question (data/manufacturing/cache/hyde_<model>.json) with the tokens and time
they took, so every HyDE rung reads the SAME passage (a fair comparison) and re-runs cost nothing; the ladder still
counts the original LLM time and tokens as part of each question.
"""

import json
import time
from functools import cache

import numpy as np
from eval_harness import settings, traced_openai

from ingest import CACHE
from retrieve import _dense, timed

PROMPT = """You write passages for equipment maintenance manuals (pump controllers, variable frequency drives).
Write the manual passage that answers the question below, the way a troubleshooting section would print it:
the fault or alarm code and its name, the likely causes, and the corrective actions. About 100 words.
No introduction, no disclaimers.

Question: {question}"""
CACHE_FILE = CACHE / f"hyde_{settings.llm_model}.json"
USAGE = {"tokens_in": 0, "tokens_out": 0, "llm_ms": 0.0}  # this question's LLM use; the ladder resets it per question
# llm_ms counts only passages served from the cache: a live call is already inside the harness's own timing


@cache
def _client():
    return traced_openai()


@cache
def _cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def fake_passage(question: str) -> str:
    """The LLM's made-up manual entry for this question (cached), adding its tokens and time to USAGE."""
    store = _cache()
    cached = question in store
    if not cached:
        start = time.perf_counter()
        resp = _client().chat.completions.create(
            model=settings.llm_model, temperature=0, max_tokens=250,
            messages=[{"role": "user", "content": PROMPT.format(question=question)}])
        store[question] = {"text": resp.choices[0].message.content.strip(),
                           "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens,
                           "ms": (time.perf_counter() - start) * 1000}
        CACHE_FILE.write_text(json.dumps(store, indent=1))
    hit = store[question]
    USAGE["tokens_in"] += hit["tokens_in"]
    USAGE["tokens_out"] += hit["tokens_out"]
    USAGE["llm_ms"] += hit["ms"] if cached else 0.0
    return hit["text"]


def hyde_vector(question: str, mix_question: bool) -> list[float]:
    """The dense query vector. mix_question=False: the fake passage alone (r6a). True: the average of the question's
    vector and the passage's vector (r6b, as in the paper, which averages the fake documents with the query)."""
    passage = fake_passage(question)
    with timed("hyde_embed"):
        doc = next(_dense().passage_embed([passage]))  # the fake passage is a DOCUMENT: no query instruction
        if not mix_question:
            return doc.tolist()
        query = next(_dense().query_embed(question))
        mean = (doc + query) / 2
        return (mean / np.linalg.norm(mean)).tolist()
