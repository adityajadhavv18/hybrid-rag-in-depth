"""Contextual chunk notes (rung 9): an LLM writes 1-2 sentences per chunk saying where it sits, and the note goes
into the chunk's INDEX text (what dense / BM25 / SPLADE read). The stored text, which the LLM answers from and the
harness matches evidence against, is unchanged. Same idea as Anthropic's "Contextual Retrieval" (2024).

Why: a chunk often doesn't say what it is about. The second window of a long entry, a table that continues an
alarm on the next page, or a wiring page that never names its manual section. The note puts that back:
    "Yaskawa GA500 troubleshooting, section 2.4 Faults: continuation of fault oL2 (Drive Overload), listing more
     causes and fixes."

Context given to the LLM: the chunk's page plus one page either side (the whole manual would cost ~50x more). The
pages come first in the prompt so chunks of the same page share a long identical prefix (OpenAI caches it).

Notes are cached by chunk_id + text hash in data/manufacturing/cache/notes_<model>.json, so a crash resumes and
re-indexing is free. Bulk indexing uses a plain OpenAI client: 1.7k traces would bury the pipeline traces.

    uv run manuals/notes.py --pilot 8   # print notes for a few chunks first
    uv run manuals/notes.py             # all chunks of the rung 8 (structure) chunking
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache

from eval_harness import settings
from eval_harness.metrics import cost_usd
from openai import OpenAI

from chunks import MANUAL_NAMES, TEXT_DIR, Chunk
from ingest import CACHE
from structure import load_struct_chunks

PROMPT = """<manual>{manual}</manual>
<pages>
{pages}
</pages>
Here is a chunk from the pages above, which an index will search:
<chunk>
{chunk}
</chunk>
Write 1-2 short sentences that situate this chunk for search: which section it is in, and which fault or alarm
code (with its name) or topic it belongs to. A chunk that begins with a code heading is that code's own entry. Only
if it begins mid-entry (or with "(continued)") does it continue an earlier entry; then name that entry.
Do not name the manual (it is added separately) and do not start with "This chunk". Answer with the sentences only."""
CACHE_FILE = CACHE / f"notes_{settings.llm_model}.json"
WORKERS = 8  # parallel API calls
PAGE_CHARS = 4000  # cap per page: a few pages are huge tables


@cache
def _pages(doc_id: str) -> list[str]:
    return (TEXT_DIR / f"{doc_id}.txt").read_text().split("\f")


def _key(c: Chunk) -> str:  # a new prompt or new chunk text means a new note
    return f"{c.chunk_id}:{hashlib.sha1((PROMPT + c.text).encode()).hexdigest()[:12]}"


def _prompt(c: Chunk) -> str:
    pages = _pages(c.doc_id)
    nearby = range(max(1, c.page - 1), min(len(pages), c.page + 1) + 1)
    return PROMPT.format(manual=MANUAL_NAMES[c.doc_id], chunk=c.text,
                         pages="\n".join(f"[page {p}]\n{pages[p - 1][:PAGE_CHARS]}" for p in nearby))


def _write(client: OpenAI, c: Chunk) -> dict:
    resp = client.chat.completions.create(model=settings.llm_model, temperature=0, max_tokens=120,
                                          messages=[{"role": "user", "content": _prompt(c)}])
    return {"note": resp.choices[0].message.content.strip(),
            "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens}


def build_notes(chunks: list[Chunk]) -> dict[str, str]:
    """chunk_id -> note, writing any that are missing from the cache."""
    store = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    todo = [c for c in chunks if _key(c) not in store]
    if todo:
        client = OpenAI()
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(_write, client, c): c for c in todo}
            for i, done in enumerate(as_completed(futures), start=1):
                store[_key(futures[done])] = done.result()
                if i % 100 == 0 or i == len(todo):  # save as we go: a crash resumes from here
                    CACHE_FILE.write_text(json.dumps(store, indent=1))
                    print(f"     notes: {i}/{len(todo)}")
    tin = sum(store[_key(c)]["tokens_in"] for c in chunks)
    tout = sum(store[_key(c)]["tokens_out"] for c in chunks)
    print(f"     notes for {len(chunks)} chunks: {tin:,} tokens in, {tout:,} out, "
          f"${cost_usd(tin, tout, settings.llm_model):.2f} (one-off, at index time)")
    return {c.chunk_id: store[_key(c)]["note"] for c in chunks}


if __name__ == "__main__":
    chunks = load_struct_chunks()
    if "--pilot" in sys.argv:
        n = int(sys.argv[sys.argv.index("--pilot") + 1])
        # a spread of kinds: an entry, a continuation window, a fixed-chunk page, each manual
        pick = [c for c in chunks if "(continued)" in c.text][:2] + chunks[:: len(chunks) // (n - 2)][: n - 2]
        for c, note in zip(pick, build_notes(pick).values()):
            print(f"\n--- {c.chunk_id}\n    chunk: {c.text[:90]!r}\n    NOTE:  {note}")
    else:
        build_notes(chunks)
