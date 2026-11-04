# Hybrid RAG #1: equipment maintenance manuals

A technician types "danfoss fc202 showing A13" or "my pump won't shut off", and gets the fault's meaning, causes and fixes from the right manual, with page citations. Search runs three ways at once (meaning, keywords, learned keywords) and the three ranked lists are merged into one.

```
INDEX (once):  5 manuals → one chunk per fault entry (structure.py) → index text = "Manual: <name>" + code tokens + chunk
               → Qdrant: dense (bge-small) + BM25 + SPLADE vectors per chunk

QUERY:  question ─┬─ gpt-4o-mini writes a fake ~100-word manual entry → dense search with it ──┐
                  ├─ BM25 on the question (+ code tokens: "A13" → a13) ─────────────────────────┼─ weighted RRF (k=20,
                  └─ SPLADE on the question ────────────────────────────────────────────────────┘  keyword lists ×1.5)
                                                                                         → top 5 → gpt-4o-mini answers
                                                                                            with [n] citations or refuses
```

Think of two librarians. One reads for **meaning**: ask "the pump keeps running" and she brings the page about "won't shut down", although no word matches. The other reads for **exact words**: ask "A13" and he finds "ALARM 13" because he was taught they're the same code. Neither is good alone. Fault manuals need both, because questions mix plain-language symptoms with exact codes, and a code like "F02" means a different thing on every machine.

Hybrid search isn't one paper. It's a stack of well-known parts, each measured here as one rung of a ladder:

- **BM25** (Robertson et al., 1990s): keyword scoring, rare words count most.
- **SPLADE** (Formal et al., 2021): a learned keyword vector that also lights up related words ("overheat" → "temperature").
- **Reciprocal Rank Fusion** (Cormack et al., 2009): merge ranked lists by position only, never raw scores.
- **HyDE** (Gao et al., 2022): search with a fake answer instead of the question.
- Tried and dropped: **ColBERT** (Khattab & Zaharia, 2020), **cross-encoder rerankers**, and Anthropic's **Contextual Retrieval** (2024) notes.

## The corpus and the golden set

Five public manuals for pump controllers and drives (PDFs aren't committed: `scripts/download_manuals.py` fetches them):

| Manual | Its codes look like |
|---|---|
| Franklin Electric SubDrive | "7 flashes" |
| Goulds Aquavar SOLO2 | F00–F09, "3 Blinks" |
| Goulds Aquavar IM043 (V120) | named errors ("Overtemp Motor", "Error 6"), part nos 0416-xxxx |
| Danfoss VLT AQUA Drive FC 202 | WARNING / ALARM 1–104 (no alarm 21) |
| Yaskawa GA500 troubleshooting | 145 codes (oL1–oL7, oH, Uv1–Uv3, CPF06…), ~60% of all text |

The golden set (in `eval_harness`) has 40 questions: **25 tune / 15 held-out test**. Every setting below was chosen on tune only. The test split was run once, at the end. The question types are the traps this corpus sets:

- **code**: "What does ALARM 14 mean?"
- **sym**: a symptom, no code ("my pump won't shut off")
- **look**: a look-alike in the same family (oL1 vs oL2, Uv2)
- **twin**: the same code on another machine or list (SOLO2 F02 = Dry Well, Franklin 2 flashes = Undervoltage; Yaskawa ov as a fault vs as an alarm)
- **spell**: written the way people type it ("A13", "OH1", "F-03")
- **multi**: two facts in one question
- **none**: no answer exists (Danfoss ALARM 21, SOLO2 F10). The right reply is "I don't know"

## How it works

**1. Clean text** (`scripts/clean_manuals.py`). English pages only. Running heads and footers are cut by where they sit on the page, and Franklin's sideways tables are rebuilt row by row. Pages stay separated by `\f`, so every chunk knows its page.

**2. One chunk per fault entry** (`structure.py`, rung 8). Each manual's fault section is found by its first and last heading. It's split where an entry *starts* (`WARNING 12, Torque limit`, `F03 / 3 Blinks`, Yaskawa's table header + a code line). The code, name, causes and fixes stay in one chunk. That gives 276 entries; an entry over 450 tokens is windowed, and every later window starts `"oL2 Drive Overload (continued)"`. Everything else keeps plain 300/50-token windows per page: 1,701 chunks in all.

**3. Index text ≠ stored text** (`ingest.py`, `codes.py`). What the searches read gets two additions the LLM never sees:
- a `Manual: Danfoss VLT AQUA Drive FC 202` line, so "F02 on the SOLO2" finds the SOLO2's F02
- a canonical token next to every code (`ALARM 13`, `A13` → `a13`; `F-03`, `F3` → `f03`; `7 blinks` → `flash7`), added to the question too. BM25 then matches the code exactly and weights it heavily, because it's rare.

**4. Three searches, one merge** (`retrieve.py`, `ladder.py`):

```python
fake = gpt-4o-mini("write the manual passage that answers: " + question)   # HyDE, cached per question
lists = {"dense":  dense_search(embed_as_passage(fake), 20),   # meaning, searched with the fake entry
         "bm25":   sparse_search(question, 20),                # exact words + code tokens
         "splade": sparse_search(question, 20)}                # learned keywords
top5  = rrf_fuse(lists, k=20, weights={"bm25": 1.5, "splade": 1.5})[:5]   # points = weight / (20 + rank)
```

Only the dense list gets the fake passage. The keyword lists keep the real question, so an exact code is never swapped for whatever code the LLM guessed.

**5. Answer** (`answer.py`). The top 5 go to gpt-4o-mini numbered, each with its manual and page. The model answers for the machine the question names, cites `[n]`, or replies exactly "I don't know based on the provided documents."

## Run it

From `hybrid_rag_in_depth/`. Qdrant runs in Docker (`open -a Docker`, then `docker start qdrant`). The OpenAI key is needed for HyDE, notes and answers.

```bash
uv run scripts/download_manuals.py && uv run scripts/clean_manuals.py
uv run scripts/check_golden.py               # every evidence quote is in the corpus, once, on one page
uv run manuals/ingest.py                     # fixed chunks, all 8 vectors   (rungs 0-5)
uv run manuals/ingest.py --struct            # one chunk per entry           (rungs 6-8)
uv run manuals/ingest.py --struct --notes    # + LLM notes                   (rung 9, ~$0.73 once)
uv run manuals/ladder.py r0 r5 r8 r6a        # retrieval-only ladder on tune (free except HyDE)
uv run manuals/fusion_sweep.py               # the rung 5 grid
uv run manuals/answer.py r6a --judge         # answers + RAGAS judge on tune
uv run manuals/answer.py r0 r6a --judge --test   # the final held-out run
```

## Results

### The ladder (tune, 23 answerable questions, retrieval only)

Each rung carries the best rung before it. One question moves MRR by about 0.04.

| Rung | What changed | recall@5 | MRR | p50 latency | Kept? |
|---|---|---|---|---|---|
| r0 | dense only (bge-small) | 0.41 | 0.31 | 12 ms | baseline |
| r1 | + BM25, RRF | 0.54 | 0.32 | 23 ms | ✅ |
| r2 | + code tokens in BM25 | 0.61 | 0.42 | 16 ms | ✅ |
| r2b | + manual name on every chunk's index text | 0.74 | 0.54 | 16 ms | ✅ |
| r3a / r3b | SPLADE / miniCOIL instead of BM25 | 0.70 / 0.74 | 0.55 / 0.54 | | ❌ |
| r3c | BM25 **and** SPLADE | 0.83 | 0.73 | 82 ms | ✅ |
| r4a / r4b | ColBERT re-score / ColBERT as a 4th list | 0.78 / 0.78 | 0.60 / 0.76 | 146 / 298 ms | ❌ |
| r5 | RRF tuned: k=20, keyword lists ×1.5 | 0.87 | 0.79 | 50 ms | ✅ |
| **r8** | one chunk per fault entry | **1.00** | 0.80 | 48 ms | ✅ |
| r7a–e | + cross-encoder rerank of the top 20 (5 models) | 0.78–0.96 | 0.68–0.84 | 0.7–8.9 s | ❌ |
| **r6a** | **+ HyDE: dense searches with the fake passage** | **1.00** | **0.89** | 2.1 s | ✅ **winner** |
| r6b | HyDE with mean(question, passage) vector | 1.00 | 0.86 | 2.1 s | ❌ |
| r9a / r9b | + LLM note per chunk in the index (on r8 / on r6a) | 1.00 / 1.00 | 0.80 / 0.85 | | ❌ |

The rungs were run in this order: 0–5, then 8 (moved up because split entries blocked everything after it), 7, 6, 9.

By question type, MRR:

| | code | look | spell | sym | twin | multi |
|---|---|---|---|---|---|---|
| r0 | 0.22 | 0.17 | 0.00 | 0.60 | 0.08 | 0.67 |
| r5 | 1.00 | 0.40 | 0.50 | 0.80 | 1.00 | 0.83 |
| r8 | 0.92 | 0.67 | 0.58 | 0.90 | 0.75 | 0.83 |
| r6a | 1.00 | 0.61 | 0.73 | 0.90 | 1.00 | 1.00 |

Code tokens and the manual name took **spell** and **twin** from near zero. Entry chunks fixed **look**. HyDE put the right entry first more often.

### The final run (held-out test, 13 answerable + 2 no-answer, RAGAS judge)

`manuals/results/final_test.json` holds every number and answer.

| | r0 (dense only) | **r6a** |
|---|---|---|
| evidence in top 5 (recall@5) | 0.31 | **0.77** |
| MRR | 0.21 | **0.73** |
| faithfulness | 0.81 | **0.90** |
| answer relevancy | 0.51 | **0.75** |
| context precision | 0.48 | **0.85** |
| context recall | 0.56 | **0.87** |
| answerable questions refused | 31% | **8%** |
| no-answer questions refused | 100% | 100% |
| cost / question | $0.00028 | $0.00032 |
| p50 latency | 1.5 s | 1.9 s |

r6a's MRR drops from 0.89 on tune to 0.73 on test. That's expected: every setting was picked by looking at tune, so tune flatters. Test is the honest number. HyDE adds $0.00009 a question, and the judge cost $0.065 for both runs.

## What went wrong, and why

- **Fixed chunks cut entries in half (fixed by r8).** The second half of the oL2 entry ("The drive capacity is too small…") never names oL2, so no question about oL2 found it. One chunk per entry took recall@5 from 0.87 to 1.00. Writing the entry splitter took the most care: contents pages repeat section headings, so the end heading is searched *after* the start. Headings right after a page break needed `(?:^|(?<=\f))`.
- **Signpost chunks.** The code index, contents pages and section intros list *every* code, so they match code questions in all three searches and rank above the real entry. After r8, many answers sat at #2 behind a signpost.
- **Cross-encoder rerankers made it worse (r7).** They read codes as ordinary words. For "oL2" they preferred generic overload paragraphs, and they don't know "A13" means "ALARM 13". The best, bge-reranker-base, reached MRR 0.84 but took 7 s a question on a laptop CPU. None was kept.
- **ColBERT hurt codes (r4).** Token-by-token matching breaks a code like "CPF06" into word pieces, and those pieces match many other chunks. As a 4th list it nudged MRR up but lowered recall@5, at 6× the latency.
- **Contextual notes didn't help (r9).** r8 already makes every entry name itself. The notes added the same boilerplate to every entry ("section detailing warnings and alarms… troubleshooting steps"), so neighbours (ALARM 12 vs 13) and fault/alarm twins looked *more* alike. The first prompt also wrote "UL4 continues UL3", until it was told that a chunk starting with a code heading is that code's own entry. The run cost $0.73 and wasn't kept.
- **HyDE works even when its facts are wrong.** The fake passage for CPF06 calls it "overcurrent" (it's a data error). It still helps, because the passage has the *shape* of a fault entry: code, name, causes, fixes. That shape pulls dense search toward real entries and away from signposts. Searching with the mean of the question and passage vectors (the paper's version) was worse, most likely because the question half pulls the signposts back in.
- **Test misses, 3 of 13:**
  - **sym-04**, "my Yaskawa trips when it slows a heavy load": the answer is the `ov` (overvoltage) entry, but the question names no code. The keyword searches latched onto "inertia" and "deceleration" in parameter tables, and the entry wasn't in the top 20.
  - **sym-02**, "Franklin pressure keeps swinging": the evidence was at #19. Same weakness: **symptom questions with no code** are this stack's soft spot, because it was tuned on mostly code questions.
  - **twin-05**, "when is ov an *alarm* rather than a *fault*": the fault entry was #2 and the alarm entry #8. gpt-4o-mini rightly refused instead of guessing. Yaskawa prints the same code in its fault list (§2.4) and its alarm list (§2.5), and nothing in the chunk says which list it's in.

## What it gets right

- **Codes, written any way.** "A13", "WARNING/ALARM 13" and "alarm 13" all hit the same entry, and F02 on the SOLO2 never returns Franklin's 2 flashes.
- **Honest refusals.** Both no-answer questions were refused on tune and on test. Answerable questions refused fell from 31% to 8%.
- **Cheap and fast enough.** About $0.0003 a question and under 2 s p50, on an 8 GB laptop with local embedders. Only HyDE and the answer call reach OpenAI.
- **Every step is measured.** Each rung changed one thing, on the same chunks and the same golden set, so each gain or loss has a cause.

## Ideas not built

- **Label fault vs alarm entries.** Add "(fault list)" / "(alarm list)" to Yaskawa's index text. This is the direct fix for twin-05.
- **Better symptom questions.** Drop or down-weight signpost chunks (parameter tables, code index), or let HyDE's passage also feed the keyword lists *without its code*, so "deceleration → overvoltage" comes from the fake entry.
- **Fine-tune a reranker** on synthetic questions written from each entry, so it learns that codes are names.
- **Code-expanded rerank query.** Give the cross-encoder "A13 (ALARM 13)" instead of "A13", or blend its score with the RRF rank instead of replacing it.
- **LLM rerank** of the top 10: slower and dearer, but an LLM knows A13 = ALARM 13.
- **Re-tune fusion with HyDE in.** The RRF weights were tuned on r5 and never re-swept after r8 and HyDE changed the lists.

## When I'd reach for it

When users ask with **exact identifiers** that dense search blurs: fault codes, part numbers, drug codes, error IDs, SKUs. And when the same identifier means different things in different documents. Then the keyword side with canonical code tokens does most of the work, and the manual name on every chunk keeps machines apart. Add structure-aware chunks whenever the documents are made of entries (fault tables, formularies, API references). It was the biggest single fix, and it's free. Add HyDE when the answers are written in a fixed shape the questions don't share. Skip all this for plain prose questions with no codes: dense search alone, or dense + BM25, is enough there.

## Files

| File | Purpose |
|---|---|
| `scripts/download_manuals.py` | Fetches the 5 public PDFs into `data/manufacturing/raw/` (git-ignored) |
| `scripts/clean_manuals.py` | PDF → clean text per manual, pages split by `\f`; writes `manuals.json` |
| `scripts/check_golden.py` | Checks every golden evidence quote: in its doc, unique in the corpus, on one page |
| `manuals/chunks.py` | Fixed 300/50-token chunks per page (rungs 0–5), manual names |
| `manuals/structure.py` | One chunk per fault entry: per-manual section bounds and entry patterns (rung 8) |
| `manuals/codes.py` | Canonical code tokens (`A13` → `a13`, `7 blinks` → `flash7`) for BM25 |
| `manuals/ingest.py` | Builds the Qdrant collections (`manuals_fixed`, `manuals_struct`, `manuals_struct_notes`); caches model outputs |
| `manuals/retrieve.py` | Dense, sparse, ColBERT, cross-encoder rerank, RRF and score fusion, with per-stage timing |
| `manuals/fusion_sweep.py` | The rung 5 grid over RRF k, list depth and keyword weights |
| `manuals/hyde.py` | The fake-passage prompt, its cache, and the HyDE query vector (rung 6) |
| `manuals/notes.py` | LLM context notes per chunk, cached (rung 9) |
| `manuals/ladder.py` | Every rung as a search function; scores them on tune (recall@5, MRR, nDCG@5, recall@20, by type) |
| `manuals/answer.py` | Generation on any rung + the harness run with the RAGAS judge, tune or test |
| `manuals/results/ladder_tune.json` | Every rung's tune numbers |
| `manuals/results/final_test.json` | The held-out run: r0 vs r6a summary, by type, per question with answers |
