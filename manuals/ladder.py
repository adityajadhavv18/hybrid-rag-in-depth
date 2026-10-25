"""The retrieval ladder: score each rung on the TUNE golden set, retrieval only (no LLM, so free).

Each rung is a search function. The harness runs it as a pipeline that returns the top TOP_K chunks and an
empty answer (the harness then scores retrieval only), and this script adds the lab-only numbers:
  nDCG@5     like MRR, but credits every useful chunk by position, not only the first
  recall@20  is the evidence anywhere in the top 20? (what a reranker could still rescue)
  by category (the golden id prefix) and latency per stage.
Results accumulate in manuals/results/ladder_tune.json; the test split is only touched at the very end.

Run from hybrid_rag_in_depth/ (Qdrant running):
    uv run manuals/ladder.py            # all rungs
    uv run manuals/ladder.py r1 r2      # just these
"""

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval_harness import PipelineOutput, evaluate, load_golden

import hyde
from ingest import (BM25_CODE, BM25_NAMED, DENSE_NAMED, MINICOIL_NAMED, NOTES_COLLECTION, SPLADE_NAMED,
                    STRUCT_COLLECTION)
from retrieve import STAGE_MS, colbert_rescore, colbert_search, dense_search, rerank, rrf_fuse, sparse_search

TUNE = "golden_manufacturing_tune_v1.jsonl"
TOP_K = 5  # chunks the LLM will read
CANDIDATES = 20  # how deep each retriever's list goes into fusion
RESULTS = Path(__file__).resolve().parent / "results" / "ladder_tune.json"

RRF_K = 20  # r5, from fusion_sweep.py (was the paper's 60)
KEYWORD_WEIGHTS = {"bm25": 1.5, "splade": 1.5}  # r5: keyword lists count 1.5x the dense list
RESCORE_POOL = 30  # fused candidates ColBERT re-scores in r4a
RERANK_POOL = 20  # r7: fused r8 candidates the cross-encoder re-sorts (r8 has recall@20 = 1.0)
RERANKERS = {  # r7 variants: FastEmbed cross-encoders, smallest first
    "r7a": "Xenova/ms-marco-MiniLM-L-6-v2",
    "r7b": "Xenova/ms-marco-MiniLM-L-12-v2",
    "r7c": "jinaai/jina-reranker-v1-turbo-en",
    "r7d": "BAAI/bge-reranker-base",
    "r7e": "BAAI/bge-reranker-v2-m3-int8",
}


def r5_stack(q: str, k: int, collection: str | None = None, dense_vector: list[float] | None = None) -> list:
    """The current best: dense + BM25 + SPLADE on named, code-tagged chunks, weighted RRF (k=20, keyword x1.5).
    dense_vector: what the dense list searches with instead of the question (HyDE); keyword lists keep the question."""
    where = {"collection": collection} if collection else {}
    return rrf_fuse({"dense": dense_search(q, CANDIDATES, using=DENSE_NAMED, vector=dense_vector, **where),
                     "bm25": sparse_search(q, CANDIDATES, using=BM25_NAMED, **where),
                     "splade": sparse_search(q, CANDIDATES, using=SPLADE_NAMED, **where)},
                    k=RRF_K, weights=KEYWORD_WEIGHTS)[:k]


def r3c_lists(q: str) -> list:
    """The r3c stack's fused ranking, cut to the re-scoring pool."""
    return rrf_fuse({"dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
                     "bm25": sparse_search(q, CANDIDATES, using=BM25_NAMED),
                     "splade": sparse_search(q, CANDIDATES, using=SPLADE_NAMED)})[:RESCORE_POOL]


RUNGS = {
    "r0": ("dense only", lambda q, k: dense_search(q, k)),
    "r1": ("dense + BM25, RRF", lambda q, k: rrf_fuse({"dense": dense_search(q, CANDIDATES),
                                                       "bm25": sparse_search(q, CANDIDATES)})[:k]),
    "r2": ("+ code-aware BM25", lambda q, k: rrf_fuse({"dense": dense_search(q, CANDIDATES),
                                                      "bm25": sparse_search(q, CANDIDATES, using=BM25_CODE)})[:k]),
    "r2b": ("+ manual name on every chunk", lambda q, k: rrf_fuse({
        "dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
        "bm25": sparse_search(q, CANDIDATES, using=BM25_NAMED)})[:k]),
    # r3: which sparse ("keyword") retriever pairs best with dense? All on the r2b named + code-tagged text.
    "r3a": ("SPLADE instead of BM25", lambda q, k: rrf_fuse({
        "dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
        "splade": sparse_search(q, CANDIDATES, using=SPLADE_NAMED)})[:k]),
    "r3b": ("miniCOIL instead of BM25", lambda q, k: rrf_fuse({
        "dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
        "minicoil": sparse_search(q, CANDIDATES, using=MINICOIL_NAMED)})[:k]),
    "r3c": ("BM25 + SPLADE together", lambda q, k: rrf_fuse({
        "dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
        "bm25": sparse_search(q, CANDIDATES, using=BM25_NAMED),
        "splade": sparse_search(q, CANDIDATES, using=SPLADE_NAMED)})[:k]),
    # r4: ColBERT on top of the r3 winner (r3c), two ways
    "r4a": ("ColBERT re-scores fused top 30", lambda q, k: colbert_rescore(q, r3c_lists(q))[:k]),
    "r4b": ("ColBERT as a 4th list in RRF", lambda q, k: rrf_fuse({
        "dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
        "bm25": sparse_search(q, CANDIDATES, using=BM25_NAMED),
        "splade": sparse_search(q, CANDIDATES, using=SPLADE_NAMED),
        "colbert": colbert_search(q, CANDIDATES)})[:k]),
    # r5: fusion tuned by fusion_sweep.py. ColBERT (r4) did not earn its place, so this builds on r3c.
    # The middle of the plateau the sweep found (k 10-20, keyword lists 1.5-2x dense), not its single best point.
    "r5": ("r3c with tuned RRF (k=20, keyword x1.5)", lambda q, k: r5_stack(q, k)),
    # r8: the same r5 search over structure-aware chunks (structure.py): one chunk per fault entry
    "r8": ("r5 on one-chunk-per-fault-entry", lambda q, k: r5_stack(q, k, STRUCT_COLLECTION)),
}
# r6: HyDE (hyde.py) on the r8 winner. The LLM's fake manual passage feeds the dense list only.
RUNGS["r6a"] = ("r8 + HyDE: dense searches with the fake passage",
                lambda q, k: r5_stack(q, k, STRUCT_COLLECTION, dense_vector=hyde.hyde_vector(q, mix_question=False)))
RUNGS["r6b"] = ("r8 + HyDE: dense searches with mean(question, passage)",
                lambda q, k: r5_stack(q, k, STRUCT_COLLECTION, dense_vector=hyde.hyde_vector(q, mix_question=True)))
# r9: contextual notes (notes.py) in the index text of the r8 chunks. r9a isolates the notes (vs r8);
# r9b is the full stack (vs r6a, the HyDE winner).
RUNGS["r9a"] = ("r8 + LLM note per chunk in the index", lambda q, k: r5_stack(q, k, NOTES_COLLECTION))
RUNGS["r9b"] = ("r6a (HyDE) + LLM note per chunk in the index",
                lambda q, k: r5_stack(q, k, NOTES_COLLECTION, dense_vector=hyde.hyde_vector(q, mix_question=False)))
# r7: r8's fused top 20, re-sorted by a cross-encoder (no RRF after it: the reranker's order is final)
for key, model in RERANKERS.items():
    RUNGS[key] = (f"r8 + rerank top {RERANK_POOL}: {model.split('/')[1]}",
                  lambda q, k, m=model: rerank(q, r5_stack(q, RERANK_POOL, STRUCT_COLLECTION), m)[:k])


def norm(text: str) -> str:  # the harness's matching rule
    return re.sub(r"\s+", " ", text).strip().lower()


def fragments(item) -> list[str]:
    return [norm(f) for f in re.split(r" \.\.\. | / ", item.evidence or "") if f.strip()]


def first_rank(frags: list[str], texts: list[str]) -> int | None:
    return next((i for i, t in enumerate(texts, start=1) if any(f in norm(t) for f in frags)), None)


def ndcg(frags: list[str], texts: list[str], k: int = TOP_K) -> float:
    """A chunk scores if it holds an evidence fragment no higher chunk already supplied."""
    found, dcg = set(), 0.0
    for i, t in enumerate(texts[:k], start=1):
        new = {f for f in frags if f in norm(t)} - found
        if new:
            dcg += 1 / math.log2(i + 1)
            found |= new
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(frags), k) + 1))
    return dcg / ideal


def run_rung(key: str) -> dict:
    label, search = RUNGS[key]
    STAGE_MS.clear()

    llm_ms = {}  # question -> LLM time (HyDE) of a passage served from the cache, so cached runs show the real cost

    def pipeline(question: str) -> PipelineOutput:
        hyde.USAGE.update(tokens_in=0, tokens_out=0, llm_ms=0.0)
        hits = search(question, TOP_K)
        llm_ms[question] = hyde.USAGE["llm_ms"]
        return PipelineOutput(answer="", contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits],
                              tokens_in=hyde.USAGE["tokens_in"], tokens_out=hyde.USAGE["tokens_out"])

    run = evaluate(pipeline, arch=f"manuals_{key}", golden_set=TUNE, use_llm_judge=False, compare_to=None)
    stages = {s: round(statistics.median(ms), 1) for s, ms in STAGE_MS.items()}
    if hyde.USAGE["tokens_in"]:  # an LLM rung: its call time, from the cache (the original call's time)
        stages["llm"] = round(statistics.median(hyde._cache()[q]["ms"] for q in llm_ms), 1)

    golden = {g.id: g for g in load_golden(TUNE)}
    rows, by_cat = [], defaultdict(list)
    for q in run.questions:
        g = golden[q.golden_id]
        if g.question_type == "no_answer":
            continue
        frags = fragments(g)
        deep = [h.text for h in search(g.question, CANDIDATES)]  # outside the timed run
        row = {"id": g.id, "recall5": q.scores["evidence_recall"], "mrr": q.scores["evidence_mrr"],
               "ndcg5": ndcg(frags, q.output.contexts),
               "recall20": sum(any(f in norm(t) for t in deep) for f in frags) / len(frags),
               "rank20": first_rank(frags, deep)}
        rows.append(row)
        by_cat[g.id.split("-")[0]].append(row)

    mean = lambda rs, m: round(statistics.mean(r[m] for r in rs), 3)
    lat = sorted(q.latency_ms + llm_ms.get(golden[q.golden_id].question, 0.0) for q in run.questions)
    result = {
        "rung": key, "label": label, "questions": len(rows),
        "recall5": mean(rows, "recall5"), "mrr": mean(rows, "mrr"), "ndcg5": mean(rows, "ndcg5"),
        "recall20": mean(rows, "recall20"),
        "latency_p50_ms": round(statistics.median(lat), 1), "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "stage_p50_ms": stages, "cost_per_query_usd": run.summary.get("cost_per_query_usd", 0.0),
        "by_category": {c: {"recall5": mean(rs, "recall5"), "mrr": mean(rs, "mrr"), "n": len(rs)}
                        for c, rs in sorted(by_cat.items())},
        "per_question": rows,
    }
    misses = [f"{r['id']}(#{r['rank20'] or '>20'})" for r in rows if r["mrr"] < 1]
    print(f"\n{key} {label}: not ranked #1 -> {', '.join(misses) or 'none'}")
    return result


def print_table(results: dict) -> None:
    print("\n| rung | what | recall@5 | MRR | nDCG@5 | recall@20 | p50 ms |")
    print("|---|---|---|---|---|---|---|")
    for r in results.values():
        print(f"| {r['rung']} | {r['label']} | {r['recall5']:.2f} | {r['mrr']:.2f} | {r['ndcg5']:.2f} "
              f"| {r['recall20']:.2f} | {r['latency_p50_ms']:.0f} |")
    cats = sorted({c for r in results.values() for c in r["by_category"]})
    print("\nMRR by category:\n| rung | " + " | ".join(f"{c} (n={next(iter(results.values()))['by_category'][c]['n']})"
                                                  for c in cats) + " |")
    print("|---|" + "---|" * len(cats))
    for r in results.values():
        print(f"| {r['rung']} | " + " | ".join(f"{r['by_category'][c]['mrr']:.2f}" for c in cats) + " |")


def main():
    keys = sys.argv[1:] or list(RUNGS)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for key in keys:
        results[key] = run_rung(key)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(dict(sorted(results.items())), indent=2))
    print_table(dict(sorted(results.items())))


if __name__ == "__main__":
    main()
