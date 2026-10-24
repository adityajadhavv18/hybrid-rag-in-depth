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

from ingest import BM25_CODE, BM25_NAMED, DENSE_NAMED
from retrieve import STAGE_MS, dense_search, rrf_fuse, sparse_search

TUNE = "golden_manufacturing_tune_v1.jsonl"
TOP_K = 5  # chunks the LLM will read
CANDIDATES = 20  # how deep each retriever's list goes into fusion
RESULTS = Path(__file__).resolve().parent / "results" / "ladder_tune.json"

RUNGS = {
    "r0": ("dense only", lambda q, k: dense_search(q, k)),
    "r1": ("dense + BM25, RRF", lambda q, k: rrf_fuse({"dense": dense_search(q, CANDIDATES),
                                                       "bm25": sparse_search(q, CANDIDATES)})[:k]),
    "r2": ("+ code-aware BM25", lambda q, k: rrf_fuse({"dense": dense_search(q, CANDIDATES),
                                                      "bm25": sparse_search(q, CANDIDATES, using=BM25_CODE)})[:k]),
    "r2b": ("+ manual name on every chunk", lambda q, k: rrf_fuse({
        "dense": dense_search(q, CANDIDATES, using=DENSE_NAMED),
        "bm25": sparse_search(q, CANDIDATES, using=BM25_NAMED)})[:k]),
}


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

    def pipeline(question: str) -> PipelineOutput:
        hits = search(question, TOP_K)
        return PipelineOutput(answer="", contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits])

    run = evaluate(pipeline, arch=f"manuals_{key}", golden_set=TUNE, use_llm_judge=False, compare_to=None)
    stages = {s: round(statistics.median(ms), 1) for s, ms in STAGE_MS.items()}

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
    lat = sorted(q.latency_ms for q in run.questions)
    result = {
        "rung": key, "label": label, "questions": len(rows),
        "recall5": mean(rows, "recall5"), "mrr": mean(rows, "mrr"), "ndcg5": mean(rows, "ndcg5"),
        "recall20": mean(rows, "recall20"),
        "latency_p50_ms": round(statistics.median(lat), 1), "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "stage_p50_ms": stages, "cost_per_query_usd": 0.0,
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
