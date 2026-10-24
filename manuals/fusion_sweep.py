"""Rung 5 sweep: how should the retrievers' lists be merged? TUNE set only.

Each retriever's top-40 list is fetched ONCE per question; every fusion setting is then replayed on those saved
lists in pure Python, so the whole grid takes seconds. Compared:
  RRF            k in {1, 5, 10, 20, 40, 60} x candidates per list {10, 20, 40} x BM25 / SPLADE weights {1, 1.5, 2}
  score fusion   min-max scaled scores, dense weight 1, BM25 / SPLADE weights {0.5, 1, 1.5} x candidates
Scored by hit@5 and MRR (first evidence chunk), the numbers the ladder table uses.

With only 23 answerable tune questions, one question moves MRR by about 0.04: prefer a setting that sits on a
broad plateau over a lone peak, and confirm on the test split only at the very end.

Run from hybrid_rag_in_depth/:
    uv run manuals/fusion_sweep.py
"""

import itertools
import statistics

from eval_harness import load_golden

from ingest import BM25_NAMED, DENSE_NAMED, SPLADE_NAMED
from ladder import TOP_K, TUNE, first_rank, fragments
from retrieve import dense_search, rrf_fuse, score_fuse, sparse_search

DEPTH = 40
# the lists to merge: the r3 winner's retrievers (r3c); dense keeps weight 1, the two keyword lists are swept
LISTS = {
    "dense": lambda q, k: dense_search(q, k, using=DENSE_NAMED),
    "bm25": lambda q, k: sparse_search(q, k, using=BM25_NAMED),
    "splade": lambda q, k: sparse_search(q, k, using=SPLADE_NAMED),
}


def evaluate(fused_per_question: list[list], golden: list) -> tuple[float, float]:
    ranks = [first_rank(fragments(g), [h.text for h in fused[:TOP_K]]) for g, fused in zip(golden, fused_per_question)]
    return statistics.mean(1.0 if r else 0.0 for r in ranks), statistics.mean(1 / r if r else 0.0 for r in ranks)


def main():
    golden = [g for g in load_golden(TUNE) if g.question_type != "no_answer"]
    saved = [{name: fn(g.question, DEPTH) for name, fn in LISTS.items()} for g in golden]
    rows = []
    for k, cand, wb, ws in itertools.product([1, 5, 10, 20, 40, 60], [10, 20, 40], [1.0, 1.5, 2.0], [1.0, 1.5, 2.0]):
        fused = [rrf_fuse({n: hits[:cand] for n, hits in lists.items()}, k=k, weights={"bm25": wb, "splade": ws})
                 for lists in saved]
        rows.append((f"RRF k={k:<2} cand={cand:<2} bm25={wb} splade={ws}", *evaluate(fused, golden)))
    for wb, ws, cand in itertools.product([0.5, 1.0, 1.5], [0.5, 1.0, 1.5], [10, 20, 40]):
        fused = [score_fuse({n: hits[:cand] for n, hits in lists.items()}, weights={"bm25": wb, "splade": ws})
                 for lists in saved]
        rows.append((f"score cand={cand:<2} bm25={wb} splade={ws}", *evaluate(fused, golden)))

    rows.sort(key=lambda r: (-r[2], -r[1]))
    # how flat is the top? count settings within one question (0.04 MRR) of the best
    best = rows[0][2]
    print(f"settings within 0.04 MRR of the best ({best:.2f}): {sum(r[2] >= best - 0.04 for r in rows)}")
    by_family = {}
    for name, hit, mrr in rows:
        fam = name.split(" cand")[0] if name.startswith("RRF") else "score"
        by_family.setdefault(fam, (hit, mrr, name))
    print("best per family:", {f: f"MRR {m:.2f} hit {h:.2f}" for f, (h, m, _) in by_family.items()})
    print(f"{len(rows)} settings on {len(golden)} answerable tune questions (sorted by MRR)\n")
    print("| setting | hit@5 | MRR |\n|---|---|---|")
    for name, hit, mrr in rows[:15]:
        print(f"| {name} | {hit:.2f} | {mrr:.2f} |")
    print("| ... | | |")
    for name, hit, mrr in rows:
        if name.startswith("RRF k=60 cand=20 bm25=1.0 splade=1.0"):
            print(f"| {name} (current) | {hit:.2f} | {mrr:.2f} |")


if __name__ == "__main__":
    main()
