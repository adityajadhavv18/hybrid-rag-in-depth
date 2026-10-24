"""Retrievers the ladder combines. Each returns a ranked list of Hits; every stage records its time in STAGE_MS.

  dense_search     question -> bge vector     -> nearest chunks by cosine        (the "meaning" librarian)
  sparse_search    question -> sparse vector  -> chunks sharing weighted words   (the "keyword" librarian)
                   BM25 / SPLADE / miniCOIL depending on `using`; code-tagged indexes also tag the question
  colbert_search   question tokens vs every chunk's tokens (MaxSim) over the whole collection
  colbert_rescore  the same scoring, but only over a given candidate list (re-ranks it)
  rrf_fuse         merge ranked lists by Reciprocal Rank Fusion (ranks only, never raw scores)
  score_fuse       merge by normalised scores instead (rung 5 compares the two)
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from chunks import EMBED_MODEL
from codes import with_codes
from ingest import (BM25, BM25_CODE, BM25_MODEL, BM25_NAMED, COLBERT_MODEL, COLBERT_NAMED, COLLECTION, DENSE,
                    MINICOIL_MODEL, MINICOIL_NAMED, QDRANT_URL, SPLADE_MODEL, SPLADE_NAMED, point_id)

STAGE_MS: dict[str, list[float]] = defaultdict(list)  # stage name -> ms per call; the ladder reads and clears it
SPARSE_MODEL = {BM25: BM25_MODEL, BM25_CODE: BM25_MODEL, BM25_NAMED: BM25_MODEL,
                SPLADE_NAMED: SPLADE_MODEL, MINICOIL_NAMED: MINICOIL_MODEL}
CODE_TAGGED = {BM25_CODE, BM25_NAMED, SPLADE_NAMED, MINICOIL_NAMED}  # indexed with codes.py tokens


@contextmanager
def timed(stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE_MS[stage].append((time.perf_counter() - start) * 1000)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float  # dense: cosine; bm25: IDF x TF; colbert: MaxSim sum; fused: fusion points
    ranks: dict[str, int] = field(default_factory=dict)  # fused only: rank in each list that found it


@cache  # load models and connect once, not on every question
def _dense() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


@cache
def _sparse(model: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model)  # query side needs no avg_len: length correction is document-only


@cache
def _colbert() -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(COLBERT_MODEL)


@cache
def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _hits(points) -> list[Hit]:
    return [Hit(p.payload["chunk_id"], p.payload["doc_id"], p.payload["page"], p.payload["text"], p.score)
            for p in points]


def dense_search(question: str, k: int, using: str = DENSE) -> list[Hit]:
    with timed(using):
        vector = next(_dense().query_embed(question)).tolist()
        return _hits(_qdrant().query_points(COLLECTION, query=vector, using=using, limit=k).points)


def sparse_search(question: str, k: int, using: str = BM25) -> list[Hit]:
    with timed(using):
        sp = next(_sparse(SPARSE_MODEL[using]).query_embed(with_codes(question) if using in CODE_TAGGED else question))
        query = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
        return _hits(_qdrant().query_points(COLLECTION, query=query, using=using, limit=k).points)


def _colbert_query(question: str) -> list[list[float]]:
    return next(_colbert().query_embed(question)).tolist()


def colbert_search(question: str, k: int) -> list[Hit]:
    with timed("colbert"):
        return _hits(_qdrant().query_points(COLLECTION, query=_colbert_query(question), using=COLBERT_NAMED,
                                            limit=k).points)


def colbert_rescore(question: str, candidates: list[Hit]) -> list[Hit]:
    """Re-rank only these candidates by ColBERT MaxSim: Qdrant scores the chunks whose ids we pass."""
    with timed("colbert_rescore"):
        ids = [point_id(h.chunk_id) for h in candidates]
        points = _qdrant().query_points(
            COLLECTION, query=_colbert_query(question), using=COLBERT_NAMED, limit=len(ids),
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)])).points
        return _hits(points)


def rrf_fuse(ranked: dict[str, list[Hit]], k: int = 60, weights: dict[str, float] | None = None) -> list[Hit]:
    """Each list gives a chunk weight / (k + rank) points; add them up and sort. k=60 is the RRF paper's default
    (Cormack et al., 2009); rung 5 tunes it. A chunk ranked well by both lists beats one ranked #1 by one list:
        #3 in both: 2/63 = 0.032      #1 in one only: 1/61 = 0.016
    Smaller k widens the gap between rank 1 and rank 10, so a confident single-list #1 survives more often.
    """
    weights = weights or {}
    with timed("fuse"):
        points: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        by_id: dict[str, Hit] = {}
        for name, hits in ranked.items():
            for rank, h in enumerate(hits, start=1):
                points[h.chunk_id] += weights.get(name, 1.0) / (k + rank)
                ranks[h.chunk_id][name] = rank
                by_id.setdefault(h.chunk_id, h)
        # ties go to the better single rank, then chunk_id, so results are stable run to run
        order = sorted(points, key=lambda c: (-points[c], min(ranks[c].values()), c))
        return [Hit(by_id[c].chunk_id, by_id[c].doc_id, by_id[c].page, by_id[c].text, points[c], ranks[c])
                for c in order]


def score_fuse(ranked: dict[str, list[Hit]], weights: dict[str, float]) -> list[Hit]:
    """Weighted sum of scores, each list min-max scaled to 0..1 first (cosine and BM25 live on different scales).
    Unlike RRF this keeps HOW MUCH better #1 was than #2. A chunk missing from a list gets 0 from it."""
    with timed("fuse"):
        points: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        by_id: dict[str, Hit] = {}
        for name, hits in ranked.items():
            if not hits:
                continue
            hi, lo = hits[0].score, hits[-1].score
            for rank, h in enumerate(hits, start=1):
                points[h.chunk_id] += weights.get(name, 1.0) * ((h.score - lo) / (hi - lo) if hi > lo else 1.0)
                ranks[h.chunk_id][name] = rank
                by_id.setdefault(h.chunk_id, h)
        order = sorted(points, key=lambda c: (-points[c], min(ranks[c].values()), c))
        return [Hit(by_id[c].chunk_id, by_id[c].doc_id, by_id[c].page, by_id[c].text, points[c], ranks[c])
                for c in order]
