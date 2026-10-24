"""Index the manuals once: every chunk stored in Qdrant with one vector per retriever the ladder compares.

  dense            bge-small "meaning" vector (384 numbers)          -> finds paraphrases, weak on exact codes
  bm25             sparse "keyword" vector (word id -> weight)       -> exact words, misses paraphrases
  bm25_code        bm25 over the code-tagged text (codes.py, r2)     -> "WARNING 23" also indexed as w23
  dense_named      dense over "Manual: <name>" + chunk (r2b)
  bm25_named       bm25_code over the named text (r2b)
  splade_named     SPLADE: learned sparse, adds related words (r3)   -> "overheat" also lights up "temperature"
  minicoil_named   miniCOIL: BM25 whose word weights know the word's sense (r3)
  colbert_named    ColBERT: one 96-number vector PER TOKEN, scored token by token (MaxSim) (r4)

Batches are small (8) because this runs on an 8 GB laptop: large batches of 300-token chunks swapped it to a halt.

Rungs only change how we QUERY these vectors, so every rung searches the exact same chunks. Model outputs are
cached by text hash (data/manufacturing/cache/), so re-indexing only embeds chunks whose text changed.

Run from hybrid_rag_in_depth/ (Qdrant must be running: docker start qdrant):
    uv run manuals/ingest.py
"""

import hashlib
import pickle
import statistics
import time
import uuid
from pathlib import Path

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from fastembed.common.utils import remove_non_alphanumeric
from qdrant_client import QdrantClient, models

from chunks import EMBED_MODEL, load_chunks, named
from codes import with_codes

QDRANT_URL = "http://localhost:6333"
COLLECTION = "manuals_fixed"
BM25_MODEL = "Qdrant/bm25"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
MINICOIL_MODEL = "Qdrant/minicoil-v1"
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
DENSE, BM25, BM25_CODE = "dense", "bm25", "bm25_code"
DENSE_NAMED, BM25_NAMED = "dense_named", "bm25_named"
SPLADE_NAMED, MINICOIL_NAMED, COLBERT_NAMED = "splade_named", "minicoil_named", "colbert_named"
SPARSE_NAMES = {BM25, BM25_CODE, BM25_NAMED, SPLADE_NAMED, MINICOIL_NAMED}
CACHE = Path(__file__).resolve().parent.parent / "data" / "manufacturing" / "cache"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))  # Qdrant ids must be ints or UUIDs


def bm25_tokens(model, text: str) -> list[str]:
    """The exact words BM25 indexes: punctuation split, lowercased, stopwords dropped, stemmed."""
    return model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text)))


def avg_words(texts: list[str]) -> float:
    """BM25-style models discount words in long chunks relative to the AVERAGE length (FastEmbed assumes 256);
    measure ours so the length correction is centred on our real chunks."""
    probe = SparseTextEmbedding(BM25_MODEL)
    return statistics.mean(len(bm25_tokens(probe.model, t)) for t in texts)


def cached(tag: str, texts: list[str], embed) -> list:
    """embed(list_of_texts) -> outputs, cached per text under data/manufacturing/cache/<tag>.pkl."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag.replace('/', '_')}.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in cache))
    if todo:
        start = time.perf_counter()
        for i, out in zip(todo, embed([texts[i] for i in todo])):
            cache[keys[i]] = out
        path.write_bytes(pickle.dumps(cache))
        print(f"     embedded {len(todo)} texts for {tag} in {time.perf_counter() - start:.0f}s")
    return [cache[k] for k in keys]


def sparse(v) -> models.SparseVector:
    return models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())


def main():
    chunks = load_chunks()
    plain = [c.text for c in chunks]
    coded = [with_codes(t) for t in plain]
    named_ = [named(c) for c in chunks]
    named_coded = [with_codes(t) for t in named_]
    print(f"1. chunks: {len(chunks)} from {len({c.doc_id for c in chunks})} manuals")

    dense_model = TextEmbedding(EMBED_MODEL)
    embed_dense = lambda ts: dense_model.passage_embed(ts, batch_size=32)
    bm25 = lambda ts: list(SparseTextEmbedding(BM25_MODEL, avg_len=avg_words(ts)).embed(ts))  # fast: no cache
    minicoil_len = avg_words(named_coded)
    vectors = {
        DENSE: cached(f"dense_{EMBED_MODEL}", plain, embed_dense),
        DENSE_NAMED: cached(f"dense_{EMBED_MODEL}", named_, embed_dense),
        BM25: bm25(plain),
        BM25_CODE: bm25(coded),
        BM25_NAMED: bm25(named_coded),
        SPLADE_NAMED: cached(f"splade_{SPLADE_MODEL}", named_coded,
                             lambda ts: SparseTextEmbedding(SPLADE_MODEL).embed(ts, batch_size=8)),
        MINICOIL_NAMED: cached(f"minicoil_{MINICOIL_MODEL}_{minicoil_len:.1f}", named_coded,
                               lambda ts: SparseTextEmbedding(MINICOIL_MODEL, avg_len=minicoil_len).embed(ts, batch_size=8)),
        COLBERT_NAMED: cached(f"colbert_{COLBERT_MODEL}", named_,
                              lambda ts: LateInteractionTextEmbedding(COLBERT_MODEL).embed(ts, batch_size=8)),
    }
    print(f"2. vectors: {', '.join(vectors)}")

    client = QdrantClient(url=QDRANT_URL, timeout=300)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = len(vectors[DENSE][0])
    client.create_collection(
        COLLECTION,
        vectors_config={
            DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            DENSE_NAMED: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            # ColBERT: many vectors per chunk, scored by MaxSim (each question token takes its best-matching
            # chunk token, then sum). m=0 skips the HNSW graph: with 1.5k chunks an exact scan is fast enough.
            COLBERT_NAMED: models.VectorParams(
                size=len(vectors[COLBERT_NAMED][0][0]), distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                hnsw_config=models.HnswConfigDiff(m=0), datatype=models.Datatype.FLOAT16),
        },
        # BM25-style vectors hold only the term-frequency half of the formula; Modifier.IDF makes Qdrant add the
        # other half (how RARE each word is across all chunks) at query time. SPLADE weights are complete already.
        sparse_vectors_config={n: models.SparseVectorParams(
            modifier=None if n == SPLADE_NAMED else models.Modifier.IDF) for n in SPARSE_NAMES},
    )
    client.upload_points(
        COLLECTION,
        # a generator, not a list: ColBERT's per-token vectors as Python floats would need >1 GB at once
        points=(
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={name: sparse(vs[i]) if name in SPARSE_NAMES else vs[i].tolist() for name, vs in vectors.items()},
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page, "text": c.text},
            )
            for i, c in enumerate(chunks)
        ),
        batch_size=32,
        wait=True,
    )
    print(f"3. stored: {client.count(COLLECTION, exact=True).count} points in '{COLLECTION}'")


if __name__ == "__main__":
    main()
