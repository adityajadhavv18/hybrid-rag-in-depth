"""Fixed-size chunking, page by page: the baseline every rung uses until rung 8 tests structure-aware chunks.

Each page of data/manufacturing/text/<doc>.txt (pages split by "\\f") is cut into windows of CHUNK_TOKENS
with OVERLAP_TOKENS shared between neighbours. A chunk never runs from one page into the next, so every chunk
carries one page number (useful for citations later).

Why 300 / 50: a manual entry (one fault code with causes and fixes) is roughly 60-400 tokens, so 300 holds a
whole short entry or most of a long one without mixing in many neighbours. The overlap of 50 is longer than
the longest golden evidence quote (42 tokens): any span up to the overlap length always lands whole inside at
least one window, so no quote is ever cut in half and every rung can reach evidence recall 1.0.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tokenizers import Tokenizer

TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "manufacturing" / "text"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 300  # measured in bge tokens: bge-small reads at most 512
OVERLAP_TOKENS = 50


# How each manual is named in its chunks' INDEX text (rung r2b): "Manual: <name>" goes on top of every chunk before
# embedding / BM25, so a machine name in the question points at that manual's chunks instead of only at the few
# pages (cover, contents) that happen to print the name. The stored chunk text, which the LLM reads, is unchanged.
MANUAL_NAMES = {
    "danfoss_fc202": "Danfoss VLT AQUA Drive FC 202 frequency converter, operating guide",
    "franklin_subdrive": "Franklin Electric SubDrive75/100/150 constant pressure controller, installation manual",
    "goulds_aquavar_im043": "Goulds Aquavar variable speed pump controller (V120 software), IM043 manual",
    "goulds_aquavar_solo2": "Goulds Aquavar SOLO2 constant pressure controller, IM260 manual",
    "yaskawa_ga500_troubleshooting": "Yaskawa GA500 AC microdrive, maintenance and troubleshooting manual",
}


def named(chunk: "Chunk") -> str:
    return f"Manual: {MANUAL_NAMES[chunk.doc_id]}\n{chunk.text}"


@dataclass
class Chunk:
    chunk_id: str  # "danfoss_fc202::p046::01"
    doc_id: str  # becomes context_ids in PipelineOutput
    page: int
    text: str
    n_tokens: int


@cache
def tokenizer() -> Tokenizer:
    tok = Tokenizer.from_pretrained(EMBED_MODEL)
    tok.no_truncation()  # we need every token of a page, not just the first 512
    return tok


def chunk_page(doc_id: str, page: int, text: str) -> list[Chunk]:
    offsets = tokenizer().encode(text, add_special_tokens=False).offsets  # (start_char, end_char) per token
    chunks, step = [], CHUNK_TOKENS - OVERLAP_TOKENS
    for i, start in enumerate(range(0, len(offsets), step)):
        end = min(start + CHUNK_TOKENS, len(offsets))
        # slice the ORIGINAL text by character offsets: bge's tokenizer lowercases, so never decode its tokens
        chunks.append(Chunk(f"{doc_id}::p{page:03d}::{i:02d}", doc_id, page,
                            text[offsets[start][0]: offsets[end - 1][1]], end - start))
        if end == len(offsets):
            break
    return chunks


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(TEXT_DIR.glob("*.txt")):
        for page, text in enumerate(path.read_text().split("\f"), start=1):
            if text.strip():
                chunks.extend(chunk_page(path.stem, page, text))
    return chunks
