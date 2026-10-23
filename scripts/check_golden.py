"""Check the manufacturing golden sets against the cleaned corpus.

    uv run scripts/check_golden.py

For both splits (tune, test): items load under the harness schema, every source_doc exists, and each
evidence fragment
  - is in its source doc, matched the way the harness matches it (whitespace collapsed, lowercase),
  - appears only once in the whole corpus (otherwise a wrong chunk could count as a hit),
  - does not cross a page break (chunks never span pages, so it could never be retrieved whole).
Prints the mix by category (the id prefix), question type and doc.
"""

import re
import sys
from collections import Counter
from pathlib import Path

from eval_harness import load_golden

TEXT = Path(__file__).resolve().parent.parent / "data" / "manufacturing" / "text"
SPLITS = ["golden_manufacturing_tune_v1.jsonl", "golden_manufacturing_test_v1.jsonl"]


def norm(text: str) -> str:  # same as eval_harness.metrics._norm, but keeps page breaks visible
    return re.sub(r"[ \t\n\r\v]+", " ", text).strip().lower()


def main() -> None:
    raw = {p.stem: p.read_text() for p in sorted(TEXT.glob("*.txt"))}
    corpus = {doc: norm(text) for doc, text in raw.items()}
    problems = 0
    for name in SPLITS:
        golden = load_golden(name, known_docs=set(corpus))
        print(f"\n{name}: {len(golden)} items")
        print("  by category:", dict(Counter(g.id.split("-")[0] for g in golden)))
        print("  by type:    ", dict(Counter(g.question_type for g in golden)))
        print("  by doc:     ", dict(Counter(g.source_doc or "(none)" for g in golden)))
        for g in golden:
            for frag in (f for f in re.split(r" \.\.\. | / ", g.evidence or "") if f.strip()):
                f = norm(frag)
                hits = {doc: text.count(f) for doc, text in corpus.items() if f in text}
                if g.source_doc not in hits:
                    # same text with page breaks treated as spaces: found only across a page break?
                    flat = re.sub(r"\s+", " ", raw[g.source_doc]).lower()
                    why = "crosses a page break" if f in flat else "not found"
                    print(f"  ❌ {g.id}: evidence {why} in {g.source_doc}: {frag[:70]!r}")
                    problems += 1
                elif sum(hits.values()) > 1:
                    print(f"  ❌ {g.id}: evidence appears {sum(hits.values())}x ({hits}): {frag[:60]!r}")
                    problems += 1
    print(f"\n{'✅ all evidence unique and verbatim' if not problems else f'❌ {problems} problem(s)'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
