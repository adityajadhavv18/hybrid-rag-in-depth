"""Turn the raw manual PDFs into clean text for the manufacturing corpus.

    uv run scripts/clean_manuals.py

Reads data/manufacturing/raw/*.pdf, writes data/manufacturing/text/<doc_id>.txt (pages separated by
a form feed "\\f", so a chunker can still tell which page a line came from) and manuals.json (what
each document is). Cleaning: keep only the English pages, drop lines repeated on most pages (running
headers/footers) and bare page numbers, fix PDF ligatures, re-attach bullets to their text, and
re-join sentences the PDF wrapped across lines.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent / "data" / "manufacturing"
RAW, TEXT = ROOT / "raw", ROOT / "text"

# doc_id -> what the manual is, and which pages are English (1-based, inclusive)
MANUALS = {
    "franklin_subdrive": {"maker": "Franklin Electric", "machine": "SubDrive75/100/150 constant pressure controller (submersible well pump)", "pages": None},
    "goulds_aquavar_solo2": {"maker": "Goulds Water Technology (Xylem)", "machine": "Aquavar SOLO2 constant pressure controller", "pages": (1, 20)},
    "goulds_aquavar_im043": {"maker": "Goulds Water Technology (Xylem)", "machine": "Aquavar pump-mounted variable speed controller, V120 software", "pages": None},
    "goulds_im154": {"maker": "Goulds Water Technology (Xylem)", "machine": "Aquavar variable speed pump control, new sizes addendum", "pages": None},
    "danfoss_fc202": {"maker": "Danfoss", "machine": "VLT AQUA Drive FC 202 (0.25-90 kW)", "pages": None},
    "yaskawa_ga500_troubleshooting": {"maker": "Yaskawa", "machine": "GA500 industrial AC microdrive", "pages": None},
}

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "­": "", " ": " "}
BULLET = re.compile(r"^[•\-–▪●]\s*$")  # a bullet the PDF left alone on its line
# a wrapped sentence continues with a plain lowercase word; fault codes like oC, bAT, ov, bb must
# stay on their own line, so 1-2 letter words and mixed case never count
WRAPPED = re.compile(r"^[a-z]{3,}\b")


def page_lines(page) -> list[str]:
    text = page.get_text()
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]


def clean_page(lines: list[str], headers: set[str], page_no: int) -> str:
    lines = [ln for ln in lines if ln not in headers]
    # bare page number as the first or last line
    for idx in (0, -1):
        if lines and lines[idx].isdigit() and abs(int(lines[idx]) - page_no) <= 5:
            lines.pop(idx)
    out: list[str] = []
    for ln in lines:
        if out and BULLET.match(out[-1]):  # "•" + "Check the fan." -> "• Check the fan."
            out[-1] = f"{out[-1][0]} {ln}"
        elif out and len(out[-1]) >= 25 and out[-1][-1] not in ".:;!?)" and WRAPPED.match(ln):
            out[-1] = f"{out[-1]} {ln}"  # sentence wrapped onto the next line
        else:
            out.append(ln)
    return "\n".join(out)


def main() -> None:
    TEXT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for doc_id, meta in MANUALS.items():
        doc = pymupdf.open(RAW / f"{doc_id}.pdf")
        first, last = meta["pages"] or (1, doc.page_count)
        pages = {n: page_lines(doc[n - 1]) for n in range(first, last + 1)}
        # running headers/footers: lines in the top/bottom 3 of a page that recur on at least 40% of
        # pages (bare numbers never count: they are fault flashes and DIP switch numbers too)
        counts = Counter(ln for lines in pages.values() for ln in set(lines[:3] + lines[-3:]))
        headers = {ln for ln, c in counts.items()
                   if c >= 0.4 * len(pages) and len(pages) > 4 and not ln.isdigit() and 3 < len(ln) < 120}
        text = "\f".join(clean_page(lines, headers, n) for n, lines in pages.items())
        (TEXT / f"{doc_id}.txt").write_text(text)
        manifest.append({"doc_id": doc_id, **{k: v for k, v in meta.items() if k != "pages"},
                         "pages_kept": [first, last], "chars": len(text), "headers_removed": sorted(headers)})
        print(f"{doc_id:32} pages {first}-{last:<4} chars {len(text):>7}  headers removed: {len(headers)}")
    (ROOT / "manuals.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
