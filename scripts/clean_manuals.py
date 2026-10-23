"""Turn the raw manual PDFs into clean text for the manufacturing corpus.

    uv run scripts/clean_manuals.py

Reads data/manufacturing/raw/*.pdf, writes data/manufacturing/text/<doc_id>.txt (pages separated by
a form feed "\\f", so a chunker can still tell which page a line came from) and manuals.json (what
each document is). Cleaning: keep only the English pages, drop running headers/footers and page
numbers (by where they sit on the page), fix PDF ligatures, re-attach bullets to their text, re-join
sentences the PDF wrapped across lines, and rebuild Franklin's sideways fault tables row by row.
"""

import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent / "data" / "manufacturing"
RAW, TEXT = ROOT / "raw", ROOT / "text"

# doc_id -> what the manual is, which pages are English (1-based, inclusive), and its page margins:
# a line ending above `top` or starting below `bottom` (PDF points) is a running header/footer or page
# number. Measured per manual: Yaskawa's real "Code / Name / Causes" table header sits as high on the
# page as Danfoss's running header, and SOLO2 body text runs to the top edge, so one rule can't fit all.
MANUALS = {
    "franklin_subdrive": {"maker": "Franklin Electric", "machine": "SubDrive75/100/150 constant pressure controller (submersible well pump)", "pages": None, "margins": (0, 800)},
    "goulds_aquavar_solo2": {"maker": "Goulds Water Technology (Xylem)", "machine": "Aquavar SOLO2 constant pressure controller", "pages": (1, 20), "margins": (0, 760)},
    "goulds_aquavar_im043": {"maker": "Goulds Water Technology (Xylem)", "machine": "Aquavar pump-mounted variable speed controller, V120 software", "pages": None, "margins": (40, 740)},
    "danfoss_fc202": {"maker": "Danfoss", "machine": "VLT AQUA Drive FC 202 (0.25-90 kW)", "pages": None, "margins": (66, 805)},
    "yaskawa_ga500_troubleshooting": {"maker": "Yaskawa", "machine": "GA500 industrial AC microdrive", "pages": None, "margins": (40, 755)},
}

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "­": "", " ": " "}
BULLET = re.compile(r"^[•\-–▪●]\s*$")  # a bullet the PDF left alone on its line
# Danfoss prints its chapter number in the side margin ("7", "7 7"); it comes out as the page's last lines
PAGE_TAIL = re.compile(r"^\d{1,3}( \d{1,3})?$")
# a wrapped sentence continues with a plain lowercase word; fault codes like oC, bAT, ov, bb must
# stay on their own line, so 1-2 letter words and mixed case never count
WRAPPED = re.compile(r"^[a-z]{3,}\b")


def clean_page(page, margins: tuple[float, float]) -> tuple[str, int]:
    """Clean text of one page, and how many margin lines were dropped."""
    top, bottom = margins
    all_lines = raw_lines(page)
    lines = [text for (_, y0, _, y1), text in all_lines if y1 > top and y0 < bottom]
    dropped = len(all_lines) - len(lines)
    while lines and PAGE_TAIL.match(lines[-1]):
        lines.pop()
    out: list[str] = []
    for ln in lines:
        if out and BULLET.match(out[-1]):  # "•" + "Check the fan." -> "• Check the fan."
            out[-1] = f"{out[-1][0]} {ln}"
        elif out and len(out[-1]) >= 25 and out[-1][-1] not in ".:;!?)" and WRAPPED.match(ln):
            out[-1] = f"{out[-1]} {ln}"  # sentence wrapped onto the next line
        else:
            out.append(ln)
    return "\n".join(out), dropped


def raw_lines(page) -> list[tuple[tuple, str]]:
    """(bbox, text) per text line, in the PDF's reading order."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"])
            for bad, good in LIGATURES.items():
                text = text.replace(bad, good)
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                out.append((line["bbox"], text))
    return out


# Franklin's two quick-reference tables are printed sideways. Plain text keeps the word order but runs
# the cause and action columns together, so rebuild each row from line positions instead: on a
# sideways page the y coordinate says which column a line is in, and lines come out row by row.
# page -> (table area: x0 below this, column edges by y0, column labels)
FRANKLIN_TABLES = {
    23: (398, (120, 190, 360), ("Number of flashes", "Fault", "Possible cause", "Corrective action")),
    24: (458, (140, 225, 390), ("Condition", "Indicator lights", "Possible cause", "Corrective action")),
}


def franklin_table(page, page_no: int) -> str:
    x_max, edges, labels = FRANKLIN_TABLES[page_no]
    heading, rows = [], []
    row = None
    for (x0, y0, _, _), text in raw_lines(page):
        if y0 > 800:  # page number
            continue
        if not (x0 < x_max and (page_no == 24 or x0 > 120)):  # outside the table: title and intro text
            heading.append(text)
            continue
        col = sum(y0 >= e for e in edges)
        if col <= 1 and (row is None or row[2] or row[3]):  # first two columns start a new row ...
            # ... and a row that only fills the second column shares the first column with the row above
            row = [row[0] if (row is not None and col == 1) else [], [], [], []]
            rows.append(row)
        if text.startswith("-") or not row[col]:
            row[col].append(text.lstrip("- ").strip() if col >= 2 else text)
        else:
            row[col][-1] += " " + text  # a cell's text wrapped onto the next line
    parts = ["\n".join(heading)]
    for first, second, causes, actions in rows:
        parts.append("\n".join([
            f"{labels[0]}: {' '.join(first)} | {labels[1]}: {' '.join(second)}",
            f"{labels[2]}:", *(f"- {c}" for c in causes),
            f"{labels[3]}:", *(f"- {a}" for a in actions),
        ]))
    return "\n\n".join(parts)


def main() -> None:
    TEXT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for doc_id, meta in MANUALS.items():
        doc = pymupdf.open(RAW / f"{doc_id}.pdf")
        first, last = meta["pages"] or (1, doc.page_count)
        pages, dropped = [], 0
        for n in range(first, last + 1):
            if doc_id == "franklin_subdrive" and n in FRANKLIN_TABLES:
                pages.append(franklin_table(doc[n - 1], n))
            else:
                text, d = clean_page(doc[n - 1], meta["margins"])
                pages.append(text)
                dropped += d
        text = "\f".join(pages)
        (TEXT / f"{doc_id}.txt").write_text(text)
        manifest.append({"doc_id": doc_id, "maker": meta["maker"], "machine": meta["machine"],
                         "pages_kept": [first, last], "chars": len(text)})
        print(f"{doc_id:32} pages {first}-{last:<4} chars {len(text):>7}  margin lines dropped: {dropped}")
    (ROOT / "manuals.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
