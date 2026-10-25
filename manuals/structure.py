"""Structure-aware chunking (rung 8): one fault entry = one chunk, instead of a cut every 300 tokens.

Fixed windows cut long entries in two, and the second half never names its code: "The drive capacity is too
small." sits in a chunk without "oL2", so a question about oL2 never finds it. Here each manual's fault section
is split where an entry STARTS (its code heading), so the code, its name, every cause and every fix stay together.

Per manual, a SECTION (the fault list, found by its first and last heading) and a pattern for where an entry starts:

  yaskawa   sections 2.4-2.7    table header "Code / Name / Causes / Possible Solutions" + a code line ("oL2").
                                The header is reprinted on every page: header + a sentence = the same entry continuing.
  danfoss   7.4 -> 7.5          "WARNING 12, Torque limit" / "ALARM 14, ..." / "WARNING/ALARM 13, ..."
  solo2     RED LIGHT CODES     "F03" + "3 Blinks" (F01: "Constant Red"); the "(continued)" stub joins the entry
  franklin  whole manual        each "Number of flashes: n | Fault: ..." / "Condition: ..." block
  im043     Repair of Faults    its named headings ("Overtemp Motor", "Pressure Sensor Error", ...)

Entries that run across pages are joined, with the page clutter between them (running heads, the reprinted table
header) removed. An entry longer than MAX_ENTRY_TOKENS (bge-small reads 512) is cut into the usual 300/50 windows,
and every window after the first starts with "<code> <name> (continued)", so each piece still names its code.
Everything outside the fault sections (intro, wiring, specs, section intros) keeps the fixed 300/50 chunks, page
by page, so this rung measures only what entry-shaped chunks change.
"""

import re
from dataclasses import dataclass

from chunks import CHUNK_TOKENS, OVERLAP_TOKENS, TEXT_DIR, Chunk, chunk_page, tokenizer

MAX_ENTRY_TOKENS = 450  # whole entry up to here; leaves room for the "Manual: ..." line and code tokens under 512

YASKAWA_HEADER = "Code\nName\nCauses\nPossible Solutions\n"
# Codes listed in the Yaskawa code index (pages 33-37): tells a code line ("oL2", "PE1, PE2") from a page-top
# continuation that starts with a sentence. Codes with a digit ("CPF25", "Er-01") are recognised without the list.
YASKAWA_INDEX_PAGES = range(33, 38)
IM043_HEADINGS = ["Lack of Water", "Conveyor Control (Value Range Control Error)", "Overtemp Motor",
                  "Overtemp Inverter", "Over Voltage", "Under Voltage", "Overloaded", "Earth Fault (Short Circuit)",
                  "Pressure Sensor Error", "Error 1 through Error 8", "Inverter Locked"]


@dataclass
class Entry:
    doc_id: str
    page: int  # where the entry starts
    code: str  # "oL2", "ALARM 14", "F03", "6 flashes", "Overtemp Motor"
    name: str  # "Drive Overload"
    text: str


def page_at(text: str, pos: int) -> int:
    return text.count("\f", 0, pos) + 1


# ---------------------------------------------------------------- where entries start, per manual
# Each returns (section_start, section_end, [(entry_start, code, name)], [stop positions]): an entry runs until
# the next entry start or stop. Stops are headings inside a section that are not entries (a new sub-section).

def _yaskawa(t: str):
    pages = t.split("\f")
    known = set()
    for page in YASKAWA_INDEX_PAGES:
        lines = pages[page - 1].split("\n")
        known |= {m.group(1) for line in lines
                  if (m := re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9\-]{0,7})(?: \([0-9A-F]{4}\))?\s*", line))}
    known -= {"Alarm", "Fault", "Name", "Type", "Flashing", "Code"}

    def is_code_line(line: str) -> bool:
        tokens = [x for x in re.split(r",\s*", line.strip()) if x]
        return bool(tokens) and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{0,7}", x) and
                                    (re.search(r"\d", x) or x in known) for x in tokens)

    start = t.index("2.4\nFault\n")
    end = t.index("2.8\nBackup Function", start)  # the first one is the contents page
    starts = []
    for m in re.finditer(re.escape(YASKAWA_HEADER), t[start:end]):
        pos = start + m.end()
        code, name = (t[pos:].split("\n", 2) + ["", ""])[:2]
        if is_code_line(code):
            starts.append((start + m.start(), code.strip().rstrip(","), name.strip()))
    stops = [start + m.start() for m in re.finditer(r"(?m)^2\.[4-7]\n", t[start:end])]
    return start, end, starts, stops


def _danfoss(t: str):
    start = t.index("7.4 List of Warnings and Alarms\nThe warning")  # not the contents page
    end = t.index("7.5 Troubleshooting\n", start)
    pat = re.compile(r"(?:^|(?<=\f))((?:WARNING/ALARM|WARNING|ALARM) \d+), ([^\n]+)", re.M)
    return start, end, [(m.start(), m.group(1), m.group(2).strip()) for m in pat.finditer(t, start, end)], []


def _solo2(t: str):
    start, end = t.index("RED LIGHT CODES"), t.index("8: INSULATION AND WINDING")
    pat = re.compile(r"^(F0\d)\n(?:\d Blinks|Constant\nRed)\n([^\n]+)", re.M)
    # "F03 / 3 Blinks / Sensor Fault (continued)" at a page top is the same entry carrying on, not a new one
    starts = [(m.start(), m.group(1), m.group(2).strip()) for m in pat.finditer(t, start, end)
              if not m.group(2).strip().endswith("(continued)")]
    return start, end, starts, []


def _franklin(t: str):
    pat = re.compile(r"^(?:Number of flashes: (\S+) \| Fault: |Condition: )([^\n]+)", re.M)
    starts = [(m.start(), f"{m.group(1)} flashes" if m.group(1) else "Condition", m.group(2).strip())
              for m in pat.finditer(t)]
    stops = [m.start() for m in re.finditer(r"\n\n|\f", t)]  # each row is one blank-line block
    return 0, len(t), starts, stops


def _im043(t: str):
    start = t.index("Repair of Faults and Errors\nLack of Water")
    end = t.index("AQUAVAR Controller Flow Chart", start)
    starts = [(m.start(), h, "") for h in IM043_HEADINGS
              for m in [re.compile(rf"(?:^|(?<=\f)){re.escape(h)}\n", re.M).search(t, start, end)]]
    return start, end, sorted(starts), []


SECTIONS = {
    "yaskawa_ga500_troubleshooting": _yaskawa,
    "danfoss_fc202": _danfoss,
    "goulds_aquavar_solo2": _solo2,
    "franklin_subdrive": _franklin,
    "goulds_aquavar_im043": _im043,
}

# Page clutter removed from INSIDE an entry when it runs across pages (running heads, reprinted table headers).
CLUTTER = {
    "yaskawa_ga500_troubleshooting": [r"\f\s*(?:Troubleshooting\n2\n)?", re.escape(YASKAWA_HEADER)],
    "danfoss_fc202": [r"\f", r"(?m)^Troubleshooting\n"],
    "goulds_aquavar_solo2": [r"\f\s*", r"Troubleshooting Error Codes \(continued\)\n",
                             r"RED LIGHT CODES\nParameter\nDisplay\nFault Code\nFlashes\nController Status\n"
                             r"Description\nThis information is to be used by professional installers or qualified "
                             r"personnel only\.\n", r"F03\n3 Blinks\nSensor Fault \(continued\)\n",
                             r"(?m)^7: TROUBLESHOOTING\n"],
    "franklin_subdrive": [r"\f"],
    "goulds_aquavar_im043": [r"\f", r"(?m)^(?:WARNING|hazardous|voltage|Repair of Faults and Errors)\n?$"],
}


def clean(doc_id: str, text: str) -> str:
    for pat in CLUTTER[doc_id]:
        text = re.sub(pat, "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def split_doc(doc_id: str, t: str) -> tuple[list[Entry], list[tuple[int, int]]]:
    """The manual's fault entries, plus the (start, end) spans of everything else (chunked the fixed way)."""
    start, end, starts, stops = SECTIONS[doc_id](t)
    cuts = sorted({s for s, _, _ in starts} | set(stops) | {start, end})
    info = {s: (code, name) for s, code, name in starts}
    entries, others = [], [(0, start), (end, len(t))]
    for a, b in zip(cuts, cuts[1:]):
        if a in info:
            text = clean(doc_id, t[a:b])
            if doc_id == "yaskawa_ga500_troubleshooting":  # the leading header is clutter too
                text = text.removeprefix(YASKAWA_HEADER.strip()).strip()
            entries.append(Entry(doc_id, page_at(t, a), *info[a], text))
        else:
            others.append((a, b))
    return entries, others


def entry_chunks(e: Entry, n: int) -> list[Chunk]:
    tok = tokenizer()
    base = f"{e.doc_id}::p{e.page:03d}::e{n:03d}"
    size = len(tok.encode(e.text, add_special_tokens=False).ids)
    if size <= MAX_ENTRY_TOKENS:
        return [Chunk(f"{base}::00", e.doc_id, e.page, e.text, size)]
    # too long: the usual windows over the entry, each later window told which entry it continues
    windows = chunk_page(e.doc_id, e.page, e.text)
    label = f"{e.code} {e.name}".strip()
    out = []
    for i, w in enumerate(windows):
        text = w.text if i == 0 else f"{label} (continued)\n{w.text}"
        out.append(Chunk(f"{base}::{i:02d}", e.doc_id, e.page, text,
                         len(tok.encode(text, add_special_tokens=False).ids)))
    return out


def load_struct_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(TEXT_DIR.glob("*.txt")):
        t = path.read_text()
        entries, others = split_doc(path.stem, t)
        for n, e in enumerate(entries):
            chunks.extend(entry_chunks(e, n))
        # everything else: fixed windows, page by page, as in chunks.py
        for s, (a, b) in enumerate(sorted(others)):
            first = page_at(t, a)
            for offset, piece in enumerate(t[a:b].split("\f")):
                if piece.strip():
                    for c in chunk_page(path.stem, first + offset, piece):
                        c.chunk_id = f"{path.stem}::p{first + offset:03d}::s{s:03d}::{c.chunk_id[-2:]}"
                        chunks.append(c)
    return chunks


if __name__ == "__main__":  # quick look: entries per manual, sizes, and the biggest ones
    import statistics
    for path in sorted(TEXT_DIR.glob("*.txt")):
        entries, _ = split_doc(path.stem, path.read_text())
        sizes = [len(tokenizer().encode(e.text, add_special_tokens=False).ids) for e in entries]
        big = sorted(zip(sizes, (e.code for e in entries)), reverse=True)[:4]
        print(f"{path.stem}: {len(entries)} entries, median {statistics.median(sizes):.0f} tokens, "
              f"max {max(sizes)}, over {MAX_ENTRY_TOKENS}: {sum(s > MAX_ENTRY_TOKENS for s in sizes)}  biggest {big}")
    chunks = load_struct_chunks()
    print(f"\n{len(chunks)} chunks ({sum('::e' in c.chunk_id for c in chunks)} from entries), "
          f"max {max(c.n_tokens for c in chunks)} tokens")
