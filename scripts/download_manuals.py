"""Download the public equipment manuals for the manufacturing corpus (Hybrid RAG #1).

The PDFs belong to their manufacturers, so they are not committed: this script fetches them
into data/manufacturing/raw/ (git-ignored). Run from hybrid_rag_in_depth/:

    uv run scripts/download_manuals.py
"""

from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "manufacturing" / "raw"

# file name -> public URL
MANUALS = {
    "franklin_subdrive.pdf": "https://siurbliai.lt/wp-content/uploads/2019/03/SUBDRIVE_Manual_-ENG.pdf",
    "goulds_aquavar_solo2.pdf": "https://www.mitchellewis.com/wp-content/uploads/2021/09/AquaVar-Solo2_Manual.pdf",
    "goulds_aquavar_im043.pdf": "https://prod.xylem.com/siteassets/brand/goulds-water-technology/resources/manual/im043.pdf",
    "goulds_im154.pdf": "https://prod.xylem.com/siteassets/brand/goulds-water-technology/resources/manual/im154.pdf",
    "danfoss_fc202.pdf": "https://assets.danfoss.com/documents/273713/AQ267037400115en-002501.pdf",
    "yaskawa_ga500_troubleshooting.pdf": "https://rspsupply.com/images/downloads/Yaskawa/G/Yaskawa%20GA50UB012ABA-134/GA500%20Maintenance%20&%20Troubleshooting%20Manual.pdf",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in MANUALS.items():
        path = RAW / name
        if path.exists():
            print(f"skip  {name} (already downloaded)")
            continue
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise SystemExit(f"{name}: {url} did not return a PDF")
        path.write_bytes(resp.content)
        print(f"saved {name} ({len(resp.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
