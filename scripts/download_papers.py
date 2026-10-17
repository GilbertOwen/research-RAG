"""Download the manifest's papers from arXiv, only if the version's licence matches. Safe to rerun.

Run: uv run python scripts/download_papers.py
"""

import csv
import re
import time
import urllib.request

import pymupdf

from research_rag.config import ROOT, load_config

LICENCE_URLS = {
    "CC BY 4.0": "creativecommons.org/licenses/by/4.0",
    "CC0 1.0": "creativecommons.org/publicdomain/zero/1.0",
}
HEADERS = {"User-Agent": "research-rag (student assessment project; polite single downloads)"}
DELAY_SECONDS = 3  


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def licence_on_arxiv(arxiv_id: str) -> str:
    page = fetch(f"https://arxiv.org/abs/{arxiv_id}").decode("utf-8", "replace")
    match = re.search(r'href="https?://((?:creativecommons\.org|arxiv\.org/licenses)/[^"]*?)/?"', page)
    return match.group(1) if match else "not found"


config = load_config()
papers_dir = ROOT / config["paths"]["papers_dir"]
papers_dir.mkdir(parents=True, exist_ok=True)

with open(ROOT / config["paths"]["manifest"], encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

for paper in papers:
    arxiv_id = paper["arxiv_id"] + paper["version"]
    pdf_path = papers_dir / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        print(f"skip {arxiv_id}: already downloaded")
        continue

    licence = licence_on_arxiv(arxiv_id)
    time.sleep(DELAY_SECONDS)
    if licence != LICENCE_URLS[paper["license"]]:
        print(f"SKIP {arxiv_id}: manifest says {paper['license']}, arXiv shows {licence}")
        continue

    pdf_bytes = fetch(f"https://arxiv.org/pdf/{arxiv_id}")
    time.sleep(DELAY_SECONDS)
    if not pdf_bytes.startswith(b"%PDF"):
        print(f"SKIP {arxiv_id}: response is not a PDF")
        continue

    pdf_path.write_bytes(pdf_bytes)
    with pymupdf.open(pdf_path) as doc:
        pages = doc.page_count
    print(f"ok   {arxiv_id}: {pages} pages, {len(pdf_bytes) / 1e6:.1f} MB, {paper['license']}")
