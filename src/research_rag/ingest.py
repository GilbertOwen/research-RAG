"""PDFs to clean page text (without reference lists) to overlapping chunks.

Run: uv run python -m research_rag.ingest
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

from research_rag.config import ROOT, load_config

MIN_NEW_WORDS = 50
ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]")
LINE_HYPHEN = re.compile(r"(\w+)-\n(\w+)")
INLINE_HYPHEN_WORD = re.compile(r"\b\w+-\w+\b")
REFERENCES_HEADING = re.compile(r"^(\d+\.?\s*)?(references|bibliography)$", re.IGNORECASE)
APPENDIX_WORD = re.compile(r"^(appendix|appendices|supplementary materials?)$", re.IGNORECASE)
# e.g. "A. Human Evaluation Dataset Construction"; not "H. A comprehensive survey" (a reference)
LETTERED_HEADING = re.compile(r"^[A-H](\.\d+)*\.?\s+[A-Z][a-z][A-Za-z\-]*(\s+[A-Za-z\-]+){0,8}$")
REFERENCE_MARKERS = re.compile(
    r"\b(In Proceedings|Proceedings of|Conference|arXiv|Advances in|Journal|Transactions|pp\.|doi)\b|https?://",
    re.IGNORECASE,
)


def page_blocks(page: pymupdf.Page) -> list[str]:
    """Text blocks of a page in reading order, without images, page numbers and the arXiv stamp."""
    blocks = []
    for x0, y0, x1, y1, text, block_no, block_type in page.get_text("blocks"):
        text = unicodedata.normalize("NFKC", text).strip()  # "ﬁ" -> "fi"
        if block_type != 0 or not text or text.isdigit() or ARXIV_STAMP.match(text):
            continue
        blocks.append(text)
    return blocks


def join_hyphenated(text: str, hyphen_words: set[str]) -> str:
    """Join words split at a line end, keeping hyphens the paper uses elsewhere (vision-language)."""

    def repair(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if f"{left}-{right}".lower() in hyphen_words:
            return f"{left}-{right}"
        return left + right

    return LINE_HYPHEN.sub(repair, text)


def is_appendix_heading(line: str, title: str) -> bool:
    if APPENDIX_WORD.match(line):
        return True
    return bool(LETTERED_HEADING.match(line)) and line.lower() != title.lower()  # running header is not a heading


def capitalised_share(block: str) -> float:
    words = block.split()
    return sum(1 for w in words if w[:1].isupper()) / len(words) if words else 0.0


def is_prose(block: str) -> bool:
    """A normal paragraph: 40+ words, mostly lowercase, no reference markers."""
    if len(block.split()) < 40 or REFERENCE_MARKERS.search(block):
        return False
    return capitalised_share(block) < 0.25


def extract_pages(pdf_path: Path, title: str) -> tuple[list[dict], dict]:
    """Pages as [{"page": 1, "text": "..."}], skipping from "References" to the appendix or first prose block."""
    with pymupdf.open(pdf_path) as doc:
        raw_pages = [page_blocks(page) for page in doc]

    all_text = "\n".join(block for blocks in raw_pages for block in blocks)
    hyphen_words = {w.lower() for w in INLINE_HYPHEN_WORD.findall(all_text)}

    report = {"references_from": None, "appendix_from": None, "appendix_heading": None, "skipped_chars": 0}
    in_references = False
    pages = []
    for number, blocks in enumerate(raw_pages, start=1):
        paragraphs = []
        for block in blocks:
            # stray reference entries after the list ends (ReFormer)
            if (report["references_from"] and len(REFERENCE_MARKERS.findall(block)) >= 2
                    and capitalised_share(block) >= 0.25):
                report["skipped_chars"] += len(block)
                continue
            if in_references and is_prose(block):
                in_references = False
                report["appendix_from"], report["appendix_heading"] = number, " ".join(block.split()[:8]) + " ..."
            kept_lines = []
            for line in block.split("\n"):
                stripped = line.strip()
                if not in_references and REFERENCES_HEADING.match(stripped) and report["references_from"] is None:
                    in_references = True
                    report["references_from"] = number
                elif in_references and is_appendix_heading(stripped, title):
                    in_references = False
                    report["appendix_from"], report["appendix_heading"] = number, stripped
                if in_references:
                    report["skipped_chars"] += len(stripped)
                else:
                    kept_lines.append(line)
            if kept_lines:
                text = join_hyphenated("\n".join(kept_lines), hyphen_words)
                paragraphs.append(" ".join(text.split()))
        pages.append({"page": number, "text": "\n\n".join(paragraphs)})
    return pages, report


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping word windows over one page; a small last piece is glued onto the one before."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    ranges = []
    start = 0
    while True:
        end = min(start + size, len(words))
        ranges.append([start, end])
        if end == len(words):
            break
        start += step
    if len(ranges) > 1 and ranges[-1][1] - ranges[-2][1] < MIN_NEW_WORDS:
        ranges[-2][1] = ranges[-1][1]
        ranges.pop()
    return [" ".join(words[a:b]) for a, b in ranges]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    papers_dir = ROOT / config["paths"]["papers_dir"]
    with open(ROOT / config["paths"]["manifest"], encoding="utf-8") as f:
        papers = list(csv.DictReader(f))

    size = config["chunking"]["size_words"]
    overlap = config["chunking"]["overlap_words"]
    index_dir = ROOT / config["paths"]["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"] + paper["version"]
        pages, report = extract_pages(papers_dir / f"{arxiv_id}.pdf", paper["title"])
        chunks = [
            {
                "id": f"{arxiv_id}:p{page['page']}:c{i}",
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "page": page["page"],
                "text": text,
                "words": len(text.split()),
            }
            for page in pages
            for i, text in enumerate(chunk_words(page["text"], size, overlap))
        ]
        all_chunks.extend(chunks)
        appendix = (f"appendix from p.{report['appendix_from']} ({report['appendix_heading']!r})"
                    if report["appendix_from"] else "no appendix found")
        print(f"{arxiv_id:14} {len(chunks):3} chunks  references from p.{report['references_from']}, "
              f"{report['skipped_chars']:,} chars skipped, {appendix}")

    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    words = [c["words"] for c in all_chunks]
    print(f"total: {len(all_chunks)} chunks, {min(words)} to {max(words)} words "
          f"(mean {sum(words) / len(words):.0f}), saved to {index_dir / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
