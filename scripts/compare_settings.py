"""Answer all test questions with other chunk settings and compare (Phase 5, step 5.5).

For each setting it builds a separate index in data/index_<label>/ (the main index is not touched),
answers every question into examples/eval_<label>/ and writes results/answers_<label>.json.
The 400 / 0 baseline is read from examples/eval_baseline/.
Run: uv run python scripts/compare_settings.py
"""

import copy
import csv
import json
import statistics
import sys

import numpy as np
from ollama import Client

from research_rag.ask import ask
from research_rag.config import ROOT, load_config
from research_rag.embed import document_text, embed_texts
from research_rag.evaluate import score_question
from research_rag.ingest import chunk_words, extract_pages
from research_rag.search import Index

SETTINGS = [("400_80", 400, 80), ("250_50", 250, 50)]


def build_index(config: dict) -> None:
    index_dir = ROOT / config["paths"]["index_dir"]
    if (index_dir / "embeddings.npy").exists():
        print(f"  index exists: {index_dir}")
        return
    index_dir.mkdir(parents=True, exist_ok=True)
    size, overlap = config["chunking"]["size_words"], config["chunking"]["overlap_words"]
    with open(ROOT / config["paths"]["manifest"], encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    chunks = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"] + paper["version"]
        pages, _ = extract_pages(ROOT / config["paths"]["papers_dir"] / f"{arxiv_id}.pdf", paper["title"])
        chunks += [{"id": f"{arxiv_id}:p{p['page']}:c{i}", "arxiv_id": arxiv_id, "title": paper["title"],
                    "page": p["page"], "text": text, "words": len(text.split())}
                   for p in pages for i, text in enumerate(chunk_words(p["text"], size, overlap))]
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    client = Client(host=config["ollama_host"])
    vectors = embed_texts(client, config["models"]["embedding"],
                          [document_text(c, config["embedding"]["add_title"]) for c in chunks],
                          config["embedding"]["batch_size"])
    np.save(index_dir / "embeddings.npy", vectors)
    with open(index_dir / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": config["models"]["embedding"], "add_title": config["embedding"]["add_title"],
                   "chunking": config["chunking"], "count": len(chunks), "dimensions": int(vectors.shape[1]),
                   "chunk_ids": [c["id"] for c in chunks]}, f, ensure_ascii=False, indent=1)
    print(f"  built {len(chunks)} chunks in {index_dir}")


def answer_all(config: dict, label: str, questions: list[dict]) -> None:
    out_dir = ROOT / "examples" / f"eval_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = Index(config)
    for q in questions:
        run = ask(index, config, q["question"])
        run.update({"id": q["id"], "reference_answer": q["answer"], "expected_evidence": q["evidence"]})
        with open(out_dir / f"{q['id']}.json", "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=1)
        print(f"  {q['id']:4} {'refused' if run['citation_check']['abstained'] else 'answered':9} {run['seconds']:5} s")


def summarise(label: str, questions: list[dict]) -> dict:
    runs = {q["id"]: json.load(open(ROOT / "examples" / f"eval_{label}" / f"{q['id']}.json", encoding="utf-8"))
            for q in questions}
    answerable = [q for q in questions if q["evidence"]]
    hits = 0
    for q in answerable:
        sources = [{**s, "arxiv_id": s["id"].split(":")[0], "rank": i}
                   for i, s in enumerate(runs[q["id"]]["sources"], start=1)]
        hits += score_question(sources, q["evidence"])["hit"]
    checks = [r["citation_check"] for r in runs.values()]
    return {
        "label": label,
        "search_hits": hits,
        "answered_answerable": sum(not runs[q["id"]]["citation_check"]["abstained"] for q in answerable),
        "refused_unanswerable": sum(runs[q["id"]]["citation_check"]["abstained"]
                                    for q in questions if not q["evidence"]),
        "citation_check_ok": sum(c["ok"] and not c["abstained"] for c in checks),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in runs.values())),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in runs.values()), 1),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    base = load_config()
    with open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    for label, size, overlap in SETTINGS:
        config = copy.deepcopy(base)
        config["chunking"] = {"size_words": size, "overlap_words": overlap}
        config["paths"]["index_dir"] = f"data/index_{label}"
        print(f"setting {label}: {size} words, overlap {overlap}")
        build_index(config)
        answer_all(config, label, questions)

    rows = [summarise("baseline", questions)] + [summarise(label, questions) for label, _, _ in SETTINGS]
    rows[0]["label"] = "baseline (400_0)"
    print(f"\n{'setting':17} {'search hits':>11} {'answered':>8} {'refused U':>9} {'cites ok':>8} "
          f"{'prompt tok':>10} {'sec':>5}")
    for r in rows:
        print(f"{r['label']:17} {r['search_hits']:>8}/22 {r['answered_answerable']:>5}/22 {r['refused_unanswerable']:>7}/3 "
              f"{r['citation_check_ok']:>8} {r['mean_prompt_tokens']:>10} {r['mean_seconds']:>5}")
    with open(ROOT / "results" / "compare_settings.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\nsaved results/compare_settings.json; grade with scripts/make_grading_sheet.py --label 400_80 / 250_50")


if __name__ == "__main__":
    main()
