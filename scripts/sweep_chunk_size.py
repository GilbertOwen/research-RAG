"""Compare chunk sizes on the test questions; everything else stays as in config.yaml.

Builds each index in memory, so data/index is not touched.
Run: uv run python scripts/sweep_chunk_size.py [--sizes 100 175 250 400 600] [--overlap-share 0.2]
"""

import argparse
import csv
import json
import sys
import time

import numpy as np
from ollama import Client

from research_rag.config import ROOT, load_config
from research_rag.embed import document_text, embed_texts
from research_rag.evaluate import score_question
from research_rag.ingest import chunk_words, extract_pages
from research_rag.search import format_query


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 175, 250, 400, 600])
    parser.add_argument("--overlap-share", type=float, default=0.2)
    parser.add_argument("--overlaps", type=int, nargs="+", help="fixed overlaps in words, instead of a share")
    parser.add_argument("--out", default="sweep_chunk_size")
    args = parser.parse_args()
    settings = [(size, overlap) for size in args.sizes
                for overlap in (args.overlaps or [round(size * args.overlap_share)])]

    config = load_config()
    client = Client(host=config["ollama_host"])
    model = config["models"]["embedding"]
    top_k = config["retrieval"]["top_k"]
    add_title = config["embedding"]["add_title"]

    with open(ROOT / config["paths"]["manifest"], encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    pages = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"] + paper["version"]
        paper_pages, _ = extract_pages(ROOT / config["paths"]["papers_dir"] / f"{arxiv_id}.pdf", paper["title"])
        pages += [{**p, "arxiv_id": arxiv_id, "title": paper["title"]} for p in paper_pages]

    with open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["evidence"]]
    queries = [format_query(q["question"], config["retrieval"]["query_instruction"]) for q in questions]
    query_vectors = np.array(client.embed(model=model, input=queries).embeddings, dtype=np.float32)

    runs = []
    for size, overlap in settings:
        chunks = [{"id": f"{p['arxiv_id']}:p{p['page']}:c{i}", "arxiv_id": p["arxiv_id"], "title": p["title"],
                   "page": p["page"], "text": text}
                  for p in pages for i, text in enumerate(chunk_words(p["text"], size, overlap))]
        start = time.perf_counter()
        print(f"size {size}, overlap {overlap}: embedding {len(chunks)} chunks")
        vectors = embed_texts(client, model, [document_text(c, add_title) for c in chunks],
                              config["embedding"]["batch_size"])
        seconds = time.perf_counter() - start

        scores = []
        for q, qv in zip(questions, query_vectors):
            sims = vectors @ qv
            best = np.argsort(-sims)[:top_k]
            results = [{**chunks[i], "rank": r + 1} for r, i in enumerate(best)]
            scores.append({"id": q["id"], **score_question(results, q["evidence"])})

        n = len(scores)
        run = {
            "size": size, "overlap": overlap, "chunks": len(chunks), "embed_seconds": round(seconds),
            "hits": sum(s["hit"] for s in scores),
            "hit_at_k": sum(s["hit"] for s in scores) / n,
            "evidence_recall": sum(s["evidence_recall"] for s in scores) / n,
            "mrr": sum(s["reciprocal_rank"] for s in scores) / n,
            "paper_recall": sum(s["paper_recall"] for s in scores) / n,
            "context_words": size * top_k,
            "questions": scores,
        }
        runs.append(run)

    print(f"\n{len(questions)} questions, top-{top_k}, instruction on, add_title {add_title}")
    print(f"{'size':>5} {'overlap':>7} {'chunks':>6} {'hits':>5} {'Hit@k':>6} {'evid':>5} {'MRR':>5} "
          f"{'paper':>5} {'context':>8} {'embed s':>7}")
    for r in runs:
        print(f"{r['size']:5} {r['overlap']:7} {r['chunks']:6} {r['hits']:5} {r['hit_at_k']:6.2f} "
              f"{r['evidence_recall']:5.2f} {r['mrr']:5.2f} {r['paper_recall']:5.2f} {r['context_words']:8} "
              f"{r['embed_seconds']:7}")

    best_hits = max(r["hits"] for r in runs)
    pick = min((r for r in runs if r["hits"] >= best_hits - 1), key=lambda r: (r["size"], r["overlap"]))
    print(f"\nDecision rule (smallest setting within one question of the best): "
          f"{pick['size']} words, overlap {pick['overlap']}")

    out = ROOT / "results" / f"{args.out}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "add_title": add_title, "settings": settings,
                   "rule_pick": [pick["size"], pick["overlap"]], "runs": runs}, f, ensure_ascii=False, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
