"""Find where each expected evidence chunk ranks when the whole index is searched.

Uses the index in data/index and the settings in config.yaml; nothing is changed.
Run: uv run python scripts/diagnose_ranks.py [--out ranks_400_0_emb4b]
"""

import argparse
import json
import sys
from collections import Counter

from research_rag.config import ROOT, load_config
from research_rag.evaluate import matches
from research_rag.search import Index

POOL = 20  # candidate pool a diversity step (option C) would rerank


def short(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def bucket(rank: int | None, top_k: int) -> str:
    if rank is None:
        return "not found"
    if rank <= top_k:
        return "top-k"
    return "near" if rank <= POOL else "deep"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="output name in results/, instead of one built from the config")
    args = parser.parse_args()
    config = load_config()
    top_k = config["retrieval"]["top_k"]
    instruction = config["retrieval"]["query_instruction"]

    with open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["evidence"]]

    index = Index(config)
    items, per_question = [], []
    for q in questions:
        results = index.search(q["question"], len(index.chunks), instruction)
        cutoff_score = results[top_k - 1]["score"]
        for ev in q["evidence"]:
            hit = next((r for r in results if matches(r, ev)), None)
            paper_first = next(r["rank"] for r in results if r["arxiv_id"] == ev["paper"])
            rank = hit["rank"] if hit else None
            items.append({
                "id": q["id"], "type": q["type"], "paper": ev["paper"], "page": ev["page"],
                "rank": rank, "paper_rank": paper_first,
                "gap": round(cutoff_score - hit["score"], 3) if hit else None,
                "bucket": bucket(rank, top_k),
            })
        expected = {ev["paper"] for ev in q["evidence"]}
        pool_papers = {r["arxiv_id"] for r in results[:POOL]}
        per_question.append({
            "id": q["id"], "type": q["type"],
            "top_k_papers": [short(r["arxiv_id"]) for r in results[:top_k]],
            "expected_papers_in_pool": len(expected & pool_papers), "expected_papers": len(expected),
        })

    print(f"{'id':4} {'paper':10} {'page':>4} {'rank':>5} {'paper#':>6} {'gap':>6}  bucket")
    for it in items:
        rank = it["rank"] if it["rank"] is not None else "-"
        gap = f"{it['gap']:.3f}" if it["gap"] is not None else "-"
        print(f"{it['id']:4} {short(it['paper']):10} {it['page']:4} {rank:>5} {it['paper_rank']:6} {gap:>6}  {it['bucket']}")

    print(f"\nTop-{top_k} papers per question, and expected papers anywhere in the top {POOL}")
    for pq in per_question:
        print(f"{pq['id']:4} {pq['expected_papers_in_pool']}/{pq['expected_papers']} in pool   "
              f"top-{top_k}: {', '.join(pq['top_k_papers'])}")

    counts = Counter(it["bucket"] for it in items)
    near, deep = counts["near"], counts["deep"]
    pick = "C (diversity)" if near >= deep else "B (hybrid)"
    print(f"\n{len(items)} evidence items: top-{top_k} {counts['top-k']}, near (rank {top_k + 1} to {POOL}) {near}, "
          f"deep (beyond {POOL}) {deep}, not found {counts['not found']}")
    print(f"Reading rule (near >= deep picks C, otherwise B): {pick}")

    label = f"{config['chunking']['size_words']}_{config['chunking']['overlap_words']}"
    if index.mode != "dense":
        label += f"_{index.mode}"
    out = ROOT / "results" / f"{args.out or 'ranks_' + label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "pool": POOL, "chunks": len(index.chunks), "counts": dict(counts),
                   "rule_pick": pick, "items": items, "questions": per_question}, f, ensure_ascii=False, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
