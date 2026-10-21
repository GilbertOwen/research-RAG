"""Score retrieval on the answerable test questions.

A hit means a retrieved chunk is from the expected paper and page and contains the evidence phrase.

Run: uv run python -m research_rag.evaluate [--no-instruction] [--top-k 5] [--label name]
"""

import argparse
import json
import sys
from datetime import datetime

from research_rag.config import ROOT, load_config, questions_path
from research_rag.search import Index


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def matches(chunk: dict, evidence: dict) -> bool:
    return (chunk["arxiv_id"] == evidence["paper"] and chunk["page"] == evidence["page"]
            and normalise(evidence["phrase"]) in normalise(chunk["text"]))


def score_question(results: list[dict], evidence: list[dict]) -> dict:
    found = [any(matches(r, ev) for r in results) for ev in evidence]
    first_rank = next((r["rank"] for r in results if any(matches(r, ev) for ev in evidence)), None)
    papers = {ev["paper"] for ev in evidence}
    return {
        "hit": any(found),
        "evidence_recall": sum(found) / len(found),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "paper_recall": len(papers & {r["arxiv_id"] for r in results}) / len(papers),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-instruction", action="store_true")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    config = load_config()
    top_k = args.top_k or config["retrieval"]["top_k"]
    instruction = None if args.no_instruction else config["retrieval"]["query_instruction"]

    with open(questions_path(config), encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["evidence"]]

    index = Index(config)
    rows = []
    print(f"{'id':4} {'hit':4} {'evid':>5} {'RR':>5} {'paper':>5}  top chunks")
    for q in questions:
        results = index.search(q["question"], top_k, instruction)
        s = score_question(results, q["evidence"])
        rows.append({"id": q["id"], "type": q["type"], **s, "retrieved": [r["id"] for r in results]})
        top = ", ".join(f"{r['arxiv_id'].split('v')[0]}:p{r['page']}" for r in results[:3])
        print(f"{q['id']:4} {'yes' if s['hit'] else 'no':4} {s['evidence_recall']:5.2f} "
              f"{s['reciprocal_rank']:5.2f} {s['paper_recall']:5.2f}  {top}")

    n = len(rows)
    summary = {
        "hit_at_k": sum(r["hit"] for r in rows) / n,
        "evidence_recall": sum(r["evidence_recall"] for r in rows) / n,
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
        "paper_recall": sum(r["paper_recall"] for r in rows) / n,
    }
    print(f"\n{n} questions, top-{top_k}, mode: {index.mode}, instruction: {'yes' if instruction else 'no'}, "
          f"chunks: {config['chunking']['size_words']}/{config['chunking']['overlap_words']} words")
    print(f"Hit@{top_k} {summary['hit_at_k']:.2f} ({sum(r['hit'] for r in rows)}/{n})   "
          f"evidence recall {summary['evidence_recall']:.2f}   MRR {summary['mrr']:.2f}   "
          f"paper recall {summary['paper_recall']:.2f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"retrieval_{args.label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "time": datetime.now().isoformat(timespec="seconds"),
                   "top_k": top_k, "mode": index.mode, "instruction": bool(instruction), "chunking": config["chunking"],
                   "add_title": config["embedding"]["add_title"], "summary": summary, "questions": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
