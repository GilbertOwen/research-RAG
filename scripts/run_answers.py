"""Run every test question through ask and save the answers for grading (Phase 5, step 5.1).

Run: uv run python scripts/run_answers.py [--label baseline]
"""

import argparse
import json
import sys
import time

from research_rag.ask import ask
from research_rag.config import ROOT, load_config, questions_path
from research_rag.evaluate import score_question
from research_rag.search import Index


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    config = load_config()
    index = Index(config)
    with open(questions_path(config), encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    out_dir = ROOT / "examples" / f"eval_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.perf_counter()
    print(f"{'id':4} {'search':6} {'answer':9} {'citations':9} {'sec':>5}")
    for q in questions:
        run = ask(index, config, q["question"])
        run["id"] = q["id"]
        run["reference_answer"] = q["answer"]
        run["expected_evidence"] = q["evidence"]
        with open(out_dir / f"{q['id']}.json", "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=1)

        check = run["citation_check"]
        sources = [{**s, "arxiv_id": s["id"].split(":")[0], "rank": i}
                   for i, s in enumerate(run["sources"], start=1)]
        search_hit = score_question(sources, q["evidence"])["hit"] if q["evidence"] else None
        answer_kind = "refused" if check["abstained"] else "answered"
        citations = "-" if check["abstained"] else ("ok" if check["ok"] else "problems")
        rows.append({"id": q["id"], "type": q["type"], "search_hit": search_hit, "answer": answer_kind,
                     "citations": citations, "prompt_tokens": run["prompt_tokens"], "seconds": run["seconds"]})
        hit_text = "-" if search_hit is None else ("hit" if search_hit else "miss")
        print(f"{q['id']:4} {hit_text:6} {answer_kind:9} {citations:9} {run['seconds']:5}")

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "questions": len(rows),
        "search_hits": sum(bool(r["search_hit"]) for r in answerable),
        "answered_when_answerable": sum(r["answer"] == "answered" for r in answerable),
        "refused_when_unanswerable": sum(r["answer"] == "refused" for r in unanswerable),
        "citation_check_ok": sum(r["citations"] == "ok" for r in rows),
        "total_seconds": round(time.perf_counter() - start),
    }
    print(f"\nsearch hits {summary['search_hits']}/{len(answerable)}, answered {summary['answered_when_answerable']}/"
          f"{len(answerable)} answerable, refused {summary['refused_when_unanswerable']}/{len(unanswerable)} "
          f"unanswerable, citation check ok {summary['citation_check_ok']}, {summary['total_seconds']} s")

    out = ROOT / "results" / f"answers_{args.label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "settings": run["settings"], "summary": summary, "questions": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"saved {out} and {out_dir}")


if __name__ == "__main__":
    main()
