"""Build the grading sheet for a saved batch run (Phase 5, step 5.2).

Writes results/grading_<label>.md (what to read) and results/grades_<label>.csv (where the marks go).
Run: uv run python scripts/make_grading_sheet.py [--label baseline]
"""

import argparse
import csv
import json
import sys

from research_rag.config import ROOT, load_config, questions_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    with open(questions_path(load_config()), encoding="utf-8") as f:
        questions = {q["id"]: q for q in map(json.loads, f)}
    run_dir = ROOT / "examples" / f"eval_{args.label}"

    lines = [f"# Grading sheet: {args.label}", "",
             "Rubric: `eval/rubric.md`. Put the marks in the CSV next to this file.", ""]
    rows = []
    for qid, q in questions.items():
        with open(run_dir / f"{qid}.json", encoding="utf-8") as f:
            run = json.load(f)
        check = run["citation_check"]
        status = "refused" if check["abstained"] else "answered"
        lines += [f"## {qid} ({q['type']})", "", f"**Question:** {q['question']}", "",
                  "**Key facts:**" if q.get("key_facts") else "**Expected:** refuse (unanswerable)"]
        lines += [f"- {fact}" for fact in q.get("key_facts", [])]
        lines += ["", f"**Reference answer:** {q['answer']}", "",
                  f"**System answer ({status}):** {run['answer']}", ""]
        if not check["abstained"]:
            problems = check["invalid"] or check["uncited_sentences"]
            lines += [f"Citation check: {'problems' if problems else 'ok'}"
                      + (f" (uncited: {len(check['uncited_sentences'])}, invalid: {check['invalid']})" if problems else ""),
                      ""]
            for s in run["sources"]:
                if int(s["label"][1:]) in check["cited"]:
                    lines += [f"**[{s['label']}] {s['title']}, page {s['page']}**", "", f"> {s['text']}", ""]
        lines += ["---", ""]
        rows.append({"id": qid, "type": q["type"], "status": status,
                     "correctness": "", "citation_support": "n/a" if status == "refused" else "", "notes": ""})

    md = ROOT / "results" / f"grading_{args.label}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    sheet = ROOT / "results" / f"grades_{args.label}.csv"
    if sheet.exists():
        print(f"{sheet} already exists; not overwritten (your marks are safe)")
    else:
        with open(sheet, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {sheet}")
    print(f"saved {md}")


if __name__ == "__main__":
    main()
