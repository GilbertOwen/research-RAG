"""Ask every test question with no sources, to see what the LLM knows on its own (Phase 5, step 5.3).

Writes results/closed_book.json, a reading sheet results/closed_book.md and an empty marks file
results/grades_closed_book.csv (never overwritten).
Run: uv run python scripts/run_closed_book.py
"""

import csv
import json
import sys
import time

from ollama import Client

from research_rag.config import ROOT, load_config

DONT_KNOW = "I DON'T KNOW"
SYSTEM_PROMPT = f"""You answer questions about research papers on image captioning from your own knowledge.
Answer in 1 to 4 short sentences.
If you do not know the answer, reply exactly: {DONT_KNOW}"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    client = Client(host=config["ollama_host"])
    with open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    rows = []
    for q in questions:
        start = time.perf_counter()
        reply = client.chat(
            model=config["models"]["llm"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q["question"]}],
            think=config["models"]["think"],
            options={"temperature": config["generation"]["temperature"], "num_ctx": config["generation"]["num_ctx"]},
        )
        answer = reply.message.content.strip()
        said_dont_know = answer.upper().startswith(DONT_KNOW)
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "answer": answer,
                     "dont_know": said_dont_know, "seconds": round(time.perf_counter() - start, 1)})
        print(f"{q['id']:4} {'dont know' if said_dont_know else 'answered':9} {answer[:90]}")

    out = ROOT / "results"
    with open(out / "closed_book.json", "w", encoding="utf-8") as f:
        json.dump({"settings": {"llm": config["models"]["llm"], "temperature": config["generation"]["temperature"]},
                   "questions": rows}, f, ensure_ascii=False, indent=1)

    by_id = {q["id"]: q for q in questions}
    lines = ["# Closed-book answers (no sources)", "", "Graded with `eval/rubric.md` (correctness only).", ""]
    for r in rows:
        q = by_id[r["id"]]
        lines += [f"## {r['id']} ({r['type']})", "", f"**Question:** {r['question']}", ""]
        lines += ["**Key facts:**"] + [f"- {k}" for k in q.get("key_facts", [])] if q.get("key_facts") \
            else ["**Expected:** say it does not know (unanswerable)"]
        lines += ["", f"**Closed-book answer:** {r['answer']}", "", "---", ""]
    (out / "closed_book.md").write_text("\n".join(lines), encoding="utf-8")

    sheet = out / "grades_closed_book.csv"
    if not sheet.exists():
        with open(sheet, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "type", "dont_know", "correctness", "notes"])
            writer.writeheader()
            writer.writerows({"id": r["id"], "type": r["type"], "dont_know": r["dont_know"],
                              "correctness": "", "notes": ""} for r in rows)

    print(f"\n{sum(r['dont_know'] for r in rows)} of {len(rows)} answered '{DONT_KNOW}'")
    print(f"saved {out / 'closed_book.json'}, closed_book.md and grades_closed_book.csv")


if __name__ == "__main__":
    main()
