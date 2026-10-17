"""Ask a question: search, answer with citations, check them, save the run to examples/.

Run: uv run python -m research_rag.ask "your question" [--debug] [--no-save]
"""

import argparse
import json
import re
import sys
from datetime import datetime

from research_rag.citations import check_citations
from research_rag.config import ROOT, load_config
from research_rag.generate import generate
from research_rag.search import Index


def ask(index: Index, config: dict, question: str) -> dict:
    chunks = index.search(question, config["retrieval"]["top_k"], config["retrieval"]["query_instruction"])
    result = generate(index.client, config, question, chunks)
    check = check_citations(result["answer"], len(chunks), config["generation"]["no_evidence_reply"])
    return {
        "question": question,
        "time": datetime.now().isoformat(timespec="seconds"),
        "settings": {"llm": config["models"]["llm"], "embedding": config["models"]["embedding"],
                     "chunking": config["chunking"], "add_title": config["embedding"]["add_title"],
                     "top_k": config["retrieval"]["top_k"], "temperature": config["generation"]["temperature"]},
        "answer": result["answer"],
        "citation_check": check,
        "sources": [{"label": f"S{i}", "id": c["id"], "title": c["title"], "page": c["page"],
                     "score": round(c["score"], 3), "text": c["text"]} for i, c in enumerate(chunks, start=1)],
        "prompt_tokens": result["prompt_tokens"],
        "answer_tokens": result["answer_tokens"],
        "seconds": result["seconds"],
    }


def print_run(run: dict, debug: bool) -> None:
    check = run["citation_check"]
    print(f"Question: {run['question']}\n\n{run['answer']}\n\nSources (* = cited):")
    for s in run["sources"]:
        mark = "*" if int(s["label"][1:]) in check["cited"] else " "
        print(f"{mark}[{s['label']}] {s['title'][:70]}, page {s['page']}  (score {s['score']})")
        if debug:
            print(f"      {s['text'][:500]}...\n")

    if check["abstained"]:
        print("\nCitation check: model said there is not enough evidence")
    elif check["ok"]:
        print("\nCitation check: ok, every sentence cites a retrieved source")
    else:
        print("\nCitation check: problems")
        if check["invalid"]:
            print(f"  cites sources that were not given: {', '.join(f'S{n}' for n in check['invalid'])}")
        for s in check["uncited_sentences"]:
            print(f"  sentence without citation: {s}")
        if not check["cited"] and not check["invalid"]:
            print("  no citations at all")
    print(f"\n{run['prompt_tokens']} prompt tokens, {run['answer_tokens']} answer tokens, {run['seconds']} s")


def save_run(run: dict) -> str:
    out_dir = ROOT / "examples"
    out_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", run["question"].lower()).strip("-")[:50]
    path = out_dir / f"{run['time'].replace(':', '')}_{slug}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=1)
    return str(path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the paper collection a question.")
    parser.add_argument("question")
    parser.add_argument("--debug", action="store_true", help="show the text of each source")
    parser.add_argument("--no-save", action="store_true", help="do not save the run to examples/")
    args = parser.parse_args()

    config = load_config()
    run = ask(Index(config), config, args.question)
    print_run(run, args.debug)
    if not args.no_save:
        print(f"saved {save_run(run)}")


if __name__ == "__main__":
    main()
