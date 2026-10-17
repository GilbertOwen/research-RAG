"""Find the chunks closest to a question.

Run: uv run python -m research_rag.search "your question" --debug
"""

import argparse
import json
import sys

import numpy as np
from ollama import Client

from research_rag.config import ROOT, load_config


def format_query(question: str, instruction: str | None) -> str:
    if not instruction:
        return question
    return f"Instruct: {instruction}\nQuery: {question}"


class Index:
    """The chunks and their vectors, loaded once."""

    def __init__(self, config: dict):
        index_dir = ROOT / config["paths"]["index_dir"]
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.vectors = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "embeddings_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        if meta["chunk_ids"] != [c["id"] for c in self.chunks]:
            sys.exit("embeddings do not match chunks.jsonl: run research_rag.embed again")
        if meta["model"] != config["models"]["embedding"]:
            sys.exit(f"index was built with {meta['model']}, config says {config['models']['embedding']}")

        self.model = meta["model"]
        self.client = Client(host=config["ollama_host"])

    def search(self, question: str, top_k: int, instruction: str | None = None) -> list[dict]:
        query = format_query(question, instruction)
        q = np.array(self.client.embed(model=self.model, input=query).embeddings[0], dtype=np.float32)
        scores = self.vectors @ q  # vectors have length 1, so this is cosine similarity
        best = np.argsort(-scores)[:top_k]
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(best)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search the paper collection.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-instruction", action="store_true", help="embed the plain question")
    parser.add_argument("--debug", action="store_true", help="print the text of each chunk")
    args = parser.parse_args()

    config = load_config()
    top_k = args.top_k or config["retrieval"]["top_k"]
    instruction = None if args.no_instruction else config["retrieval"]["query_instruction"]

    results = Index(config).search(args.question, top_k, instruction)
    print(f"Question: {args.question}\n")
    for r in results:
        print(f"[{r['rank']}] {r['score']:.3f}  {r['id']}  {r['title'][:60]}, p.{r['page']}")
        if args.debug:
            print(f"    {r['text'][:400]}...\n")


if __name__ == "__main__":
    main()
