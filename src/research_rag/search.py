"""Find the chunks closest to a question.

Run: uv run python -m research_rag.search "your question" --debug
"""

import argparse
import json
import sys

import numpy as np
from ollama import Client

from research_rag.config import ROOT, load_config
from research_rag.keyword import BM25


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
        self.mode = config["retrieval"].get("mode", "dense")
        self.rrf_k = config["retrieval"].get("rrf_k", 60)
        self.mmr_lambda = config["retrieval"].get("mmr_lambda", 0.7)
        self.pool = config["retrieval"].get("pool", 20)
        if self.mode == "hybrid":
            self.bm25 = BM25([c["text"] for c in self.chunks])

    def search(self, question: str, top_k: int, instruction: str | None = None) -> list[dict]:
        query = format_query(question, instruction)
        q = np.array(self.client.embed(model=self.model, input=query).embeddings[0], dtype=np.float32)
        dense = self.vectors @ q  # vectors have length 1, so this is cosine similarity
        if self.mode == "mmr":
            return self.mmr(dense, top_k)
        if self.mode != "hybrid":
            best = np.argsort(-dense)[:top_k]
            return [{**self.chunks[i], "score": float(dense[i]), "rank": r + 1} for r, i in enumerate(best)]

        # Reciprocal rank fusion: 1 / (k + rank) from each list, added up
        keyword = self.bm25.scores(question)  # the plain question, without the instruction
        fused = np.zeros(len(self.chunks), dtype=np.float32)
        for scores in (dense, keyword):
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            fused += 1 / (self.rrf_k + ranks)
        best = np.argsort(-fused, kind="stable")[:top_k]
        return [{**self.chunks[i], "score": float(fused[i]), "dense_score": float(dense[i]),
                 "keyword_score": float(keyword[i]), "rank": r + 1} for r, i in enumerate(best)]

    def mmr(self, dense: np.ndarray, top_k: int) -> list[dict]:
        """Reorder the top `pool` chunks: relevant to the question, but not too similar to chunks already picked."""
        order = np.argsort(-dense)
        pool = list(order[:self.pool])
        picked = []
        while pool and len(picked) < top_k:
            if picked:
                redundancy = (self.vectors[pool] @ self.vectors[picked].T).max(axis=1)
            else:
                redundancy = np.zeros(len(pool))
            scores = self.mmr_lambda * dense[pool] - (1 - self.mmr_lambda) * redundancy
            picked.append(pool.pop(int(np.argmax(scores))))
        best = picked + list(order[self.pool:])[:max(0, top_k - len(picked))]  # beyond the pool: dense order
        return [{**self.chunks[i], "score": float(dense[i]), "rank": r + 1} for r, i in enumerate(best)]


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
