"""Embed every chunk; writes embeddings.npy (rows in chunks.jsonl order) and embeddings_meta.json.

Run: uv run python -m research_rag.embed
"""

import json
import sys
import time

import numpy as np
from ollama import Client

from research_rag.config import ROOT, load_config


def document_text(chunk: dict, add_title: bool) -> str:
    if add_title:
        return f"Title: {chunk['title']}\n\n{chunk['text']}"
    return chunk["text"]


def embed_texts(client: Client, model: str, texts: list[str], batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        rows.extend(client.embed(model=model, input=batch).embeddings)
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(rows, dtype=np.float32)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    index_dir = ROOT / config["paths"]["index_dir"]
    model = config["models"]["embedding"]
    add_title = config["embedding"]["add_title"]

    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    client = Client(host=config["ollama_host"])
    start = time.perf_counter()
    vectors = embed_texts(client, model, [document_text(c, add_title) for c in chunks],
                          config["embedding"]["batch_size"])
    seconds = time.perf_counter() - start

    np.save(index_dir / "embeddings.npy", vectors)
    meta = {
        "model": model,
        "add_title": add_title,
        "chunking": config["chunking"],
        "count": len(chunks),
        "dimensions": int(vectors.shape[1]),
        "chunk_ids": [c["id"] for c in chunks],
    }
    with open(index_dir / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"{len(chunks)} chunks embedded with {model} in {seconds:.0f} s, "
          f"{vectors.shape[1]} numbers each, vector lengths {norms.min():.4f} to {norms.max():.4f}")
    print(f"saved {index_dir / 'embeddings.npy'} and embeddings_meta.json")


if __name__ == "__main__":
    main()
