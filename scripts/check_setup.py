"""Check that both Ollama models answer before building anything.

Run: uv run python scripts/check_setup.py
"""

import time

from ollama import Client

from miracl_rag.config import load_config

config = load_config()
client = Client(host=config["ollama_host"])
llm = config["models"]["llm"]
embedder = config["models"]["embedding"]

start = time.perf_counter()
reply = client.chat(
    model=llm,
    messages=[{"role": "user", "content": "Jawab dengan satu kata: ibu kota Indonesia?"}],
    think=config["models"]["think"],
)
print(f"LLM {llm}: {reply.message.content.strip()!r} ({time.perf_counter() - start:.1f} s)")

start = time.perf_counter()
vectors = client.embed(model=embedder, input=["Candi Borobudur terletak di Magelang."])
print(f"Embedding {embedder}: {len(vectors.embeddings[0])} numbers per text ({time.perf_counter() - start:.1f} s)")
