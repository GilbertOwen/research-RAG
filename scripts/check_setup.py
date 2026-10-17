"""Check that Ollama and both models work. Run: uv run python scripts/check_setup.py"""

from ollama import Client

from research_rag.config import load_config

config = load_config()
client = Client(host=config["ollama_host"])

answer = client.chat(model=config["models"]["llm"], messages=[{"role": "user", "content": "Say OK."}],
                     think=config["models"]["think"])
print("LLM ok:", answer.message.content.strip())

vector = client.embed(model=config["models"]["embedding"], input="test").embeddings[0]
print("Embedding ok:", len(vector), "numbers")
