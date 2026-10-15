# miracl-rag

Retrieval-augmented question answering over a subset of Indonesian Wikipedia, evaluated with the MIRACL Indonesian relevance labels.

Work in progress. Sections on the documents, architecture, model choices, results and limitations are added as each part is built.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) running locally, and the two models:

```
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
uv sync
```

Check that both models answer:

```
uv run python scripts/check_setup.py
```

Settings are in `config.yaml`. The Ollama address can be changed through `OLLAMA_HOST` (see `.env.example`). No API keys are needed.
