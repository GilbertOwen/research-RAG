# research-rag

A local retrieval-augmented assistant for research papers on image captioning and vision-language model evaluation. Answers come only from a curated collection of openly licensed arXiv papers and cite the paper and page they come from.

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

Download the papers (20 PDFs, about 87 MB, into `data/papers/`):

```
uv run python scripts/download_papers.py
```

Settings are in `config.yaml`. The Ollama address can be changed through `OLLAMA_HOST` (see `.env.example`). No API keys are needed.

## Document collection

20 arXiv papers on image captioning and its evaluation, listed in [`papers/manifest.csv`](papers/manifest.csv) with the exact version used: background surveys, captioning methods, news captioning with context, caption evaluation and bias in metrics, and object hallucination. 351 pages in total. Each paper is used under its own licence (19 under CC BY 4.0, one under CC0 1.0). The download script reads the licence on the arXiv page of each version and skips any paper whose licence does not match the manifest. The PDFs are not stored in this repository; credit for every paper goes to its authors, as listed on arXiv.
