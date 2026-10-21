# research-rag

A local retrieval-augmented assistant for research papers on image captioning and vision-language model evaluation. Answers come only from a curated collection of openly licensed arXiv papers and cite the paper and page they come from. Everything runs locally with Ollama on a 6 GB laptop GPU; no API keys are needed.

Results at a glance, on 22 answerable test questions and 3 unanswerable ones:

- Retrieval: the expected evidence is among the 5 retrieved passages for 16 of 22 questions.
- Answers: 13 fully correct, 9 partly correct, 0 wrong; all 3 unanswerable questions refused.
- Without retrieval, the same model got 0 fully correct and invented answers to all 3 unanswerable questions.
- On a held-out collection of 10 papers on another topic (visual question answering), the same frozen system got 9 of 13 fully correct, 0 wrong, and refused all 3 unanswerable questions.
- Three retrieval changes were tested afterwards, each against a rule fixed before the run. Hybrid keyword search and MMR diversity gave fewer fully correct answers (11 each); a larger embedding model ranked evidence higher but found evidence for no more questions (16 of 22), so its answers were not graded. The baseline stays.

Contents:
[Problem](#problem) ·
[Document collection](#document-collection) ·
[Architecture](#architecture) ·
[Models and tools](#models-and-tools) ·
[Setup and running](#setup-and-running) ·
[Configuration](#configuration) ·
[Example queries](#example-queries) ·
[Evaluation and results](#evaluation-and-results) ·
[Design decisions](#design-decisions-and-experiments) ·
[Optional exploration](#optional-exploration) ·
[Limitations](#limitations) ·
[Next steps](#next-steps) ·
[AI use](#ai-use) ·
[Credits](#credits)

## Problem

Reading up on a research topic means checking many papers for small, exact details: which metric a paper uses, how large a dataset is, what limitation the authors admit. Searching 20 PDFs by hand for each question is slow. Asking a general chatbot is faster but risky, because it answers from memory and fills gaps with details that sound plausible.

That risk shows up clearly on this collection. Asked the 22 test questions with no sources, `qwen3.5:4b` got none fully correct (6 partly, 16 wrong). It also answered all three unanswerable questions instead of saying it did not know. For example, it said the CLIP paper reports a CIDEr score of about 41.1 on MS-COCO; the CLIP paper has no captioning results, and 41.1 appears in it only as a zero-shot EuroSAT accuracy (`results/closed_book_vs_rag.md`).

research-rag answers only from passages retrieved from the collection, cites the paper and page for every claim so the reader can check it, and replies NOT ENOUGH EVIDENCE when the retrieved passages do not contain the answer.

## Document collection

20 arXiv papers on image captioning and its evaluation, listed in [`papers/manifest.csv`](papers/manifest.csv) with the exact version used: background surveys (2), captioning methods (6), news captioning with context (1), caption evaluation and bias in metrics (6), and object hallucination (5). 351 pages in total, about 87 MB of PDF. Each paper is used under its own licence (19 under CC BY 4.0, one under CC0 1.0). Famous papers such as Attention Is All You Need, ViT, BERT and CLIP were left out because their arXiv licence is not CC BY. The PDFs are not stored in this repository; credit for every paper goes to its authors, as listed on arXiv.

A second collection, 10 CC BY 4.0 papers on visual question answering ([`papers/manifest_vqa.csv`](papers/manifest_vqa.csv)), is used only as a held-out test ([Held-out test set](#held-out-test-set)).

Preprocessing (`src/research_rag/ingest.py`):

- Page text is read in reading order from PyMuPDF text blocks, so two-column papers come out in the right order.
- Images, bare page numbers and the arXiv margin stamp are removed; ligatures are normalised.
- A hyphen at the end of a line is joined (`necessar-` + `ily`), unless the paper writes the hyphenated word elsewhere (`vision-language` is kept).
- Reference lists are dropped; appendices after them are kept.
- Each page is split into windows of 400 words with no overlap, and a chunk never crosses a page, so every chunk has one page number to cite. A short leftover at the end of a page is joined to the previous chunk.
- Result: 520 chunks, each with an id `paper:page:chunk` (for example `2111.08940v2:p3:c0`).

### Getting the PDFs

The download script is the recommended way:

```
uv run python scripts/download_papers.py
```

It reads the licence on the arXiv page of each version, skips any paper whose licence does not match the manifest, and skips files that are already downloaded.

You can also download the PDFs from arXiv yourself and drop them into `data/papers/`. In that case:

- Name each file exactly as the manifest builds it, `<arxiv_id><version>.pdf`, for example `2111.08940v2.pdf`. The ingest step reads only the papers listed in the manifest, ignores any other PDF in the folder, and stops with an error if a listed paper is missing.
- Use the version in the manifest. A different version can have different page numbers, so the citations and the evaluation results (which expect specific pages) may not match.
- The licence check is skipped for files you add by hand; checking the licence is then up to you.
- To add a paper of your own, add a row to the manifest (`arxiv_id`, `version`, `group`, `title`, `license`) and run ingest and embed again. The test questions only cover the 20 papers above.

## Architecture

```
PDFs ──ingest──> chunks.jsonl ──embed──> embeddings.npy
                                              │
question ──(instruction + embed)──> search: top 5 by cosine similarity
                                              │
                    generate: prompt with sources [S1] to [S5] ──> qwen3.5:4b ──> answer
                                              │
                    citations: check the labels; map each label back to paper and page
```

| Step | File | What it does | Output |
|---|---|---|---|
| Download | `scripts/download_papers.py` | Fetches the manifest's PDFs, checking each licence | `data/papers/*.pdf` |
| Ingest | `src/research_rag/ingest.py` | Clean page text, reference lists removed, 400-word chunks within a page | `data/index/chunks.jsonl` |
| Embed | `src/research_rag/embed.py` | One vector of 1,024 numbers per chunk with `qwen3-embedding:0.6b` | `data/index/embeddings.npy`, `embeddings_meta.json` |
| Search | `src/research_rag/search.py` | Embeds the question with a retrieval instruction and returns the 5 closest chunks | ranked chunks with scores |
| Generate | `src/research_rag/generate.py` | Puts the 5 chunks into the prompt as [S1] to [S5] and asks `qwen3.5:4b` to answer only from them | answer text |
| Check citations | `src/research_rag/citations.py` | Finds missing, invalid or uncited labels in the answer | citation report |
| Ask | `src/research_rag/ask.py` | Runs search, generate and the citation check for one question, prints the answer with its sources | `examples/*.json` |

Design points:

- Vector search is plain NumPy, with no vector database: the vectors have length 1, so a dot product with the question vector is the cosine similarity. 520 vectors need nothing more.
- The LLM never writes page numbers. It cites short labels ([S1] to [S5]), and code maps each label back to the paper and page, so a citation cannot point to a page that was not retrieved.
- `search.py` stops if the vectors do not match `chunks.jsonl`, or if the index was built with a different embedding model than the one in `config.yaml`.
- `search.py` also has two optional retrieval modes from the experiments, `hybrid` (BM25 keyword scores from `keyword.py`, fused with the vector ranking) and `mmr` (diversity reranking). The default is `dense`; see [Optional exploration](#optional-exploration).
- Evaluation scripts (`src/research_rag/evaluate.py` and the scripts in `scripts/`) reuse the same functions, so the evaluated pipeline is the one a user runs.

## Models and tools

| Part | Choice and version | Why |
|---|---|---|
| Answer model (LLM) | `qwen3.5:4b` via Ollama (3.4 GB, 4-bit `Q4_K_M`) | Fits fully in a 6 GB laptop GPU; free and local, so a reviewer needs no API key |
| Embedding model | `qwen3-embedding:0.6b` via Ollama (639 MB, vectors of 1,024 numbers) | Small and fast (the 520 chunks embed in 40 s) and made for retrieval with a task instruction. The 4B version was tested: it ranked evidence higher but found evidence for no more questions and made answers about three times slower on this GPU ([Optional exploration](#optional-exploration)) |
| Model server | Ollama 0.35.1 | Runs both models locally behind one API |
| Python | 3.12.13, packages managed with [uv](https://docs.astral.sh/uv/) | `uv.lock` pins every version; one command installs everything |
| PDF text | PyMuPDF 1.28.2 | Text blocks with positions, so two-column papers can be read in order |
| Vector search | NumPy 2.5.4 | No framework or vector database: a dot product over 520 vectors is enough, and every step can be explained |
| Other | `ollama` 0.6.3 (Python client), PyYAML 6.0.3 (config) | |

Generation settings: temperature 0 so answers repeat, a context window of 8,192 tokens, thinking mode off. Over the 25 evaluation questions the prompts used 1,864 to 6,795 tokens (mean 3,054), all inside the window. Measured on an RTX 3060 Laptop GPU (6 GB VRAM) with 16 GB RAM on Windows 11: about 4 seconds per answer; the first question takes longer while the model loads.

## Setup and running

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

Download the papers (20 PDFs, about 87 MB, into `data/papers/`; see [Getting the PDFs](#getting-the-pdfs) for adding them by hand):

```
uv run python scripts/download_papers.py
```

Build the index (clean text and chunks, then one vector per chunk; under a minute after the models load):

```
uv run python -m research_rag.ingest
uv run python -m research_rag.embed
```

Ask a question:

```
uv run python -m research_rag.ask "Which dataset was SPECS trained on?"
```

```
SPECS was trained on the ShareGPT-4V dataset [S2]. This dataset contains 1.2 million high-quality
image-caption pairs sourced from COCO, SAM, and LAION [S2]. The captions in this dataset are
approximately 143 tokens long on average [S2].

Sources (* = cited):
 [S1] SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluati, page 9  (score 0.737)
*[S2] SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluati, page 5  (score 0.718)
 [S3] SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluati, page 6  (score 0.718)
 ...

Citation check: ok, every sentence cites a retrieved source
```

`--debug` also prints the text of each source. Every run is saved as JSON in `examples/`; `--no-save` turns that off.

Reproduce the evaluation (use a new label so the saved results are not overwritten):

```
uv run python -m research_rag.evaluate --label rerun
uv run python scripts/run_answers.py --label rerun
```

The first prints Hit@5, evidence recall, MRR and paper recall for the 22 answerable questions; the second answers all 25 questions and saves them to `examples/eval_rerun/` for grading.

### The held-out collection

The visual question answering collection has its own config, `config_vqa.yaml`. Set `RAG_CONFIG` and run the same commands:

```
$env:RAG_CONFIG = "config_vqa.yaml"      # PowerShell; in bash: export RAG_CONFIG=config_vqa.yaml
uv run python scripts/download_papers.py
uv run python -m research_rag.ingest
uv run python -m research_rag.embed
uv run python -m research_rag.evaluate --label vqa_rerun
```

Unset it afterwards (`Remove-Item Env:RAG_CONFIG` or `unset RAG_CONFIG`) to go back to the captioning collection.

## Configuration

Settings are in `config.yaml`. The Ollama address can be changed through `OLLAMA_HOST` (see `.env.example`). No API keys are needed. The commented lines in `config.yaml` are earlier values, kept to show how the settings changed.

| Block | Key | Value | What it controls |
|---|---|---|---|
| `models` | `llm` | `qwen3.5:4b` | The answer model |
| | `embedding` | `qwen3-embedding:0.6b` | The model that turns chunks and questions into vectors |
| | `think` | `false` | Qwen's thinking mode, off for short direct answers |
| `chunking` | `size_words` | `400` | Words per chunk ([Design decisions](#design-decisions-and-experiments)) |
| | `overlap_words` | `0` | Words repeated between neighbouring chunks |
| `embedding` | `add_title` | `false` | Whether the paper title is put before each chunk before embedding |
| | `batch_size` | `32` | Chunks sent to the embedding model at once |
| `retrieval` | `top_k` | `5` | Sources given to the answer model |
| | `mode` | `dense` | `dense` (vector search), or the tested alternatives `hybrid` and `mmr` |
| | `rrf_k` | `60` | Rank fusion constant for `hybrid` |
| | `pool`, `mmr_lambda` | `20`, `0.7` (default) | Candidate pool and relevance weight for `mmr` |
| | `query_instruction` | one sentence | Added before each question when it is embedded (not before the chunks) |
| `generation` | `temperature` | `0` | Repeatable answers |
| | `num_ctx` | `8192` | Context window in tokens |
| | `no_evidence_reply` | `NOT ENOUGH EVIDENCE` | The exact reply when the sources do not answer the question |
| | `topic` | `image captioning` (default) | The topic named in the answer prompt |
| `paths` | `manifest`, `papers_dir`, `index_dir` | `papers/manifest.csv`, `data/papers`, `data/index` | Where the paper list, the PDFs and the index are |
| | `questions` | `eval/questions.jsonl` (default) | The test questions used by the evaluation scripts |

Changing `chunking` or `embedding` means running `ingest` and `embed` again; `search.py` refuses to run if the index was built with another embedding model or no longer matches the chunks. `RAG_CONFIG` selects another config file, such as `config_vqa.yaml` for the held-out collection.

## Example queries

Four of the 25 evaluation runs, chosen to show a clean answer, a cross-paper answer, a retrieval failure and a refusal. All 25 runs, with the full source text, are in [`examples/eval_baseline/`](examples/eval_baseline/); grades and notes are in [`results/grades_baseline.csv`](results/grades_baseline.csv).

### 1. A single-paper fact (S06): works

**Question:** What is the PAO-EvalBias dataset and how large is it?

**Retrieved** (all from *Gender Biases in Automatic Evaluation Metrics for Image Captioning*): p.4, p.3, p.2 (cited), p.2 (cited), p.9.

**Answer:**
> The PAO-EVALBIAS dataset consists of 92,049 images featuring people with various professions, activities, and objects [S4]. It was constructed using templates to create reference sentences paired with candidate captions containing correct or incorrect genders [S3]. The dataset includes 88 professions, 52 activities, and 39 objects to systematically analyze gender biases in evaluation metrics [S4].

**Verdict:** fully correct, every claim supported by the cited page. All five key facts (92,049 images, 88 professions, 52 activities, 39 objects, built to study gender bias in metrics) come from page 2, which search ranked third and fourth.

### 2. A cross-paper question (X05): partly works

**Question:** How do the papers in the collection study social bias in image captioning?

**Retrieved:** *Understanding and Evaluating Racial Biases in Image Captioning* p.1 (cited) and p.4 (cited); *Gender Biases in Automatic Evaluation Metrics* p.1 (cited), p.9 (cited) and p.4.

**Answer:**
> The papers study social bias by analyzing racial and gender biases through manual annotations of skin color and perceived gender [S1]. They also investigate how model-based evaluation metrics like CLIPScore encode societal biases that can amplify these issues during generation [S2, S3]. Additionally, researchers examine disparate performance across demographics to understand bias propagation pathways within captioning systems [S1, S4].

**Verdict:** partly correct. Search found both papers, and the racial bias study is described correctly. The second study is described only as "societal biases" in CLIPScore: the answer does not say that it is about gender or that it uses the PAO-EvalBias dataset, although the retrieved page 1 and page 4 of that paper both say so. This is a generation problem: the evidence was there, but the answer left out the key detail.

### 3. A retrieval failure (S13): struggles

**Question:** Which Indonesian image captioning datasets are listed in the multilingual survey of attention-based models?

**Retrieved:** the multilingual survey p.4, p.1, p.20 (cited), p.3, and *Beyond RNNs* p.8.

**Answer:**
> The multilingual survey lists the **Indonesian FEEH-ID** dataset as one of the benchmarks for evaluating multilingual image captioning performance [S3]. This dataset is mentioned alongside Bengali BORNON to highlight challenges in generating captions for non-English languages [S3]. The survey also notes that achieving high performance on such culturally diverse datasets remains a significant area for future research [S3].

**Verdict:** partly correct. The survey lists three Indonesian datasets (Flickr8k Bahasa, FEEH-ID and an Indonesian Flickr8k) in a results table on page 12, but that page ranked 26th, so the answer could only name the one dataset page 20 describes. About 300 of the table page's 400 words are numbers, "NA" entries or reference markers, so its vector says little about "Indonesian". This is a retrieval problem, not a generation problem: what the answer says is supported by its source.

### 4. A question the collection cannot answer (U03): refuses

**Question:** What CIDEr score does the original CLIP paper report on MS-COCO?

**Retrieved:** pages that mention CLIPScore and CIDEr (the gender bias paper p.8 twice, the 2018 survey p.23, SPECS p.8, THumB p.6), but nothing from the CLIP paper, which is not in the collection.

**Answer:**
> NOT ENOUGH EVIDENCE

**Verdict:** correct refusal. Without retrieval, the same model answered this question with "a CIDEr of about 41.1"; the CLIP paper has no captioning results, and 41.1 appears in it only as a zero-shot EuroSAT accuracy.

## Evaluation and results

### Test set

25 questions in [`eval/questions.jsonl`](eval/questions.jsonl):

- 15 single-paper questions;
- 5 cross-paper questions that need two or more papers;
- 2 "uses vs mentions" questions (which papers *use* CLIP or GPT-4 in their own method, not just cite it);
- 3 unanswerable questions, whose answer is not in the collection.

Each answerable question has its expected evidence (paper, page and a phrase that must appear in the retrieved chunk; 39 items in total) and its key facts (63 in total), which were fixed before any answer was graded. The answer keys started as drafts and were checked against the page text before grading. Six questions changed in that check (S01, S02, S03, X02, M01, M02), so retrieval numbers are reported with the checked keys; with the draft keys the same index scored 14/22 instead of 16/22, a correction of the measurement, not an improvement of the system.

### How answers are graded

The rubric is in [`eval/rubric.md`](eval/rubric.md). Each answer gets two marks:

- **Correctness**: 2 when every key fact is present and nothing false is claimed; 1 when at least one key fact is present, or all of them plus a false side claim; 0 when none is present or the main fact is wrong. Refusing an answerable question counts as a failure; refusing an unanswerable one is correct.
- **Citation support**: yes, partly or no, judged by reading the cited passage, not by matching words.

### Retrieval

| Metric | What it measures | Result |
|---|---|---|
| Hit@5 | At least one expected evidence chunk (right paper, page and phrase) among the 5 sources | 16/22 (0.73) |
| Evidence recall | Share of a question's expected evidence that was found | 0.58 |
| MRR | 1 / rank of the first correct chunk, averaged (0 when none is found) | 0.39 |
| Paper recall | Share of the expected papers among the 5 sources, any page | 0.89 |

Source: [`results/retrieval_final_checked_keys.json`](results/retrieval_final_checked_keys.json). The phrase-based hit is stricter than the task: S04 and S08 were answered fully correctly from pages other than the expected ones.

### Answers

| | Fully correct (2) | Partly (1) | Wrong (0) |
|---|---|---|---|
| All 22 answerable | 13 | 9 | 0 |
| Single-paper (15) | 12 | 3 | 0 |
| Cross-paper (5) | 1 | 4 | 0 |
| Uses vs mentions (2) | 0 | 2 | 0 |

- Unanswerable questions: 3 of 3 refused with NOT ENOUGH EVIDENCE.
- Citation support: 11 yes, 11 partly, 0 no.
- Grades and notes: [`results/grades_baseline.csv`](results/grades_baseline.csv).

### Does retrieval help? A closed-book comparison

The same model answered the same questions with no sources, and its answers were graded the same way ([`results/closed_book_vs_rag.md`](results/closed_book_vs_rag.md)):

| | Fully correct | Partly | Wrong | Unanswerable refused |
|---|---|---|---|---|
| Closed-book | 0 | 6 | 16 | 0 of 3 |
| With retrieval | 13 | 9 | 0 | 3 of 3 |

All 13 fully correct answers needed retrieval: the model got none of them fully right on its own. Closed-book never used the "I DON'T KNOW" reply it was offered. Its invented details were checked against the papers and other sources: the CIDEr of 41.1 for the CLIP paper is wrong, CHAIR is attributed to the wrong authors, and the claimed ReFormer setup (a V100 GPU, about 24 hours) is not stated anywhere in the paper.

### Where the system fails

[`results/failure_log.md`](results/failure_log.md) traces every answer below full marks to one cause:

| Cause | Count | Examples |
|---|---|---|
| Retrieval: the evidence was not among the 5 sources | 5 | S13 (table page), X01, X02, M01, M02 |
| Generation: the evidence was there, but a detail was missed or misread | 4 | S05, S15, X04, X05 |
| Ingestion: broken PDF text | 0 | |

All retrieval failures are on cross-paper or uses vs mentions questions, plus one table page. Four fully correct answers still had a citation problem (a sentence cited to the wrong page, or not cited at all).

### Held-out test set

All settings above were chosen on these 25 questions. To check how the frozen system does on papers it was not tuned on, it was also run on a second collection: 10 CC BY 4.0 papers on visual question answering (benchmarks, robustness and bias), with 16 new questions (10 single-paper, 3 cross-paper, 3 unanswerable). Only the file paths and the topic word in the answer prompt changed ([`config_vqa.yaml`](config_vqa.yaml)).

| | Captioning (tuned on) | VQA (held out) |
|---|---|---|
| Hit@5 | 16/22 | 8/13 |
| Fully / partly / wrong | 13 / 9 / 0 | 9 / 4 / 0 |
| Single-paper fully correct | 12 of 15 | 9 of 10 |
| Multi-paper fully correct (cross-paper and uses vs mentions) | 1 of 7 | 0 of 3 |
| Unanswerable refused | 3 of 3 | 3 of 3 |

The same pattern holds on the unseen papers: single-paper questions are mostly answered fully, cross-paper questions only partly, and every unanswerable question is refused. Details, including two answer keys corrected during checking: [`results/heldout_vqa.md`](results/heldout_vqa.md).

## Design decisions and experiments

### Chunk size and overlap

We started with 250 words per chunk because a typical paragraph explains one idea, such as a definition, the main topic or the purpose of a passage. In our papers a typical paragraph has about 58 words, so a 250-word chunk holds about three to four paragraphs. We started with 50 words of overlap because it guarantees that any passage of up to 50 words stays whole inside at least one chunk, even when a chunk boundary falls in the middle of it.

That starting point was then tested rather than kept on reasoning alone. Every setting was measured on the 22 answerable questions with everything else fixed, and each decision rule was written down before its run.

| Experiment | Result (Hit@5 of 22, draft keys) | Rule fixed before the run | Decision |
|---|---|---|---|
| Query instruction on / off (250 / 50) | 9 / 8 | | Kept: the difference is one question, and the instruction is the embedding model's intended use |
| Paper title before each chunk on / off (250 / 50) | 9 / 12, MRR 0.22 / 0.32 | | Removed: the title made chunks of one paper look alike, so the wrong page of the right paper ranked first |
| Chunk size 100 / 175 / 250 / 400 / 600 words, overlap 20% | 7 / 9 / 12 / 14 / 12 | Smallest size within one question of the best | 400 words |
| Overlap at 400 words: 0 / 40 / 80 / 120 | 14 / 14 / 14 / 15 | Smallest overlap within one question of the best | 0 words |

Chunking was judged by retrieval quality on the questions, not by a clustering score such as the silhouette score: chunks can form clean clusters and still split an answer in two.

### Checking the choice on answers

Retrieval scores alone do not show whether the answers get better, so the final setting and the two strongest alternatives were compared end to end on all 25 questions ([`results/compare_settings.md`](results/compare_settings.md)):

| Setting | Fully correct | MRR (checked keys) | Prompt tokens (mean) | Seconds per answer |
|---|---|---|---|---|
| 400 / 0 | 13 | 0.39 | 3,054 | 4.0 |
| 400 / 80 | 13 | 0.50 | 3,142 | 4.3 |
| 250 / 50 | 11 | 0.43 | 2,300 | 3.2 |

The rule, fixed before the run, was to keep 400 / 0 unless another setting had at least 2 more fully correct answers and still refused all 3 unanswerable questions. Neither did, so 400 / 0 stayed. 400 / 80 ranks evidence higher and ties on answers; it is a reasonable alternative.

### Other decisions

- **PDFs, not HTML or Markdown:** more general, since PDF is the format papers are shared in; the cost is cleaning (reading order, hyphens, reference lists).
- **Only openly licensed papers:** every paper is CC BY 4.0 or CC0 for its exact arXiv version, checked by the download script, so the collection can be rebuilt and shared.
- **Chunks never cross a page:** every chunk has exactly one page to cite.
- **Short source labels instead of page numbers:** the model cites [S1] to [S5] and code maps them to paper and page, so it cannot invent a page.
- **Refusal left to the model:** the search score cannot tell a missing answer from a present one (the top score for the unanswerable questions, 0.68 to 0.72, falls inside the range for answerable ones, 0.62 to 0.82), so a score threshold would not work; the prompt asks for an exact NOT ENOUGH EVIDENCE reply instead.
- **Temperature 0:** the same question gives the same answer, so every run can be compared and repeated.

## Optional exploration

The weakest part of the baseline is cross-paper retrieval: all five retrieval failures are cross-paper, uses vs mentions or table questions. The exploration asked two things: where does the missing evidence actually rank, and can one retrieval change bring it into the top 5 without making the answers worse? Each step had its decision rule written down before it ran. Dense search was kept unless a change gave at least 2 more fully correct answers (15 or more of 22) and still refused all 3 unanswerable questions.

### 1. Diagnosis: where does the missing evidence rank?

[`scripts/diagnose_ranks.py`](scripts/diagnose_ranks.py) searches all 520 chunks for every question and records the rank of each of the 39 expected evidence items ([`results/ranks_400_0.json`](results/ranks_400_0.json)):

- 18 are in the top 5, 10 are near misses (rank 6 to 20), 11 are deep misses (beyond rank 20), and none is missing from the index.
- 13 of the 21 misses come from a paper that already has another page in the top 5: the right paper, but the wrong page.
- On the multi-paper questions, 19 of the 21 expected papers are somewhere in the top 20.

The rule fixed before the run pointed to keyword search if deep misses outnumbered near misses, and to diversity otherwise. It picked keyword search, by 11 to 10.

### 2. Hybrid search (keyword + vector)

BM25 keyword scores on the plain question ([`src/research_rag/keyword.py`](src/research_rag/keyword.py)), merged with the vector ranking by reciprocal rank fusion (`1 / (60 + rank)` from each list). Results: [`results/hybrid_search.md`](results/hybrid_search.md).

- 3 of the 10 near misses reached the top 5, none of the deep misses did, and 3 items that were already in the top 5 dropped out.
- MRR rose from 0.39 to 0.49, but fully correct answers fell from 13 to 11.
- Cross-paper questions use common words (CLIP, GPT-4, BLEU), so keyword matching did not single out the right pages.

### 3. Diversity (MMR)

Maximal marginal relevance reorders the top 20, preferring chunks that are relevant but not too similar to chunks already picked (relevance weight 0.7, not tuned). It was run only after it passed a gate: paper recall on the 7 multi-paper questions had to beat dense search (0.733 against 0.657). Results: [`results/mmr_search.md`](results/mmr_search.md).

- Hit@5 fell from 16 to 9: two pages of the same paper are so similar to each other that the penalty pushed out the right paper's second page.
- Fully correct answers fell to 11, one answerable question was refused, and one unanswerable question got a hedged answer instead of the exact refusal.

### 4. A larger embedding model

`qwen3-embedding:4b` instead of `0.6b`, same chunks, in a separate index. A timing check first projected about 2 minutes to embed the 520 chunks. Answers were to be graded only if Hit@5 reached at least 17. Results: [`results/emb4b_search.md`](results/emb4b_search.md).

- It ranked evidence best of all settings (MRR 0.49, paper recall 0.95, 0.848 on multi-paper questions, 20 of 39 items in the top 5), but found evidence for no more questions (16 of 22).
- The gate failed, so its answers were not graded; nothing is claimed about its answer quality.
- With both the embedding model (2.5 GB) and the answer model (3.4 GB) on a 6 GB GPU, answers took about 14 seconds instead of 4.

### Summary

| Setting | Hit@5 | MRR | Paper recall (all / multi-paper) | Evidence in top 5 | Answers 2 / 1 / 0 | Unanswerable refused |
|---|---|---|---|---|---|---|
| Dense, 0.6b (kept) | 16 | 0.39 | 0.89 / 0.657 | 18 | 13 / 9 / 0 | 3/3 |
| Hybrid | 15 | 0.49 | 0.93 / 0.771 | 18 | 11 / 11 / 0 | 3/3 |
| MMR | 9 | 0.31 | 0.92 / 0.733 | 9 | 11 / 9 / 1, plus 1 refused | 2/3 |
| 4b embedder | 16 | 0.49 | 0.95 / 0.848 | 20 | not graded (gate) | 3/3 (run, not graded) |

Each change improved at least one ranking number. Neither of the two whose answers were graded improved the answers, and the larger embedding model found evidence for no more questions. Because every decision rule was based on the answers and fixed in advance, none of them was adopted on a ranking number alone. The baseline stays: dense search with `qwen3-embedding:0.6b` (`retrieval.mode: dense`). The other modes stay in the code so the results can be reproduced. With 22 questions, the differences of one or two answers are within noise. The diagnosis itself is the main finding: most missed evidence is in the right paper on another page, which none of the three changes addressed directly.

## Limitations

- **Small test sets.** 22 answerable and 3 unanswerable questions on captioning, 13 and 3 on VQA. One question is 4.5 percentage points on the first set and 7.7 on the second, so differences of one or two questions are within noise.
- **Tuned and reported on the same questions.** All settings were chosen on the captioning questions. The held-out VQA set reduces this, but its questions were written and checked by the same side, without an independent check, and two of its keys were corrected after the run.
- **One grader.** Every answer was graded once against fixed key facts, with a spot check, not by independent graders.
- **Weak cross-paper answers.** 1 of 7 multi-paper questions fully correct on captioning, 0 of 3 on VQA. Most missed evidence is the right paper on the wrong page, and none of the tested retrieval changes fixed that.
- **Generation slips.** Some answers are right except for one missed or wrong detail, even when the evidence is in the sources (S05, S15, X04, X05; VQA: VS09 took a number from another paper's table).
- **The citation checker is shallow.** It proves that labels exist and point to retrieved sources, not that the cited text supports the claim. It splits a sentence at "Focus!", flags a sentence that starts with its label, and does not recognise labels written without brackets (M02).
- **Tables and appendices.** Table pages come out as rows of numbers and embed poorly (S13's table page ranked 26th). Appendix text before the first prose block is skipped in some papers (SPECS, ROPE, CapArena, F-CLIPScore), and one reference fragment remains.
- **The retrieval metric is stricter than the task.** Some answers are correct from a page other than the expected one (S04, S08; VS05, VS06), so Hit@5 slightly understates the system.
- **Refusal depends on the prompt.** Search scores for unanswerable questions overlap with answerable ones, so the system relies on the model to say NOT ENOUGH EVIDENCE.
- **Hardware.** On a 6 GB GPU, a larger embedding model makes answers about three times slower, and a larger answer model would not fit fully in VRAM.
- **Scope.** English arXiv PDFs on two topics in the same field (vision-language).

## Next steps

Each of these follows from a finding above.

1. **Find the right page, not just the right paper.** Most missed evidence is a different page of a paper that is already retrieved. A reranker that scores the question and each of the top 20 chunks together (a cross-encoder) could pick the right page from the right paper, which neither keyword search nor diversity did.
2. **Split cross-paper questions.** On the multi-paper questions, 19 of 21 expected papers are already in the top 20. Breaking a question such as "how do the papers study social bias?" into one search per sub-question or per paper could bring each paper's evidence into the sources.
3. **Index tables separately.** Turn each table into a short text chunk with its caption and row labels, so a question about "Indonesian datasets" can match the table page (S13).
4. **Check that citations support their claims.** Add a second check that asks whether each cited passage actually states the claim, and fix the sentence splitter and bare-label handling (S10, X03, M02).
5. **Compare a larger answer model.** Test `qwen3.5:9b` on the generation failures, accepting that it does not fit fully in 6 GB VRAM.
6. **A bigger, independently checked test set.** More questions, especially cross-paper ones, written and graded by someone other than the system's builder, so differences of a few points can be trusted.

## AI use

I used Generative AI in these parts of the work:

* Help building the code based on the structure that I want
* Help debugging
* Help giving the risk of my decision
* Drafting the summary points
* Help converting the essential information that have been kept as a decision point to summary
* Searching arXiv for candidate papers and checking each paper's licence.
* Drafting the test questions, answer keys and key facts, for both the captioning set and the held-out VQA set.
* Grading the answers against the rubric.
* Drafting the README text, including the result summaries.

## Credits

- **Papers:** the 20 captioning papers and the 10 VQA papers, credited to their authors as listed on arXiv. Each is used under its own licence (CC BY 4.0 or CC0 1.0), recorded per version in [`papers/manifest.csv`](papers/manifest.csv) and [`papers/manifest_vqa.csv`](papers/manifest_vqa.csv). The PDFs are not included in this repository.
- **Models:** Qwen3.5 and Qwen3-Embedding by the Qwen team, run through [Ollama](https://ollama.com/).
- **Libraries:** [PyMuPDF](https://pymupdf.readthedocs.io/) (AGPL-3.0), [NumPy](https://numpy.org/), [ollama-python](https://github.com/ollama/ollama-python), [PyYAML](https://pyyaml.org/) and [uv](https://docs.astral.sh/uv/).
