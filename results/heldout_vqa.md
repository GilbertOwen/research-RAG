# Held-out test set: visual question answering papers (2026-10-19)

Every setting of the system (chunk size, overlap, title prefix, top-5, query instruction, models) was chosen on the image captioning questions. To see how it does on papers and questions it was not tuned on, the same frozen configuration was run on a second collection about a different topic: benchmarks, robustness and bias in visual question answering (VQA).

What changed, and only this (`config_vqa.yaml`):

- Paths: `papers/manifest_vqa.csv`, `data/papers_vqa/`, `data/index_vqa/`, `eval/questions_vqa.jsonl`.
- The topic named in the answer prompt: "visual question answering" instead of "image captioning". The retrieval instruction was left unchanged; it already covers vision-language model evaluation.

## Collection and questions

- 10 arXiv papers, all CC BY 4.0 (licence read on the arXiv page of each version, checked again by `scripts/download_papers.py`): 3 benchmarks, 2 robustness studies, 5 debiasing papers (groups in the manifest). 139 pages, 226 chunks.
- 16 questions: 10 single-paper (one per paper), 3 cross-paper, 3 unanswerable. 23 evidence items and 36 key facts; every evidence phrase was checked to be on its stated page. The questions were written for this test after the papers were chosen, then every reference answer and key fact was checked against the page text (`checked: true`).
- Two keys were corrected. VX03 (debiasing methods evaluated on VQA-CP v2) missed ADS-ICT, which the key check found on p.8 of the multiple-choice bias paper; retrieval was rerun with the added evidence (paper recall 0.91 to 0.90, deep misses 3 to 4; Hit@5 and MRR unchanged). VS10 asked for the "two kinds" of language bias named in CIBi's abstract, but the paper's introduction divides it into three (syntactic structure, keyword, context). The answer followed the paper, so the key was corrected and the question text kept as it was run.

## Results, side by side

| | Captioning (tuned on) | VQA (held out) |
|---|---|---|
| Papers / pages / chunks | 20 / 351 / 520 | 10 / 139 / 226 |
| Answerable questions | 22 (15 single, 5 cross, 2 uses-vs-mentions) | 13 (10 single, 3 cross) |
| Hit@5 | 16/22 (73%) | 8/13 (62%) |
| Evidence recall | 0.58 | 0.59 |
| MRR | 0.39 | 0.35 |
| Paper recall | 0.89 | 0.90 |
| Evidence items in the top 5 | 18 of 39 | 11 of 23 |
| Missed items: near (rank 6 to 20) / deep (beyond 20) | 10 / 11 | 8 / 4 |
| Fully / partly / wrong | 13 / 9 / 0 (59% fully) | 9 / 4 / 0 (69% fully) |
| Single-paper fully correct | 12 of 15 | 9 of 10 |
| Multi-paper fully correct (cross and uses-vs-mentions) | 1 of 7 | 0 of 3 |
| Citation support yes / partly | 11 / 11 | 7 / 6 |
| Unanswerable refused | 3/3 | 3/3 |
| Seconds per answer (mean; first call includes model load) | 4.0 | 4.5 |

Sources: `retrieval_vqa.json`, `ranks_vqa.json`, `answers_vqa.json`, `grades_vqa.csv`; answers in `examples/eval_vqa/`. One question is 7.7 percentage points on the VQA set and 4.5 on the captioning set.

## What the held-out set shows

- The same pattern appears on unseen papers: single-paper questions are mostly answered fully (9 of 10), cross-paper questions are only partly answered (0 of 3 fully), and all unanswerable questions are refused.
- Two answers are correct although search missed the expected chunk (VS05, VS06): the retrieval metric is stricter than the task, as on the captioning set (S04, S08).
- Here most missed evidence is a near miss (8 near, 4 deep), unlike the captioning set (10 near, 11 deep).
- One answer added a false number from another paper's table (VS09: CIBi's 57.62% given as CCB's).

## Limits

- 13 answerable questions: differences of one or two questions are within noise.
- The questions were written and checked by the same side (no independent check), and two keys were corrected after the run.
- Different topic, but the same broad field (vision-language), the same models and the same kind of papers (arXiv research papers).
