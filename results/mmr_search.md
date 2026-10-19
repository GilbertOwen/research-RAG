# MMR diversity (step 6.3, 2026-10-19)

Dense search compared with maximal marginal relevance (MMR): the top 20 dense results are reordered one pick at a time by `0.7 × similarity to the question − 0.3 × highest similarity to a chunk already picked` (lambda 0.7, the common default, not tuned). Everything else is the baseline. Code: `mmr` mode in `src/research_rag/search.py`.

Rules, fixed before the run:
- D1: run and grade answers only if mean paper recall on the 7 cross-paper and uses-vs-mentions questions beats dense (0.657).
- R1: switch to MMR only with at least 15 fully correct answers and 3/3 refusals.

## Retrieval (22 answerable questions, checked keys)

| | Dense | Hybrid | MMR |
|---|---|---|---|
| Hit@5 | 16/22 | 15/22 | 9/22 |
| Evidence recall | 0.58 | 0.60 | 0.30 |
| MRR | 0.39 | 0.49 | 0.31 |
| Paper recall, all questions | 0.89 | 0.93 | 0.92 |
| Paper recall, cross-paper and uses-vs-mentions | 0.657 | 0.771 | 0.733 |

Hits lost: S02, S03, S05, S06, S07, S14, M01; none gained. Evidence items in the top 5 fell from 18 to 9 (19 near, 11 deep). The penalty pushes out the second page of the right paper: two pages of one paper are more similar to each other than their question scores differ (S02: five different papers in the top 5, THumB p.1 at rank 12). Gate D1 passed (0.733 > 0.657), so the answers were run. Sources: `retrieval_mmr.json`, `ranks_400_0_mmr.json`.

## Answers (graded with `eval/rubric.md`)

| | Dense | Hybrid | MMR |
|---|---|---|---|
| Fully correct | 13 | 11 | 11 |
| Partly correct | 9 | 11 | 9 |
| Wrong | 0 | 0 | 1 (S05) |
| Refused an answerable question | 0 | 0 | 1 (S02) |
| Single-paper (2 / 1 / 0 or refused) | 12 / 3 / 0 | 11 / 4 / 0 | 10 / 3 / 2 |
| Cross-paper (2 / 1) | 1 / 4 | 0 / 5 | 1 / 4 |
| Uses vs mentions (2 / 1) | 0 / 2 | 0 / 2 | 0 / 2 |
| Citation support yes / partly | 11 / 11 | 12 / 10 | 12 / 9 |
| Unanswerable refused | 3/3 | 3/3 | 2/3 |

U01 said the sources have no ReFormer GPU or training time, mentioned another study's GPUs, then gave NOT ENOUGH EVIDENCE: correct in substance but not the exact refusal, so it counts as not refused. X04 is the first fully correct X04 (all three CLIPScore weaknesses in different papers' chunks). Per-question notes: `grades_mmr.csv`; answers: `examples/eval_mmr/`.

## Outcome by the rule

MMR gives 11 fully correct answers and 2/3 refusals, so R1 keeps dense search as the default.
