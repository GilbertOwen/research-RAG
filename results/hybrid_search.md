# Hybrid search (step 6.2, 2026-10-19)

Dense search (vector only) compared with hybrid search: BM25 keyword scores on the plain question, merged with the vector ranking by reciprocal rank fusion (`1 / (60 + rank)` from each list). Everything else is the baseline: 400 words, overlap 0, no title prefix, query instruction, top-5, `qwen3.5:4b`, temperature 0. Code: `src/research_rag/keyword.py`, `src/research_rag/search.py` (`retrieval.mode` in `config.yaml`).

Decision rule, fixed before the run: switch to hybrid only if it gives at least 2 more fully correct answers than dense (15 or more of 22) and still refuses 3/3.

## Retrieval (22 answerable questions, checked keys)

| | Dense | Hybrid |
|---|---|---|
| Hit@5 | 16/22 | 15/22 |
| Evidence recall | 0.58 | 0.60 |
| MRR | 0.39 | 0.49 |
| Paper recall | 0.89 | 0.93 |

Hits gained: S04. Hits lost: S14 (expected page from rank 4 to 6) and M01 (SPECS p.1 from rank 4 to 25). Higher MRR comes from single-paper questions whose evidence moved up (S02, S03, S12 to rank 1).

Evidence items by rank bucket (`ranks_400_0.json` vs `ranks_400_0_hybrid.json`, 39 items):

| Dense bucket | Hybrid: top 5 | near (6 to 20) | deep (beyond 20) |
|---|---|---|---|
| top 5 (18) | 15 | 2 | 1 |
| near (10) | 3 | 3 | 4 |
| deep (11) | 0 | 4 | 7 |

Near misses that reached the top 5: S01 p.3, S04 p.3, X05 p.2. No deep miss reached the top 5; S13's table page moved from rank 26 to 14. Sources: `retrieval_final_checked_keys.json`, `retrieval_hybrid.json`.

## Answers (graded with `eval/rubric.md`)

| | Dense | Hybrid |
|---|---|---|
| Fully correct | 13 | 11 |
| Partly correct | 9 | 11 |
| Wrong | 0 | 0 |
| Single-paper (2 / 1) | 12 / 3 | 11 / 4 |
| Cross-paper (2 / 1) | 1 / 4 | 0 / 5 |
| Uses vs mentions (2 / 1) | 0 / 2 | 0 / 2 |
| Citation support yes / partly | 11 / 11 | 12 / 10 |
| Unanswerable refused | 3/3 | 3/3 |
| Seconds per answer (mean, first call includes model load) | 4.0 | 4.6 |

Grade changes (dense to hybrid):

| ID | Dense | Hybrid | Why |
|---|---|---|---|
| S05 | 1 | 2 | The false claim about low-frequency object pairs is gone |
| S06 | 2 | 1 | The 88 / 52 / 39 counts left out although the source has them; one misread claim |
| S14 | 2 | 1 | Expected page dropped to rank 6; the averaging step is missing |
| X03 | 2 | 1 | Only HallE-Control's view of CHAIR; the clock-on-the-beach paper not used |

X01 and X03 were rechecked: each keeps 1 because the substance of one key fact is present (reasons in the notes). Grading both 0 would change partly / wrong to 9 / 2, not the fully correct count. Per-question notes: `grades_hybrid.csv`; answers: `examples/eval_hybrid/`.

## Outcome by the rule

Hybrid gives 11 fully correct answers, 2 fewer than dense (13), so the rule keeps dense search as the default. Refusals stay 3/3.
