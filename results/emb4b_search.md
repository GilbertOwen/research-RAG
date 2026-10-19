# Larger embedding model (step 6.4, 2026-10-19)

`qwen3-embedding:0.6b` (Q8_0, vectors of 1,024) compared with `qwen3-embedding:4b` (Q4_K_M, vectors of 2,560). Same chunks (520, 400 words, overlap 0), same query instruction, dense search, top-5; the 4b index is in `data/index_4b`, so the 0.6b index is unchanged.

Rules, fixed before the result was read:
- Gate: run and grade answers only if Hit@5 is at least 17/22 (dense: 16).
- R1: switch only with at least 15 fully correct answers and 3/3 refusals.

## Retrieval (22 answerable questions, checked keys)

| | 0.6b (dense) | Hybrid | MMR | 4b |
|---|---|---|---|---|
| Hit@5 | 16/22 | 15/22 | 9/22 | 16/22 |
| Evidence recall | 0.58 | 0.60 | 0.30 | 0.59 |
| MRR | 0.39 | 0.49 | 0.31 | 0.49 |
| Paper recall, all questions | 0.89 | 0.93 | 0.92 | 0.95 |
| Paper recall, cross-paper and uses-vs-mentions | 0.657 | 0.771 | 0.733 | 0.848 |
| Evidence items in the top 5 (of 39) | 18 | 18 | 9 | 20 |

Hits gained: S04, S13, X01, X04. Hits lost: S05, S09, S15, M01. Rank buckets against 0.6b (`ranks_400_0.json` vs `ranks_400_0_emb4b.json`): 5 of 10 near misses and 1 of 11 deep misses reached the top 5; 4 of 18 top-5 items dropped to near. S13's table page is in the top 5 for the first time.

## Gate

Hit@5 is 16/22, not 17 or more, so the gate fails and the answers are not graded. The answer run was made (`answers_emb4b.json`, `examples/eval_emb4b/`); it refused 3/3 unanswerable questions and refused one answerable question (S09).

## Cost (measured)

| | 0.6b | 4b |
|---|---|---|
| Embedding 520 chunks | 40 s | about 128 s (projected from 64 chunks) |
| Model size | 639 MB | 2.5 GB |
| Answer run, 25 questions | 126 s (hybrid run) | 553 s |
| Seconds per answer | about 4 | about 14 |

With the 4b embedder and the 3.4 GB LLM both needed on a 6 GB GPU, every answer is about three times slower.

## Outcome by the rule

The gate keeps `qwen3-embedding:0.6b` as the default.
