# Settings comparison on answers (step 5.5, 2026-10-18)

All 25 questions answered with three chunk settings; everything else fixed (no title prefix, query instruction, top-5, `qwen3.5:4b`, temperature 0, same prompt). Graded with `eval/rubric.md`; grades in `grades_baseline.csv`, `grades_400_80.csv`, `grades_250_50.csv`. Automatic numbers from `compare_settings.json`.

Decision rule, fixed before the run: keep 400 / 0 unless another setting has at least 2 more fully correct answers (of 22) and still refuses all 3 unanswerable questions.

## Results

| | 400 / 0 (primary) | 400 / 80 | 250 / 50 |
|---|---|---|---|
| Fully correct (2) | 13 | 13 | 11 |
| Partly correct (1) | 9 | 9 | 9 |
| Wrong (0) | 0 | 0 | 2 |
| Citation support yes / partly / no | 11 / 11 / 0 | 12 / 10 / 0 | 13 / 8 / 1 |
| Unanswerable refused | 3 / 3 | 3 / 3 | 3 / 3 |
| Search hits (checked keys) | 16 / 22 | 16 / 22 | 14 / 22 |
| Mean prompt tokens | 3,054 | 3,142 | 2,300 |
| Mean seconds per answer | 4.0 | 4.3 | 3.2 |

By type (fully correct): single-paper 12 / 12 / 11 of 15; cross-paper 1 / 1 / 0 of 5; uses-vs-mentions 0 / 0 / 0 of 2.

## Per-question differences from 400 / 0

| Setting | Better | Worse |
|---|---|---|
| 400 / 80 | S05 (no false claim), X05 (both studies covered) | S14 (formula misstated), X03 (clock paper not used) |
| 250 / 50 | S05 (no false claim) | S06 (counts left out), S10 and S13 (evidence not retrieved, answered NOT ENOUGH EVIDENCE), X03 (clock paper not used) |

## Retrieval on the same runs (checked answer keys)

Recomputed from the five sources saved with each answer, so all three use the checked keys (the earlier overlap sweep used draft keys).

| | Hit@5 | Evidence recall | MRR | Paper recall |
|---|---|---|---|---|
| 400 / 0 | 16 / 22 | 0.58 | 0.39 | 0.89 |
| 400 / 80 | 16 / 22 | 0.60 | 0.50 | 0.91 |
| 250 / 50 | 14 / 22 | 0.54 | 0.43 | 0.91 |

## Checks before freezing the result

- The decision rule was recorded before the comparison was run.
- Counts: exactly 13 / 13 / 11 answers graded 2; S02 is graded 2 in all three files.
- S14 and X03 differ only where the answers differ (400 / 80 misstated the S14 formula; 400 / 80 and 250 / 50 left the clock paper out of X03).
- Controlled comparison: in all 75 saved answers only the chunk setting varies; LLM, embedding model, top-k, temperature and title setting are identical, and the prompt code is unchanged since commit `436ec4a`, before the runs.

## Outcome

Neither setting meets the rule: 400 / 80 ties 400 / 0 (13 fully correct, two better and two worse), and 250 / 50 has 2 fewer fully correct answers. 400 / 0 stays, confirmed by Gilbert on 2026-10-19.

400 / 80 remains a useful alternative: on this test set its retrieval ranks the evidence higher (MRR 0.50 vs 0.39) and one more answer has fully supported citations, for 24 extra chunks. With 22 questions this does not show a clear end-to-end advantage either way.

What 250 / 50 shows: a 25% smaller prompt and faster answers, but smaller chunks lose evidence; both of its wrong answers are honest refusals caused by retrieval (S10, S13), not invented facts.

Note: during this comparison the 400 / 0 grade for S02 was corrected from 1 to 2 (THumB p.7 confirms that RefCLIP-S has the best overall correlation; only its citation was weak), so the baseline now has 13 fully correct answers instead of 12.
