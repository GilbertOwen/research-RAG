# Failure log (baseline run, 2026-10-18)

Every answer below full marks, plus citation problems on fully correct answers. Sources: `grades_baseline.csv`, the saved runs in `examples/eval_baseline/`, and `retrieval_final_checked_keys.json`. Settings: 400 words, overlap 0, no title prefix, top-5, `qwen3.5:4b`, temperature 0.

## Failure types

| Type | Meaning |
|---|---|
| Retrieval | The evidence needed for a key fact was not among the five sources |
| Generation | The evidence was there, but the answer misread it, added a false claim or left a key fact out |
| Attribution | The claim is right, but its citation is missing, wrong or does not support it |
| Ingestion | The PDF text was broken, so the evidence could not be used |

## Answers graded 1 (9 of 22)

| ID | Main type | What happened | Evidence in the five sources |
|---|---|---|---|
| S13 | Retrieval | Named only FEEH-ID; the table listing all three Indonesian datasets (p.12) was never retrieved | 0 of 1 expected pages |
| X01 | Retrieval (and generation) | THumB and SPECS points missing because neither paper was retrieved; also says CIDEr correlates poorly, while its own source says CIDEr correlates better than BLEU | 0 of 3 expected pages |
| X02 | Retrieval | Hyper-detailed captioning and clock-on-the-beach causes missing (papers not retrieved); HallE-Control's cause paraphrased loosely | 2 of 5 expected pages |
| M01 | Retrieval | Focus! and F-CLIPScore missing (not retrieved); one citation points to the wrong source | 1 of 3 expected pages |
| M02 | Retrieval (and generation) | HallE-Control and hyper-detailed captioning missing (not retrieved); counts papers that only test or mention GPT-4 (uses-vs-mentions trap); labels written as **S1** without brackets | 0 of 3 expected pages |
| S05 | Generation | Attributes better handling of low-frequency object pairs to sentence simplification; the source says that about the augmentation, and that simplification had little effect | 1 of 1 |
| S15 | Generation | WER missing although a retrieved chunk discusses WER; adds CIDEr, which the paper only lists as a repository script | 1 of 1 |
| X04 | Generation | Noun-level granularity missing although two retrieved chunks describe F-CLIPScore's noun scores | 0 of 3 expected pages, but the facts were in other retrieved pages |
| X05 | Generation | Describes the metric paper as societal bias in CLIPScore without saying gender or PAO-EvalBias; the retrieved abstract says gender | 1 of 2 expected pages |

Count by main type: retrieval 5, generation 4, attribution 0, ingestion 0.

## Fully correct answers with citation problems (attribution)

| ID | What happened |
|---|---|
| S01 | Third sentence cites a results table that does not say how the total score is computed |
| S02 | "Specifically RefCLIP-S" is correct (THumB p.7) but cites the abstract, which does not mention it. Regraded from 1 to 2 during step 5.5; first listed here as a generation failure |
| S07 | First sentence (28,315 people) has no citation |
| S10 | Middle sentence has no citation (true, but uncited) |

## Not system failures

| ID | What happened |
|---|---|
| S04, S08 | Search missed the expected page, but the answer is fully correct from another page: the retrieval metric is stricter than the task |
| S10, X03 | Citation checker false positives: it splits a sentence at "Focus!" and flags a sentence that starts with its label |

## Other settings (step 5.5)

The same failure types appear with 400 / 80 and 250 / 50 (`compare_settings.md`). With 250 / 50, two answers became honest refusals because the needed evidence was not retrieved (S10, S13), and 400 / 80 misstated the F-CLIPScore formula (S14) although its source had it right.

## Patterns

1. Retrieval failures are all on cross-paper and uses-vs-mentions questions (X01, X02, M01, M02) plus one table page (S13): one or two papers fill the five slots.
2. Generation failures are mostly a correct answer with one false or missing detail, not a wrong main answer.
3. No answer was graded 0, and no failure traced back to broken PDF text.
4. The citation checker catches missing labels, but none of the generation errors above; those needed reading the source.
