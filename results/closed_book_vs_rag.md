# Closed-book vs RAG (baseline, 2026-10-18)

Same model (`qwen3.5:4b`), temperature 0. Closed-book gets only the question; RAG gets the five retrieved chunks. Grades in `grades_closed_book.csv` and `grades_baseline.csv`, rubric in `eval/rubric.md`.

## Correctness on the 22 answerable questions

| | 2 | 1 | 0 |
|---|---|---|---|
| Closed-book | 0 | 6 | 16 |
| RAG | 13 | 9 | 0 |

## Four boxes ("correct" = 2)

| | RAG correct | RAG not correct |
|---|---|---|
| Closed-book correct | 0 (model already knew) | 0 (retrieval hurt) |
| Closed-book not correct | 13 (retrieval helped) | 9 (both not fully correct) |

RAG counts include the S02 regrade (1 to 2) made during step 5.5.

## Unanswerable questions (U01 to U03)

| | Refused or said it does not know | Invented an answer |
|---|---|---|
| Closed-book | 0 | 3 (a V100 GPU and 24 hours; BART / T5 claims; a CIDEr score of 41.1) |
| RAG | 3 | 0 |

Closed-book never used its "I DON'T KNOW" reply on any of the 25 questions.

## Invented claims checked against outside sources (2026-10-18)

| Closed-book claim | What outside sources show |
|---|---|
| CHAIR was introduced by "Wang et al. (2019)" | CHAIR is from Rohrbach, Hendricks, Burns, Darrell and Saenko, EMNLP 2018 |
| "Gebru et al. (2018) annotated 500 COCO images" | No such study found; the COCO annotation study is Zhao, Wang and Russakovsky (ICCV 2021), 28,315 people |
| The 2018 survey is "by Goel et al." | It is by Hossain, Sohel, Shiratuddin and Laga |
| Indonesian datasets "IIC" and "IndoCaption" | Neither found; FEEH-ID (translated Flickr8k) exists |
| ReFormer used "a single V100, about 24 hours" | Not stated in the paper; no official code repository found |
| The CLIP paper reports "CIDEr of about 41.1" on MS-COCO | The CLIP paper has no captioning results and no CIDEr; 41.1 appears only as a zero-shot EuroSAT accuracy |
| CapArena has "over 200" battles | Over 6,000 (paper and ACL Anthology listing) |
| CLIP-Adapter, OpenCLIP, ImageBind use CLIP | True, but none is in the collection |
