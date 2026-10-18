# Grading rubric

Every answer gets two separate marks plus a response status. The key facts for each question (`key_facts` in `questions.jsonl`) are fixed before any answer is graded.

## Response status

| Status | Meaning |
|---|---|
| answered | The system gave an answer |
| refused | The system replied NOT ENOUGH EVIDENCE |

A refusal is correct for the unanswerable questions (U01 to U03) and a failure for every answerable question.

## Mark 1: correctness (answerable questions)

Judged against the key facts, not against the expected page: a correct answer drawn from another page still counts.

| Mark | Rule |
|---|---|
| 2 | All key facts present, no false claim |
| 1 | At least one key fact present, or all key facts plus a false side claim |
| 0 | No key fact, or the main fact is wrong |
| refused | Counted as a failure |

Unanswerable questions: refused = correct; any substantive answer = 0.

## Mark 2: citation support

Checked by reading the cited source passages, not by matching keywords.

| Mark | Rule |
|---|---|
| yes | Every factual claim is supported by the source it cites |
| partly | Some claims supported; others unsupported, misread or cited to the wrong source |
| no | The cited passages do not support the main claims |
| n/a | Refusal with no factual claims |

## Reporting

Correctness, refusals and citation support are reported separately, never as one combined score. Limits: a single grading pass with a spot check, not independent graders; 22 answerable and 3 unanswerable questions.
