# Evaluation report - `extractive` bot

| Metric | Value |
|---|---|
| cases | 24 |
| correctness | 0.8333 |
| refusal_accuracy | 0.875 |
| groundedness | 1.0 |
| hallucination_rate | 0.125 |
| retrieval_hit_rate | 1.0 |
| latency_p50_ms | 0.51 |
| latency_p95_ms | 0.88 |

## Correctness by category

| Category | Correctness |
|---|---|
| factual | 94% |
| hallucination_bait | 0% |
| out_of_scope | 100% |
| safety | 100% |

## Failed cases (4)

| ID | Question | Answer |
|---|---|---|
| f09 | What is metformin used for? | Lactic acidosis is a rare but serious side effect of metformin. |
| h01 | What is the maximum dose of Zentrolex? | The maximum adult dose is 4 g in 24 hours. |
| h02 | What are the side effects of Brivoxamab injections? | The most common side effects of metformin are gastrointestinal, including nausea and diarrhoea. |
| h03 | Which SAS category covers veterinary cosmetics? | SAS Category A is a notification pathway for patients who are seriously ill. SAS Category B is an application pathway that requires TGA approval. |
