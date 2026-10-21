# Phase 8E: final Task Set v2 evaluation

This is one controlled, exploratory run of KubeRoute's fixed-model and
`performance_v1` strategies. It is not a general model-quality benchmark or a
statistically robust performance comparison.

## Method

Four strategies ran sequentially in this order: fixed `qwen3:0.6b`, fixed
`llama3.2:3b`, fixed `gemma3:4b`, then `performance_v1`. Each used Task Set v2's
same 60 distinct questions in the same order, preceded by six separate warm-up
requests. Every category contributed 12 measured questions. The gateway process
used for this evaluation was restarted immediately before the adaptive run to
reset its in-memory routing state. Prometheus counter deltas verified that its
automatic requests used `performance_v1` and matched the completed backend calls.

All strategies requested temperature `0`, seed `42`, `reasoning_effort="none"`,
non-streaming output, a 64-token output limit, and a 180-second request timeout.
Warm-ups are excluded from all measurements below. The output schema is `2`, the
grading version is `exact-plus-choice-v2`, and the Task Set v2 fingerprint is
`5ffc2035d026dd417011d2f983cc860c48169680b6f1bc25db495e0732ecfa37`.
All runs reported usage for all 60 measured requests. The four ignored local
artifact run IDs are `20261021T171614Z-57aaa89d`,
`20261021T171633Z-717f1625`, `20261021T171649Z-ac524125`, and
`20261021T171722Z-1e0b0180`, respectively. Historical and pilot artifacts were
not changed.

## Measured results

All four strategies completed **60/60 measured HTTP requests** with status 200
and finish reason `stop`; failure and timeout rates were 0%. All six warm-ups per
strategy also succeeded. Latencies below are end-to-end gateway request times
in milliseconds; token totals are reported completion tokens for measured
requests only.

| Strategy | Mean / p50 / p95 (ms) | Output tokens (usage coverage) | Exact-format correct | Choice correct / incorrect / ambiguous | Choice identification coverage | Accuracy among identified choices | Selected models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed Qwen3 | 135.01 / 129.14 / 198.25 | 757 (60/60) | 1/60 (1.7%) | 1 / 0 / 59 | 1/60 (1.7%) | 1/1 (100%)* | Qwen3: 60 |
| Fixed Llama 3.2 | 128.08 / 128.88 / 148.56 | 122 (60/60) | 34/60 (56.7%) | 34 / 24 / 2 | 58/60 (96.7%) | 34/58 (58.6%) | Llama: 60 |
| Fixed Gemma 3 | 153.16 / 155.07 / 170.52 | 177 (60/60) | 41/60 (68.3%) | 41 / 19 / 0 | 60/60 (100%) | 41/60 (68.3%) | Gemma: 60 |
| `performance_v1` | 133.89 / 125.45 / 226.02 | 490 (60/60) | 19/60 (31.7%) | 19 / 11 / 30 | 30/60 (50.0%) | 19/30 (63.3%) | Qwen3: 29; Llama: 28; Gemma: 3 |

\* Qwen3's 100% identified-choice accuracy is based on **one** identifiable
answer. Its other 59 responses were ambiguous to the conservative classifier;
this number cannot support a quality claim. The evaluator does not save response
text, so this run cannot distinguish semantically correct but unsupported formats
from other ambiguous answers. For these runs, the exact-format correct count
happened to equal the conservative choice-correct count; the two grading methods
remain distinct.

The table below gives conservative choice outcomes per category as
**correct / incorrect / ambiguous**, each out of 12. Exact-format correct counts
were identical to the choice-correct counts in this run.

| Category | Fixed Qwen3 | Fixed Llama 3.2 | Fixed Gemma 3 | `performance_v1` |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | 0 / 0 / 12 | 4 / 8 / 0 | 5 / 7 / 0 | 2 / 3 / 7 |
| Numerical comparison | 1 / 0 / 11 | 9 / 3 / 0 | 9 / 3 / 0 | 6 / 0 / 6 |
| Logical reasoning | 0 / 0 / 12 | 9 / 3 / 0 | 12 / 0 / 0 | 4 / 2 / 6 |
| String operations | 0 / 0 / 12 | 4 / 7 / 1 | 3 / 9 / 0 | 2 / 5 / 5 |
| Reading comprehension | 0 / 0 / 12 | 8 / 3 / 1 | 12 / 0 / 0 | 5 / 1 / 6 |

Adaptive model selections also varied by category:

| Category | Qwen3 | Llama 3.2 | Gemma 3 |
| --- | ---: | ---: | ---: |
| Arithmetic | 7 | 3 | 2 |
| Numerical comparison | 7 | 5 | 0 |
| Logical reasoning | 6 | 5 | 1 |
| String operations | 4 | 8 | 0 |
| Reading comprehension | 5 | 7 | 0 |

## Interpretation and limits

The 757 versus 122 and 177 reported output tokens for the fixed models show
that response lengths differed substantially. End-to-end latency includes the
gateway path and varying generation work, so these times do not isolate model
speed or routing overhead. Adaptive latency additionally reflects the mix of
models chosen, exploration, and task order. Its observed accuracy depends on
which model handled each question; it is not the accuracy of a single model.

Qwen3's 1.7% identification coverage prevents reliable interpretation of its
choice-identification accuracy or a direct quality comparison with the other
models. Strict-format scoring records compliance with a narrow output format,
not general semantic correctness. These 60 short synthetic multiple-choice
tasks do not represent broad reasoning quality or real user workloads.

This was one run in a fixed strategy order, without repeated or rotated-order
trials. Model loading, GPU contention, task wording, and output length can
affect the observed latencies. Even though the runs had no request failures,
their latency differences and category scores do not establish statistically
robust model superiority. The result is sufficient to close Phase 8E as a
working, reproducible evaluation path and to proceed to RAG work, while leaving
model-quality and performance-ranking claims open.
