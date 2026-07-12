# Question-generation quality gates

## Acceptance metrics

The generator is accepted when a representative generated sample meets every
hard threshold below. Distribution measurements are reported but not optimized
or forced.

| Metric | Threshold | Meaning |
| --- | ---: | --- |
| Complete layout yield | at least 95% | Considered layouts that emit a complete eight-task bundle |
| Layout completeness | 100% | Exactly one record for each of the eight tasks per emitted layout |
| Schema validity | 100% | Required typed fields, messages, and provenance are present |
| Prompt validity | 100% | Fixed template, serialized layout, and strict final-answer contract agree |
| Reference reproducibility | 100% | Regeneration from recorded seed and parameters reproduces the reference |
| Geometry-witness validity | 100% | Repositioning, Max Box, Placement, and path witnesses pass exact geometry checks |
| Placement certification | 100% | `True` has a valid pose; `False` has a geometric impossibility certificate |
| Shortest-path validity | 100% | Endpoints and every segment satisfy the declared 0.15 m-clearance navigation geometry |
| Max Box convergence | 100% | Search reports convergence with at most 2% relative improvement over its final window |
| Deterministic file match | 100% | Same seed and inputs produce byte-identical JSONL |

The observed room-source distribution is compared with the released corpus
(30% bedroom, 10% HSSD, 30% kitchen, and 30% living room). Placement `True`
rate is also reported. Both are advisory because layouts use a uniform shuffle
and Placement uses a uniform catalog order with rejection only when the solver
cannot produce a witnessed or certified answer; neither is selected to hit a
target balance.

## Evaluation result

Attempt 4 evaluated 800 records from 100 uniformly sampled layouts using seed
`20260712`. The same sample was generated twice and the evaluator independently
regenerated all records.

All hard metrics reached 100%, except complete layout yield, which was 99.0%
(100 emitted layouts from 101 considered layouts) and exceeded its 95% threshold.
Max Box convergence used a 2% relative-improvement threshold.

Observed, unforced distributions:

| Distribution | Result |
| --- | ---: |
| Bedroom layouts | 27% |
| HSSD layouts | 9% |
| Kitchen layouts | 33% |
| Living-room layouts | 31% |
| Placement `True` answers | 97% |

Attempt 3 initially reported one invalid repositioning witness in HSSD room
190. Its final pose crossed the room boundary by only 5.07e-8 m2, below the
solver's declared 1e-7 m2 geometry tolerance. Attempt 4 made the witness check
use that same declared tolerance; no question or reference answer was changed.
