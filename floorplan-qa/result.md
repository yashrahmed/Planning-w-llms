# Qwen 3.5 4B baseline results

Evaluation date: 2026-07-11

## Setup

Both local Qwen 3.5 4B runners were evaluated on the five generated
pair-distance questions in `datasets/train-qa/questions.jsonl`.

| Backend | Model |
| --- | --- |
| Ollama | `qwen3.5:4b` |
| Hugging Face with MLX-LM | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |

The evaluator retained only system and user messages from each JSONL record, so
the reference assistant response was never passed to the model. Both runs used
greedy generation with thinking disabled and a 4,096-token output limit. Numeric
answers were compared at the three-decimal precision specified by the dataset.

## Scores

| Backend | Correct | Score |
| --- | ---: | ---: |
| Ollama | 4/5 | 80% |
| Hugging Face with MLX-LM | 4/5 | 80% |

## Per-example results

| Example | Expected | Ollama | Hugging Face/MLX | Result |
| --- | ---: | ---: | ---: | --- |
| `pair-distance-bedroom-0` | 0.347 | 0.347 | 0.347 | Both correct |
| `pair-distance-hssd-0` | 3.255 | 3.210 | No final answer | Both incorrect |
| `pair-distance-kitchen-0` | 2.219 | 2.219 | 2.219 | Both correct |
| `pair-distance-living_room-0` | 2.007 | 2.007 | 2.007 | Both correct |
| `pair-distance-bedroom-1` | 2.147 | 2.147 | 2.147 | Both correct |

## Failure analysis

Both models failed on the same HSSD example, which uses irregular polygons, but
they failed differently.

### Ollama

The Ollama model averaged the polygon vertices instead of computing an
area-weighted polygon centroid. It corrected some intermediate summation errors
but retained the incorrect centroid method and returned `3.210` rather than
`3.255`.

### Hugging Face with MLX-LM

The Hugging Face/MLX model initially attempted a shoelace calculation but used
an incorrect centroid formula. It restarted with another incorrect expression,
continued recalculating, and reached the 4,096-token output limit without an
explicit `*Final answer*:` line.

## Performance observation

Ollama appeared slightly faster in these runs, but the comparison was not
controlled: the Hugging Face/MLX model generated substantially more text on the
failed HSSD example. The runners do not yet record prompt-processing time,
generation throughput, or per-example wall-clock duration, so this should not
be treated as a formal performance benchmark.

## Conclusion

Both raw 4B baselines solve the rectangular cases reliably but struggle with
area-weighted centroids for irregular polygons. A deterministic polygon
centroid tool is the clearest first addition for a stronger FloorplanQA
harness.
