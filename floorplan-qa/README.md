# floorplan-qa

Utilities for downloading and working with the
[FloorplanQA layouts dataset](https://huggingface.co/datasets/OldDelorean/FloorplanQA-Layouts).

## Download the dataset

From this directory, run:

```shell
uv run download-floorplan-qa
```

The files are downloaded to `datasets/FloorplanQA-Layouts/`. This directory is
ignored by Git.

The dataset is public, so authentication is normally unnecessary. If Hugging
Face requests authentication, set the `HF_TOKEN` environment variable or run
`hf auth login` before running the download command.

## Check out the question-generation tooling

To materialize the upstream FloorplanQA code without creating a nested Git
repository, run:

```shell
uv run checkout-fpqa-tooling
```

The command downloads a GitHub source archive into `fpqa-tooling/`. Source
archives do not contain `.git` metadata, and the destination is ignored by the
main repository. Use `--revision <branch-tag-or-commit>` to select something
other than `main`.

## Generate training questions

Generate all eight deterministic QA tasks for each requested layout. Layouts
are drawn by a seeded uniform shuffle over the complete released corpus; the
generator does not force room-source or answer-class balance:

```shell
./scripts/generate_questions.sh 20 1
```

This selects 20 layouts and writes 160 records to
`datasets/train-qa/questions.jsonl`, replacing that file on each run. It also
writes `generation-report.json` with complete-layout yield and observed source
counts. If the seed is omitted, it defaults to `0`.

Each record includes task parameters, a typed reference answer, fixed-template
prompt messages, input-validation results, solver settings, convergence data,
and version provenance. The `paper-v2` implementation uses continuous
first-collision repositioning, configuration-space placement with exact
witnesses or certified negatives, deterministic global Max Box search, and
0.15 m-clearance grid A* for shortest paths. Visibility intentionally uses
actual polygon intersections rather than paper-style bounding boxes.

## Evaluate generation quality

Generate the same sample twice, then compare it while revalidating every
reference answer and geometry witness:

```shell
./scripts/generate_questions.sh 20 7
cp datasets/train-qa/questions.jsonl /tmp/questions-first.jsonl
./scripts/generate_questions.sh 20 7
./scripts/evaluate_generation.sh \
  datasets/train-qa/questions.jsonl \
  /tmp/questions-first.jsonl
```

The hard gates are documented in
[`question-gen.md`](question-gen.md#quality-metrics-and-validation).
The evaluator also reports the unforced room-source and Placement-answer
distributions as advisory measurements.

## Evaluate local Qwen 3.5 4B models

Both evaluators accept a QA JSONL path, remove reference/assistant messages from
each record, run one example at a time, print a per-example verdict, and finish
with an aggregate score. Scoring follows the paper: 2% relative error for
scalar tasks, 5% for free space, exact Boolean Placement, set-equality
Visibility, and collision-free shortest paths within 0.6 m discrete Frechet
distance of the reference.

With Ollama running and `qwen3.5:4b` available:

```shell
./scripts/evaluate_ollama.sh datasets/train-qa/questions.jsonl
```

For a reproducible batch evaluation, generate an 80-layout pool and uniformly
sample 150 questions with a seed (default `0`):

```shell
caffeinate -dimsu ./scripts/evaluate_ollama_random.sh 0
```

The batch uses `qwen3.5:4b` with temperature zero, thinking disabled, and a
768-token output budget per question. It checkpoints a structured JSON report
after every response at `datasets/evaluations/qwen3.5-4b-seed-<seed>.json`, so a
partial run remains inspectable. Pass a second argument to choose another JSON
path. `MODEL`, `SAMPLE_SIZE`, `POOL_LAYOUTS`, and `MAX_TOKENS` environment
variables are available for controlled smoke tests or alternate runs; the
defaults perform the requested 150-question evaluation.

On Apple silicon, run the Hugging Face-hosted MLX-LM 4-bit conversion with:

```shell
./scripts/evaluate_huggingface.sh datasets/train-qa/questions.jsonl
```

Pass `--thinking` to either command to enable Qwen's thinking mode. Use
`--model` or `--max-tokens` after the JSONL path to override the defaults.

## Ollama tool experiment

Generated checkpoints, tool traces, feedback, and summaries are written below
`experiments/`. That directory is intentionally ignored by Git. This README is
the canonical after-action record for committed experiment results.

### Protocol and data boundary

Both tool evaluations used `qwen3.5:4b` through Ollama with thinking disabled,
temperature `0`, seed `0`, and a limit of 2,500 generated tokens per question
across all agent turns. Questions were sampled uniformly without replacement
from the fixed 640-question seed-`0` evaluation pool. The training generator
retained its existing 50 examples selected with seed `1`.

The leak-free evaluator enforces this boundary:

| Component | Available data |
|---|---|
| Model | User-authored question and answer-format instructions, with raw layout JSON removed |
| Tool runtime | Source layout path, layout directory, and evaluation seed |
| Post-response scorer only | Task label, structured parameters, expected answer, and reference answer |

Qwen must infer the requested operation and supply every tool argument from the
visible question. The tool runtime does not hold the full evaluation example.

The later v4 typed router, v5 `solve_current_question` solver, and v6
final-answer tool were removed together with their generated results. In
particular, v6's 200/200 result depended on hidden task and parameter metadata
exposed through `AGENT_DATA_BOUNDARY`; it was an oracle baseline, not a valid
agent-tool evaluation. The leak-free v3 rerun below is the retained result.

### Retained v3 tools

| Tool | Description |
|---|---|
| `search_entities` | Search object, door, and window labels in the current floorplan. |
| `inspect_entity` | Get exact centroid, bounds, polygon area, and kind for one floorplan entity. |
| `measure_pair` | Measure two entity centroids and their relationships: distance, angle from north, and line intersections. |
| `measure_space` | Measure room area, non-occupied floor area, or the largest obstacle-free rectangle. |
| `slide_object` | Measure how far an object can translate before first contact with a blocker or room boundary. |
| `test_placement` | Test whether a rectangle can fit at any rotation in free floor space. |
| `find_shortest_path` | Find a valid shortest waypoint path between entity centroids with obstacle clearance. |

The system prompt is:

> You are a FloorplanQA tool agent. Infer the task and all arguments only from
> the user's question. Call the single most relevant geometry tool immediately;
> never do polygon arithmetic yourself. After a tool returns, use its computed
> values to answer in the user's requested format. You may call another tool
> only when the first tool cannot answer the question.

### Historical 25-question tool evolution

This prototype loop predates the stricter structural data boundary above. It is
retained here to document how the specialist toolset evolved, not as the final
accuracy measurement.

| Version | Change | Correct | Formatting failures | Runtime | Generated tokens |
|---|---|---:|---:|---:|---:|
| v1 | Entity search, inspection, and pair measurements | 8/25 | 15 | 913.6 s | 15,239 |
| v2 | Added space measurement and object sliding | 18/25 | 7 | 405.5 s | 6,036 |
| v3 | Added rotated placement and clearance-aware paths | 25/25 | 0 | 173.5 s | 1,990 |

V1 was perfect on pair-distance, view-angle, and visibility questions but lacked
dedicated tools for most other tasks. V2 fixed free-space, largest-box, and
repositioning questions; its seven remaining failures were placement and
shortest-path questions. V3 added those two missing capabilities and answered
the 25-question prototype sample completely.

### Leak-free 200-question v3 evaluation

The final retained evaluation reran only v3 on 200 fixed-seed questions after
removing task-derived prompt hints and severing the runtime's access to task,
parameter, expected-answer, and reference-answer metadata.

| Task | Correct |
|---|---:|
| Free space | 28/28 |
| Largest box | 26/26 |
| Pair distance | 26/26 |
| Placement | 23/23 |
| Repositioning | 22/22 |
| Shortest path | 25/25 |
| View angle | 21/21 |
| Visibility | 14/29 |
| **Overall** | **185/200 (92.5%)** |

The run generated 46,461 tokens and took 2,341.1 seconds, averaging 11.71
seconds per question. There were no runtime errors or geometry-tool exceptions.

All 15 failures were visibility questions that exhausted the eight-tool-call
loop without emitting a parsable final answer:

| Failure mode | Count | What happened |
|---|---:|---|
| Never called `measure_pair` | 9 | Qwen spent every call on entity search and inspection. |
| Chose the wrong solver | 1 | Qwen called `find_shortest_path` for a line-intersection question. |
| Called `measure_pair` with the wrong entity arguments | 2 | The tool ran correctly for a pair that did not match the requested endpoints. |
| Received the exact correct `measure_pair` result but continued | 3 | Qwen kept searching or inspecting and never formatted the returned visibility answer. |

Across failed questions, Qwen made 120 tool calls: 93 entity inspections, 21
entity searches, five pair measurements, and one shortest-path call. Failures
averaged 936.9 generated tokens and reached a maximum of 1,864. The failure
cluster therefore points to tool discoverability, argument selection, and
stopping behavior rather than the deterministic geometry implementation.

### Reproduce the retained evaluation

Run the 200-question v3 evaluation while preventing macOS sleep:

```shell
caffeinate -dimsu ./scripts/evaluate_ollama_tools.sh \
  --sample-size 200 \
  --seed 0 \
  --max-tokens 2500 \
  --start-iteration 3 \
  --max-iterations 3 \
  --output-dir experiments/tool-loop-200
```

The generated reports remain available locally for inspection but are excluded
from commits by `floorplan-qa/experiments/` in the repository `.gitignore`.

### Validation

- `uv run python -m unittest discover -s tests -v`: 15 tests passed.
- `bash -n scripts/evaluate_ollama_tools.sh`: evaluator launcher syntax passed.
- `git diff --check`: no whitespace errors.
