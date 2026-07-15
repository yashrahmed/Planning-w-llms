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
writes one layout JSON per selected layout in that same directory, using names
such as `living_room-249-train.json`, and a `generation-report.json` with
complete-layout yield and observed source counts. Questions contain the exact
reference `Room layout can be found in file : <layout-file>.json` instead of
embedding the layout JSON. If the seed is omitted, it defaults to `0`.

Pass a third argument of `test` or `val` when generating another split; the
split is included in each layout filename:

```shell
./scripts/generate_questions.sh 20 7 val
```

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

## V1: evaluate with the full raw context

V1 is intentionally a single-response, tool-free baseline. Each record's
complete system and user messages are sent directly to the model. For legacy
records this includes the full embedded `Room layout:` JSON. Newly generated
records instead contain the layout-file reference described above; the current
V1 evaluator does not open that file for the model. Adding explicit file access
is a separate evaluator change. The evaluator does not compact the prompt,
preload a hidden layout runtime, advertise tools, accept tool calls, or run an
agent loop.

### Why the evaluator was reset

The previously merged experiment used the v3 specialist-tool evaluator:

| Aspect | Previous merged state |
|---|---|
| Model input | A compacted question with the raw layout JSON removed |
| Layout setup | The host read `example.source_layout`, loaded that JSON into `FloorplanToolRuntime`, and bound the correct layout before the first model call |
| Tools | `search_entities`, `inspect_entity`, `measure_pair`, `measure_space`, `slide_object`, `test_placement`, and `find_shortest_path` |
| Agent loop | Up to eight model turns sharing a 2,500-generated-token budget |
| Recorded result | 185/200 on the fixed-seed 200-question sample; all 15 failures were visibility questions |

The runtime did not receive the expected answer, but it did receive the hidden
`source_layout` selection. Consequently, the model never had to identify, open,
or parse the floorplan. It only selected geometry operations against a plan the
evaluator had already chosen and loaded. That is a reasonable abstraction for
an application with a visibly active document, but the evaluated interaction
did not provide such an application state to the model or user.

The 185/200 score therefore measured tool routing and argument selection on a
pre-bound floorplan, not end-to-end FloorplanQA behavior. V1 removes that hidden
pre-binding: the complete floorplan is explicit model input, the model receives
no tools, and scorer-only metadata is used only after its single response.

Earlier v4-v6 variants were already absent from the merged state. In particular,
v6's 200/200 result had been rejected because it used hidden task and parameter
metadata and functioned as an oracle rather than an agent evaluation.

Both model backends accept a QA JSONL path, remove reference/assistant messages
from each record, run one example at a time, print a per-example verdict, and
finish with an aggregate score. Scoring follows the paper: 2% relative error
for scalar tasks, 5% for free space, exact Boolean Placement, set-equality
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
