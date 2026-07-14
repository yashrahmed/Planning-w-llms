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

## Run the iterative Ollama tool experiment

Run the reproducible fixed-seed tool-design loop while preventing macOS sleep:

```shell
caffeinate -dimsu ./scripts/evaluate_ollama_tools.sh
```

The command first ensures that `datasets/train-qa/questions.jsonl` contains 50
examples (seed `1`). It then samples the same 25 questions from the 640-question
seed-`0` evaluation pool on every iteration, limits total generated tokens across
all agent turns to 2,500 per question, and stops after five iterations or as soon
as all 25 questions pass.

Each cumulative tool version is driven by feedback from the preceding result:

1. entity search, entity inspection, and pair measurements;
2. exact free-space/largest-box measurements and object sliding;
3. arbitrary-rotation placement testing and clearance-aware shortest paths;
4. a typed task router, only if failures remain;
5. a current-question geometry solver, only as a final fallback.

Results, tool traces, per-iteration feedback, and the final summary are written
to `experiments/tool-loop/`. The completed seed-`0` experiment stopped at
iteration 3 with scores of 8/25, 18/25, and 25/25.
