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

Generate a requested number of deterministic QA examples from all eight
FloorplanQA task families. Task types are balanced in shuffled blocks, and the
seed controls layout order, task order, and task parameters:

```shell
./scripts/generate_questions.sh 20 1
```

The command writes `datasets/train-qa/questions.jsonl`, replacing that file on
each run. If the seed is omitted, it defaults to `0`. Each record includes its
task parameters, typed reference answer, prompt messages, and solver provenance.
The first implementation follows the paper's task semantics while marking
approximate solvers as `experimental-v1`.

## Evaluate local Qwen 3.5 4B models

Both evaluators accept a QA JSONL path, remove reference/assistant messages from
each record, run one example at a time, print a per-example verdict, and finish
with an aggregate score.

With Ollama running and `qwen3.5:4b` available:

```shell
./scripts/evaluate_ollama.sh datasets/train-qa/questions.jsonl
```

On Apple silicon, run the Hugging Face-hosted MLX-LM 4-bit conversion with:

```shell
./scripts/evaluate_huggingface.sh datasets/train-qa/questions.jsonl
```

Pass `--thinking` to either command to enable Qwen's thinking mode. Use
`--model` or `--max-tokens` after the JSONL path to override the defaults.
