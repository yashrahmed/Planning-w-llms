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
