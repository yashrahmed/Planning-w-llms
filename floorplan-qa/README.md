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
Fixed blocker-policy prose is not repeated in Repositioning, Max Box, or
Placement questions; the generator and geometry tools retain those semantics.

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

## Explicit-file tool evaluator

The model-facing tool set contains ten focused operations:

| Tool | Description |
|---|---|
| `inspect_room(file_id)` | Reads the exact layout filename referenced by the question and returns a compact natural-language inventory grouped by object type, followed by explicit door and window counts and IDs. It does not return geometry, measurements, walls, or raw JSON. |
| `pair_distance(object_id_1, object_id_2, file_id)` | Returns the Euclidean distance in meters between the polygon centroids of two exact entity IDs, rounded to three decimals. Object, door, and window IDs are accepted. |
| `view_angle(object_id_1, object_id_2, file_id)` | Returns the unsigned angle from north (the positive y-axis) to the vector between two entity centroids, in the range 0 through 180 degrees and rounded to three decimals. |
| `inspect_entity(object_id, file_id)` | Returns one entity's kind, polygon centroid, axis-aligned minimum and maximum coordinates, polygon area, and vertex count. Object, door, and window IDs are accepted. |
| `ray_trace(object_id_1, object_id_2, file_id)` | Traces the finite centroid-to-centroid segment and returns every other intersected entity ID in traversal order. Object, door, and window IDs are accepted. |
| `largest_empty_area(file_id)` | Returns the width, length, and area of the largest rectangle that fits inside the room at any rotation under the benchmark's blocking rules. |
| `occupied_floor_area(file_id)` | Returns the unioned area of occupied object polygons. Rugs count as occupied, while openings and ceiling-only fixtures do not. |
| `calculator(operand_1, operand_2, operator)` | Applies `add`, `sub`, `mul`, or `div` and returns a result rounded to three decimals. Division by zero returns `Error`. |
| `shortest_path(object_id_1, object_id_2, file_id)` | Returns an ordered shortest centroid-to-centroid waypoint path while maintaining the benchmark's fixed 0.15 m clearance from other blocking entities. |
| `test_movement(object_id, direction, file_id)` | Returns the maximum distance an object can translate up, down, left, or right before first contact with a blocking object or the room boundary. |

For example, its output has this form:

```text
Room type: living room.
Room floor area: 25.000 square meters.
3 objects:
- 2 chairs with IDs [chair_1, chair_2].
- 1 table with ID [table].
2 openings:
- 1 door with ID [door].
- 1 window with ID [window].
```

Only `file_id` serves as the model-visible layout locator. The host supplies the
allowed layout directory, and the runtime rejects absolute paths, directory
components, non-JSON filenames, and files outside that directory. The V1
evaluator remains the tool-free baseline described above; the separate tool
evaluator exposes the tools in this table while preserving the same explicit
file boundary.

### Eight-question tool evaluation (2026-07-15)

The first eight records of the fixed seed-0 50-question set form one complete
living-room layout with all eight task types. Ollama `qwen3.5:4b` evaluated them
in file order with temperature 0, thinking disabled, at most eight agent turns,
and a total 2,500-generated-token budget per question. The model received only
input system/user messages, tool schemas, and results from tool calls it chose;
task labels, parameters, and reference answers remained scorer-only.

| Task | Result | Relevant calls |
|---|---:|---|
| Pair distance | Correct | `pair_distance` |
| Free space | Correct | `inspect_room`, `occupied_floor_area`, `calculator` |
| View angle | Correct | `inspect_room`, `view_angle` |
| Repositioning | Incorrect | Exhausted eight turns inspecting entities; no final answer |
| Max Box | Correct | `inspect_room`, `largest_empty_area` |
| Placement | Correct | `largest_empty_area` |
| Shortest path | Incorrect | Exhausted eight turns inspecting entities; no final answer |
| Visibility | Correct | `ray_trace` |

The final score was **6/8 (75%)**, with no runtime errors. The run took 190.343
seconds, generated 4,828 tokens over 33 model calls, and made 27 tool calls.
For Free Space, `inspect_room` returned `24.000 m2`,
`occupied_floor_area` returned `16.540 m2`, and `calculator(sub)` returned the
correct `7.460` answer. The model redundantly made that calculator call twice.
The two failures identify absent capabilities rather than arithmetic errors:
maximum directional translation for Repositioning and clearance-aware routing
for Shortest Path. The full checkpoint is generated under
`datasets/evaluations/` and remains ignored by Git.

#### Rerun after adding movement and path tools

After adding `test_movement` and `shortest_path`, the same first eight records
were evaluated again with the same model, seed, temperature, thinking setting,
turn limit, and 2,500-generated-token budget. Generated question text also no
longer states which categories are nonblocking; that policy remains encoded in
the relevant geometry operation.

| Metric | Initial run | Rerun |
|---|---:|---:|
| Correct | 6/8 (75%) | **8/8 (100%)** |
| Formatting failures | 2 | 0 |
| Runtime errors | 0 | 0 |
| Runtime | 190.343 s | 126.255 s |
| Generated tokens | 4,828 | 3,333 |
| Model calls | 33 | 22 |
| Tool calls | 27 | 14 |

| Task | Rerun result | Relevant calls |
|---|---:|---|
| Pair distance | Correct | `pair_distance` |
| Free space | Correct | `inspect_room`, `occupied_floor_area`, `calculator` |
| View angle | Correct | `view_angle` |
| Repositioning | Correct | `inspect_room`, `test_movement` |
| Max Box | Correct | `inspect_room`, `largest_empty_area` |
| Placement | Correct | `inspect_room`, `largest_empty_area` |
| Shortest path | Correct | `inspect_room`, `shortest_path` |
| Visibility | Correct | `ray_trace` |

The two previously failing tasks now routed directly to their dedicated
geometry operations and produced final answers within the turn limit. In this
rerun, runtime fell by 33.7%, generated tokens by 31.0%, model calls by 33.3%,
and tool calls by 48.1%. These measurements describe this deterministic
eight-question rerun; they are not yet evidence of the same gains over the full
50-question set. The completed report is
`datasets/evaluations/qwen3.5-4b-explicit-file-tools-seed-0-8-v2.json` and is
ignored by Git.

### Replay of the leak-free V3 failures (2026-07-15)

The leak-free V3 report from commit `8fc1200` scored 185/200. Its 15 failures
were all Visibility questions and all were formatting failures: the model
exhausted eight turns without emitting a final answer. It did not submit 15
incorrect intersection lists. The V3 toolset lacked a dedicated finite
centroid-to-centroid segment intersection operation, so the failures broke down
as follows:

| V3 behavior | Questions | Mistake |
|---|---:|---|
| Used only `search_entities` and `inspect_entity` | 9 | Tried to reconstruct polygon intersections manually from centroids and axis-aligned bounds until the turn limit expired |
| Called `measure_pair` with the wrong endpoint pair | 3 | Measured a different segment from the one in the question; one happened to return the same expected set |
| Called `measure_pair` with the correct pair | 2 | The overloaded result already contained the correct intersection list, but the model kept investigating and never finalized |
| Called `find_shortest_path` | 1 | Computed a collision-avoiding route instead of entities intersecting the requested straight segment |

Across those failures, V3 made 120 model calls and 120 tool calls: 93
`inspect_entity`, 21 `search_entities`, five `measure_pair`, and one
`find_shortest_path`. It generated 14,053 tokens over 677.831 seconds. There
were no runtime or tool errors.

The exact 15 layout, endpoint, and reference-answer combinations were replayed
with the current evaluator. Only the prompt representation changed: raw layout
JSON was replaced by its explicit `Room layout can be found in file : ...`
reference, and the model had to pass that visible filename to a tool. The
scorer-only task parameters and answers were not exposed to the agent or tool
runtime.

| Metric | V3 failed subset | Current tools |
|---|---:|---:|
| Correct | 0/15 | **15/15** |
| Formatting failures | 15 | 0 |
| Runtime errors | 0 | 0 |
| Runtime | 677.831 s | 151.077 s |
| Generated tokens | 14,053 | 4,207 |
| Model calls | 120 | 30 |
| Tool calls | 120 | 15 |

Every replayed question used exactly one `ray_trace` call and returned the
correct ordered intersection list. Relative to V3's attempts on these same
questions, runtime fell by 77.7%, generated tokens by 70.1%, model calls by
75.0%, and tool calls by 87.5%. The completed ignored report is
`datasets/evaluations/qwen3.5-4b-v3-failures-current-tools.json`; the ignored
replay input and copied layouts are under `datasets/train-qa/v3-failures/`.

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
