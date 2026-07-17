# floorplan-qa

Utilities for downloading and working with the
[FloorplanQA layouts dataset](https://huggingface.co/datasets/OldDelorean/FloorplanQA-Layouts).

## Install

Placement and repositioning use a native CGAL configuration-space extension.
On macOS, install CGAL and its native dependencies before syncing the Python
environment:

```shell
brew install cgal
uv sync
```

On Linux, install CGAL 6.x, a C++17 compiler, CMake, Boost headers, and an exact
number backend such as GMP/MPFR through the system package manager, then run
`uv sync`. The Python environment and lockfile remain managed by uv;
`scikit-build-core` and `pybind11` compile the extension during `uv sync`.

CGAL's 2D Minkowski Sums package is GPL-licensed. Review GPL compatibility
before distributing this extension or its compiled wheels; CGAL's commercial
license is the alternative for incompatible distribution terms.

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
are drawn by a seeded uniform shuffle over the complete released corpus. The
layout count must be even: every Boolean task is deterministically stratified
to contain exactly 50% `True` and 50% `False` reference answers. Placement is
currently the only Boolean task; layouts that cannot produce a witnessed
positive or certified negative for their assigned class are skipped:

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

For a negative Placement target, the generator first tries the fixed catalog.
If none of its rectangles has an area-based impossibility certificate, it
deterministically scales the largest catalog rectangle until its area exceeds
the largest connected free-space component. The displayed dimensions are the
same three-decimal values used by the solver, so the negative remains
independently checkable from the emitted question.

Each record includes task parameters, a typed reference answer, fixed-template
prompt messages, input-validation results, solver settings, convergence data,
and version provenance. The `paper-v6-cgal-configuration-space`
implementation uses CGAL Minkowski configuration spaces for placement and
first-contact repositioning, exact witnesses or area-certified Placement
negatives, contact-event plus deterministic SHGO Max Box search, and 0.15
m-clearance grid A* for shortest paths. Visibility intentionally uses actual
polygon intersections rather than paper-style bounding boxes.

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
The evaluator also reports the unforced room-source distribution and verifies
that regenerated records retain their assigned Boolean-answer targets.

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

The model-facing tool set contains eleven focused operations:

| Tool | Description |
|---|---|
| `inspect_room(file_id)` | Reads the exact layout filename referenced by the question and returns a compact natural-language inventory grouped by object type, followed by explicit door and window counts and IDs. It does not return geometry, measurements, walls, or raw JSON. |
| `pair_distance(object_id_1, object_id_2, file_id)` | Returns the Euclidean distance in meters between the polygon centroids of two exact entity IDs, rounded to three decimals. Object, door, and window IDs are accepted. |
| `view_angle(object_id_1, object_id_2, file_id)` | Returns the unsigned angle from north (the positive y-axis) to the vector between two entity centroids, in the range 0 through 180 degrees and rounded to three decimals. |
| `inspect_entity(object_id, file_id)` | Returns one entity's kind, polygon centroid, axis-aligned minimum and maximum coordinates, polygon area, and vertex count. Object, door, and window IDs are accepted. |
| `ray_trace(object_id_1, object_id_2, file_id)` | Traces the finite centroid-to-centroid segment and returns every other intersected entity ID in traversal order. Object, door, and window IDs are accepted. |
| `largest_empty_area(file_id)` | Returns the width, length, and area of one maximum-area rectangle that fits inside the room at any rotation. Its side lengths are not global limits for other aspect ratios. |
| `find_space_with_size(width, length, file_id)` | Returns whether the requested rectangle fits anywhere in the room at any evaluated rotation and includes a valid center and rotation when it does. |
| `occupied_floor_area(file_id)` | Returns the unioned area of occupied object polygons and its percentage of total room area; it correctly excluded. |
| `calculator(operand_1, operand_2, operator)` | Applies `add`, `sub`, `mul`, or `div` and returns a result rounded to three decimals. Division by zero returns `Error`. |
| `shortest_path(object_id_1, object_id_2, file_id)` | Returns an ordered shortest centroid-to-centroid waypoint path while maintaining the benchmark's fixed 0.15 m clearance from other blocking entities. |
| `test_movement(object_id, direction, file_id)` | Returns the maximum distance an object can translate up, down, left, or right before first contact with a blocking object or the room boundary. |

The evaluator prepends this system prompt:

```text
You are a FloorplanQA agent. Solve each question using the available floorplan
tools whenever they provide relevant geometric evidence. The room layout is not
embedded in the prompt: pass the exact file identifier written in the question
to any tool that requires it. Choose tool names and arguments only from the
user-visible question and tool schemas. Do not invent tool results. When a tool
directly returns the quantity requested by the question, and the tool does not
report an error, your next response must be the final answer. Do not call
another tool to verify, recompute, reformat, or inspect that result. Never
repeat a tool call with the same arguments. In particular, do not use the
calculator on coordinates, paths, entity lists, distances, angles, or Boolean
values already returned by another tool. End with the exact final-answer format
requested by the user.
```

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
file boundary. Layout parsing and geometry solvers live in the shared
`floorplan_qa.geometry` module, which is used by the generator, quality checks,
scorer, and tool runtime; the tool runtime does not import the question
generator.

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

### Three-hundred-question tool evaluation (2026-07-15)

All 300 generated examples were evaluated in file order with the explicit-file
tool runtime and Ollama `qwen3.5:4b`. The run used seed 0, temperature 0,
thinking disabled, at most eight agent turns, five Ollama retries, and a total
2,500-generated-token budget per question. The model saw the question's layout
filename, tool schemas, and tool results it requested; task labels, parameters,
reference answers, and source-layout provenance remained scorer-only.

| Task | Correct | Total | Formatting failures |
|---|---:|---:|---:|
| Pair Distance | 38 | 38 | 0 |
| Free Space | 38 | 38 | 0 |
| View Angle | 37 | 38 | 1 |
| Repositioning | 38 | 38 | 0 |
| Max Box | 37 | 37 | 0 |
| Placement | 29 | 37 | 3 |
| Shortest Path | 37 | 37 | 0 |
| Visibility | 37 | 37 | 0 |
| **Overall** | **291** | **300** | **4** |

The final score was **291/300 (97.0%)**. The run completed in 5,281.518
seconds (88 minutes 1.5 seconds), generated 137,035 tokens over 872 model
calls, and made 569 tool calls. There were no runtime or tool errors. The full
checkpoint is
`datasets/evaluations/qwen3.5-4b-explicit-file-tools-seed-0-300.json` and
remains ignored by Git.

#### Failure analysis

| Failure class | Count | Question IDs | What happened |
|---|---:|---|---|
| Placement false negative | 5 | `placement-bedroom-123`, `placement-bedroom-286`, `placement-bedroom-245`, `placement-bedroom-85`, `placement-bedroom-138` | Each reference answer was `True`, but the model compared the candidate's sides with the single rectangle returned by `largest_empty_area` and answered `False`. That rectangle maximizes area; its side lengths are not independent global bounds on every other feasible aspect ratio. A longer, narrower candidate can fit elsewhere even when it does not fit inside the maximum-area witness. |
| Placement without a final answer | 3 | `placement-bedroom-8`, `placement-bedroom-475`, `placement-living_room-229` | The model repeatedly reconsidered how to interpret the maximum-area witness and consumed all 2,500 generated tokens. In the living-room case, the returned 2.840 by 1.500 meter witness directly dominated the requested 2.500 by 1.000 meter candidate, but the model still failed to finalize. |
| View Angle without a final answer | 1 | `view-angle-bedroom-189` | `view_angle` directly returned the correct 19.148-degree answer on the second tool call. The model then inspected both entities and made four redundant calculator calls before exhausting all eight turns without returning an answer. |

The Placement failures are a capability mismatch, not an arithmetic or
scoring defect. `largest_empty_area` answers Max Box by returning one
maximum-area feasible rectangle. It is a sufficient Placement witness when a
candidate fits inside that returned rectangle, but it cannot prove that a
different aspect ratio does not fit. The model treated it as a complete
feasibility oracle in all five false negatives and became indecisive around the
same ambiguity in all three Placement formatting failures.

Recommended changes, in priority order:

1. Add a candidate-aware `find_space_with_size(width, length, file_id)` geometry tool
   that tests the requested rectangle over the benchmark's allowed positions
   and rotations and returns `True` or `False`, optionally with a witness pose.
   This directly represents the Placement task and removes the invalid need to
   infer feasibility from a maximum-area rectangle.
2. Clarify that `largest_empty_area` returns one rectangle selected by maximum
   area and that its individual side lengths are not global clearance limits.
   This preserves its Max Box role while reducing misuse on other aspect
   ratios.
3. Add a generic agent instruction to finalize from a tool result when it
   directly supplies the requested quantity, unless the tool reports an error.
   This targets the View Angle failure and avoids redundant entity inspection
   and arithmetic without exposing task labels or reference data.
4. Preserve a small final-response reserve or force a final-answer turn when
   the generated-token or turn budget is nearly exhausted. This is a fallback
   for formatting robustness; it does not fix the missing Placement
   capability by itself.

#### Deterministic replay after the targeted fixes

The exact nine failures were replayed before changing the toolset. A repeatable
`--question-id` evaluator option selected those records directly from the same
300-question JSONL while retaining file order. This selector does not alter the
prompt, tools, model input, or scoring. With the original prompt and ten tools,
the replay reproduced every original failure: each response, parsed answer,
generated-token count, model-call count, stop reason, and ordered tool-name
sequence matched its result in the full run. Runtime differed only in wall-clock
duration.

The implementation then added `find_space_with_size`, which applies the same
configuration-space placement solver used by question generation to the width,
length, and visible layout filename supplied by the model. A successful result
includes a valid center and rotation; every result states `True` or `False` in
natural language. The placement solver and its shared free-space helpers now
live in `floorplan_qa.geometry`, so generation and tool execution use the same
geometry implementation. The generic agent prompt also now says to finalize
from a tool result that directly supplies the requested quantity instead of
independently recomputing it, unless the tool reports an error.

The same nine records were then replayed with identical model and evaluation
settings: Ollama `qwen3.5:4b`, seed 0, temperature 0, thinking disabled, eight
turns, five retries, and 2,500 generated tokens per question.

| Metric | Before | After |
|---|---:|---:|
| Correct | 0/9 (0%) | **9/9 (100%)** |
| Placement | 0/8 | **8/8** |
| View Angle | 0/1 | **1/1** |
| Formatting failures | 4 | 0 |
| Runtime | 340.867 s | 95.501 s |
| Generated tokens | 13,137 | 2,481 |
| Model calls | 32 | 21 |
| Tool calls | 24 | 12 |

All eight Placement questions called `find_space_with_size` and returned
`True`; six used it as their only tool, while two first called `inspect_room`.
The View Angle question called `inspect_room` and `view_angle`, accepted the
direct 19.148-degree result, and finalized without entity inspection or
calculator calls. Relative to the deterministic baseline, runtime fell by
72.0%, generated tokens by 81.1%, model calls by 34.4%, and tool calls by 50.0%.
The completed before and after reports are
`datasets/evaluations/qwen3.5-4b-nine-failures-before-find-space.json` and
`datasets/evaluations/qwen3.5-4b-nine-failures-after-find-space.json`; both
remain ignored by Git.

#### Ten additional random Placement questions

Ten other Placement records were sampled without replacement from the remaining
29 Placement questions using Python's seeded `random.Random(0)`. The eight
Placement failures used in the targeted replay were excluded. Evaluation kept
the same model, seed, temperature, thinking setting, turn limit, retry count,
and 2,500-token budget.

The model answered **10/10 correctly** in 102.720 seconds, generated 2,525
tokens over 23 model calls, and made 13 tool calls. Every question called
`find_space_with_size`; seven used it as their only tool and three called
`inspect_room` first. No question called `largest_empty_area`, and there were
no formatting failures, runtime errors, entity-inspection calls, or calculator
calls.

This result has an important limitation. All 37 Placement records in the
current 300-question dataset have reference answer `True`; the generator
produced no negative Placement examples. Moreover, the rounded dimensions of
the maximum-area witness were sufficient to contain the requested rectangle in
all ten questions in this random sample. Consequently, this 10/10 score proves
that the model consistently routes Placement questions to the new
candidate-aware tool, but it does not independently prove that the new geometry
operation was necessary for these ten answers. The earlier eight Placement
failures remain the relevant aspect-ratio counterexamples where inference from
the maximum-area witness failed.

A stronger follow-up dataset should deliberately include both `True` and
`False` Placement questions and should stratify positive cases between those
that are and are not dimensionally dominated by the maximum-area witness. The
completed ignored report is
`datasets/evaluations/qwen3.5-4b-random-other-placement-seed-0-10.json`.

#### Balanced Boolean Placement agent check

The balanced generator follow-up used 20 layouts selected with generation seed
7 and emitted 20 Placement questions: exactly 10 `True` and 10 `False`. The
explicit-file tool agent used Ollama `qwen3.5:4b`, agent seed 0, temperature 0,
thinking disabled, eight turns, and a 2,500-generated-token budget per
question.

The first agent run scored **19/20**. It correctly answered all 10 negative
questions and 9 of 10 positive questions, with no formatting or runtime
failures. The miss was not a model-routing error: for
`placement-kitchen-253`, the model called `find_space_with_size(2.4, 1.2,
kitchen-253-test.json)`, received `False`, and faithfully returned `False`,
while the generator had recorded `True`.

The mismatch came from the two callers seeding sampled placement-center search
differently. The generator used its global generation seed, while
`find_space_with_size` used a query-specific seed. Both called the same geometry
function, but a narrow feasible region could therefore be sampled by one caller
and missed by the other. The temporary shared-seed repair reproduced the
generator's answer, but it coupled a real tool to hidden dataset provenance and
was therefore unsuitable as the final design.

After regenerating the same seeded 20-layout set, the corrected run scored
**20/20**: 10/10 `True` and 10/10 `False`, with zero formatting failures and
zero runtime errors. It completed in 195.597 seconds, generated 4,541 tokens
over 45 model calls, and made 25 tool calls. Every question called
`find_space_with_size`; five first called `inspect_room`. The raw before and
after reports remain ignored at
`datasets/evaluations/qwen3.5-4b-balanced-placement-seed-7-before-shared-seed.json`
and
`datasets/evaluations/qwen3.5-4b-balanced-placement-seed-7-after-shared-seed.json`.

#### Seed-independent geometry follow-up

The shared placement seed has now been removed. Geometry answers depend only
on the layout polygons and the visible query dimensions. Both question
generation and `find_space_with_size` enumerate the same deterministic
configuration-space candidates: one representative per connected component,
representatives from a polygon triangulation, and a fixed Halton
low-discrepancy sequence. Every `True` answer includes a placement center and
rotation that passes an independent full-rectangle containment check. The
former `kitchen-253` query for a 2.4 by 1.2 meter rectangle now consistently
returns a valid witness centered at `(3.1875, 2.48)` and rotated 90 degrees,
without reading a seed, source group, split, or layout ID.

A failed finite search is not accepted as proof of `False`. The tool returns
`False` only when the query rectangle's area exceeds the area of every
connected free-space component. Otherwise it reports that the result is
inconclusive. The balanced generator follows the same rule: positives require
a checked witness and negatives require the area certificate.

Max Box no longer consumes a generation seed either. Its initial centers come
from the same deterministic geometry candidates, and its population mutation
and refinement schedule is fixed. On `kitchen-253`, generation seeds 0, 7, and
99 all produced the identical 4.6513739646 square-meter witness. This removes
the generator/tool seed-alignment problem, but the result remains a valid
numerical lower bound rather than a proof of the global maximum.

The Max Box follow-up implements the paper's exact Type-A contact event: every
pair of polygon-boundary vertices is tested as the opposite corners of a
square, and every resulting square is independently checked against the full
free-space polygon, including holes. For the remaining Type B-F continuous
contact configurations, the solver uses SciPy's deterministic simplicial
homology global optimizer (SHGO) over center, width, length, and rotation, with
rectangle-outside-free-space area as a nonlinear constraint. Every optimizer
result is shrunk if necessary and then independently checked by Shapely before
it can become the answer. The previous deterministic candidate-refinement
solver remains a valid lower-bound fallback and a feasible starting point for
local contact refinement.

No available library implements the complete arbitrary-orientation six-type
event map from
[Maximum-Area Rectangles in a Simple Polygon](https://arxiv.org/abs/1910.08686).
CGAL supports convex inscribed k-gons and axis-aligned empty rectangles;
`largestinteriorrectangle` is an axis-aligned binary-grid routine; and PyAEDT's
arbitrary-orientation implementation uses a finite quasi-lattice. Consequently,
the current hybrid explicitly marks `global_optimum_certified: false`: it
implements exact Type A events and deterministic continuous optimization for
Types B-F, but not the paper's full staircase/ray-shooting event data
structures.

The exact target for fixed-aspect placement is the largest-similar-polygon
algorithm described in
[Largest similar copies of convex polygons amidst polygonal obstacles](https://arxiv.org/abs/2012.06978).

#### CGAL configuration-space before/after

Placement and repositioning now share a native CGAL primitive. The extension
forms the complement of free space inside a safely expanded domain, computes
its Minkowski sum with the reflected query footprint, and subtracts that
forbidden reference-point region from the room domain. Placement searches the
resulting valid-center regions. Repositioning intersects a directional ray
with the valid region containing the moving object's current reference point
and returns the first-contact distance.

The before and after comparison replayed the same 75 applicable records from
the current 300-question JSONL: 37 Placement and 38 Repositioning records. It
called the real `find_space_with_size` and `test_movement` tools with the
recorded visible arguments; it did not run a language model.

| Metric | Shapely custom solver | CGAL configuration space |
|---|---:|---:|
| Placement answer matches | 37/37 | 37/37 |
| Repositioning rounded-distance matches | 38/38 | 38/38 |
| Placement inconclusive results | 0 | 0 |
| Snapshot runtime | 0.508 s | 0.437 s |

All 38 Repositioning output strings remained byte-identical. Twenty-five
Placement strings changed because CGAL selected a different valid center or
rotation; every Boolean answer remained `True` and every returned pose passed
the independent full-rectangle check. The existing 300-question file contains
no negative Placement records, so regression tests additionally cover an
impossible rectangle and a case where all four corners are clear but the
rectangle body crosses an obstacle.

The first CGAL replay exposed an exact-contact edge case: eleven objects began
in zero-width configuration-space corridors while touching cabinets on both
sides, and CGAL's regularized polygon set discarded those lower-dimensional
corridors. The final implementation insets the moving/query footprint by the
declared 1e-7-meter geometry tolerance before forming the Minkowski sum. It
also backs a reported movement contact off by 1e-5 meters, matching the former
solver's lower-bound convention and keeping the independently translated final
pose valid. This restored all 38/38 Repositioning matches, including movement
away from an initial boundary contact.

The ignored detailed reports are
`datasets/evaluations/cgal-geometry-before.json` and
`datasets/evaluations/cgal-geometry-after.json`. The reproducible comparison
driver is `scripts/compare_geometry_backends.py`. A clean `uv sync --reinstall`
successfully rebuilt the native extension, all 65 unit tests passed, and
`uv build --wheel` produced a wheel containing the compiled CGAL module. The
local macOS wheel links Homebrew GMP/MPFR and therefore still requires the
documented native runtime libraries; it is not a self-contained portable wheel.

#### CGAL 50-question tool-agent evaluation

After commit `253cc8a`, an eight-layout seed-0 pool was regenerated and reduced
to a deterministic, task-stratified 50-record set in source order. Pair
Distance and Free Space contain seven records each; the other six task types
contain six records each. Placement is exactly balanced at three `True` and
three `False` answers. All 50 records parsed successfully, regenerated
deterministically, referenced available layout files, and passed their
task-specific geometry-witness checks before model evaluation.

Ollama `qwen3.5:4b` evaluated the set with the explicit-file tools, agent seed
0, temperature 0, thinking disabled, a shared 2,500-generated-token budget per
question, at most eight turns, and five transient-connection retries. The run
completed under `caffeinate` and its wrapper exited normally.

| Task | Correct | Total |
|---|---:|---:|
| Pair Distance | 7 | 7 |
| Free Space | 5 | 7 |
| View Angle | 6 | 6 |
| Repositioning | 6 | 6 |
| Max Box | 6 | 6 |
| Placement | 6 | 6 |
| Shortest Path | 6 | 6 |
| Visibility | 6 | 6 |
| **Overall** | **48** | **50** |

The 96% run took 791.777 seconds, generated 17,475 tokens over 139 model calls,
and made 88 tool calls. It had no formatting failures or runtime errors. Both
CGAL-backed tasks scored 100%: Placement was 6/6, including all three negative
cases, and Repositioning was 6/6.

Both failures were Free Space arithmetic errors rather than geometry or tool
errors. For `free-space-bedroom-135`, `inspect_room` returned 18.240 square
meters and `occupied_floor_area` returned 9.710, but the model called 9.710 the
non-occupied area instead of subtracting to obtain 8.530. For
`free-space-bedroom-245`, it similarly returned the occupied 8.045 rather than
subtracting it from 21.750 to obtain 13.705. In both cases the model had all
required values, omitted the calculator call, and confidently misread
"occupied" as "non-occupied." The other five Free Space questions performed
the subtraction correctly.

The complete ignored report is
`datasets/evaluations/qwen3.5-4b-cgal-seed-0-50.json`.

#### CGAL 300-question tool-agent evaluation (2026-07-16)

A fresh 38-layout, seed-0 pool produced 304 records. The first 300 evaluated
records were deterministically arranged to retain all 38 Placement questions,
including the generator's exact 19 `True` / 19 `False` split. The selected set
contained 37 Pair Distance, Free Space, View Angle, and Repositioning questions
and 38 Max Box, Placement, Shortest Path, and Visibility questions. All records
used explicit layout-file references; scorer-only task labels, parameters,
reference answers, and source provenance remained outside both model and tool
runtime input.

Ollama `qwen3.5:4b` ran with the current eleven-tool set, agent seed 0,
temperature 0, thinking disabled, a shared 2,500-generated-token budget per
question, at most eight turns, and five transient-connection retries. The run
completed under `caffeinate`, and the evaluator and wrapper both exited
normally.

| Task | Correct | Total | Formatting failures |
|---|---:|---:|---:|
| Pair Distance | 37 | 37 | 0 |
| Free Space | 28 | 37 | 2 |
| View Angle | 37 | 37 | 0 |
| Repositioning | 37 | 37 | 0 |
| Max Box | 38 | 38 | 0 |
| Placement | 38 | 38 | 0 |
| Shortest Path | 37 | 38 | 1 |
| Visibility | 37 | 38 | 1 |
| **Overall** | **289** | **300** | **4** |

The final score was **289/300 (96.3%)**. The run took 4,535.204 seconds
(75 minutes 35.2 seconds), generated 118,326 tokens over 856 model calls, and
made 557 tool calls. There were no runtime or tool errors. Placement,
Repositioning, Max Box, Pair Distance, and View Angle all scored 100%, so the
CGAL-backed configuration-space changes introduced no observed agent-level
regression on this set.

All eleven incorrect results are classified as **model failures**. No tool
returned an incorrect value, and there were no geometry, reference-answer,
scoring, or runtime defects. Seven were semantic answer errors, two were
tool-selection or repetition failures that never produced a final answer, and
two contained the correct tool-derived answer but failed the required response
format. The latter expose response-protocol brittleness, but they remain model
compliance failures under the evaluation contract.

Nine of the eleven failures were Free Space questions:

- Seven semantic errors (`free-space-bedroom-135`,
  `free-space-bedroom-245`, `free-space-kitchen-512`,
  `free-space-living_room-352`, `free-space-bedroom-521`,
  `free-space-hssd-41`, and `free-space-living_room-229`) followed the same
  trace: `inspect_room` returned total room area, `occupied_floor_area`
  returned occupied area, and the model immediately mislabeled the occupied
  value as non-occupied instead of subtracting. The first two exactly
  reproduced the two failures in the preceding 50-question run.
- `free-space-hssd-101` called `largest_empty_area` and
  `occupied_floor_area` but not `inspect_room`. It correctly explained that
  largest empty rectangle area is not total non-occupied area, then spent all
  2,500 tokens claiming total room area was unavailable rather than obtaining
  it from `inspect_room`.
- `free-space-kitchen-413` obtained the correct 13.900 result from the
  calculator six times, exhausted all eight turns, and never emitted a final
  answer.

The other two failures also had correct direct tool results. For
`visibility-kitchen-141`, `ray_trace` returned `[chair_1]`; the model then made
seven unrelated calculator calls and ended with an untagged answer. For
`shortest-path-kitchen-551`, `shortest_path` returned the exact reference
waypoints; the model made six redundant calculator calls and repeated the
correct path without the required `*Final answer*:` marker. Both were scored as
formatting failures rather than geometry failures.

The failures suggest three targeted changes:

1. Preserve the intended arithmetic test, but make the agent prompt explicit
   that non-occupied floor area is total room area minus occupied floor area
   and that the calculator should perform that subtraction. A dedicated
   available-area tool would hide the arithmetic capability being evaluated.
2. Add a generic finalization guard: after a tool directly supplies the
   requested result, or after a calculator returns the required arithmetic
   result, the next model turn should answer instead of calling unrelated or
   duplicate tools.
3. Reserve a final-answer-only turn in the evaluator and detect repeated
   identical tool calls. When the normal turn budget is about to expire, the
   evaluator should ask for the required final-answer format with tools
   disabled. This would address all four formatting failures without changing
   reference answers or exposing scorer-only data.

The complete report remains ignored at
`datasets/evaluations/qwen3.5-4b-cgal-seed-0-300.json`.

#### Repeated-tool-call prompt tuning

The Visibility and Shortest Path formatting failures were replayed before and
after strengthening the generic agent prompt. Both evaluations selected the
same two question IDs from the same JSONL and used Ollama `qwen3.5:4b`, seed 0,
temperature 0, thinking disabled, eight turns, five retries, and a
2,500-generated-token budget per question. The only experimental change was the
system prompt.

The original prompt already told the model to accept a direct tool result, but
the deterministic baseline reproduced both failures and their original tool
sequences. The tuned prompt requires the next response after a successful,
direct tool result to be the final answer; it also prohibits verification or
recomputation calls, repeated calls with identical arguments, and calculator
calls on values already supplied by another tool.

| Metric | Before | After |
|---|---:|---:|
| Correct | 0/2 | **2/2** |
| Formatting failures | 2 | **0** |
| Runtime | 122.111 s | **38.930 s** |
| Generated tokens | 3,114 | **1,080** |
| Model calls | 16 | **7** |
| Tool calls | 16 | **4** |

For `visibility-kitchen-141`, the baseline called `ray_trace` once and then
`calculator` seven times. After tuning it called `ray_trace`, made one remaining
unnecessary `calculator` call, and returned the correctly tagged
`["chair_1"]`. For `shortest-path-kitchen-551`, the baseline called
`inspect_room`, `shortest_path`, and then `calculator` six times. After tuning
it called only `inspect_room` and `shortest_path` before returning the correctly
tagged waypoint list. Thus the repeated-call loops and both scoring failures
were eliminated, although the Visibility trace shows that prompt instructions
alone did not completely eliminate irrelevant one-off tool use.

The ignored before and after reports are
`datasets/evaluations/qwen3.5-4b-two-repeat-failures-before-prompt.json` and
`datasets/evaluations/qwen3.5-4b-two-repeat-failures-after-prompt.json`.

As a sanity check, the other nine failures from the 300-question run were then
evaluated with the tuned prompt and compared with their original results:

| Metric | Original 300-run traces | Tuned-prompt replay |
|---|---:|---:|
| Correct | 0/9 | **4/9** |
| Formatting failures | 2 | **0** |
| Runtime | 207.669 s | 200.294 s |
| Generated tokens | 6,185 | 5,216 |
| Model calls | 32 | 35 |
| Tool calls | 24 | 26 |

The prompt fixed `free-space-bedroom-135`, `free-space-kitchen-512`,
`free-space-hssd-101`, and `free-space-hssd-41`. In particular,
`free-space-hssd-101` replaced its irrelevant `largest_empty_area` call with
`inspect_room`, `occupied_floor_area`, and `calculator`, then returned the
correct subtraction.

Five Free Space questions remained incorrect. Four preserved the original
semantic error: the model treated `occupied_floor_area` as if it directly
returned non-occupied area and skipped subtraction. For
`free-space-kitchen-413`, the original model repeatedly obtained the correct
calculator result but never finalized; after tuning it stopped repeating calls
and produced a correctly formatted answer, but prematurely returned the
occupied area instead. This changed the failure from formatting to semantics
without changing its score.

The sanity check therefore found no new scored regression among these nine
already-failing questions and improved four of them, but it exposed an
interaction between the new finalization rule and the model's existing
Free Space misconception. When the model incorrectly classifies
`occupied_floor_area` as a direct answer, stronger finalization instructions
make it commit to that wrong value sooner. The generic no-repeat rule should be
kept separate from a future explicit prompt rule that non-occupied floor area
equals total room area from `inspect_room` minus occupied area from
`occupied_floor_area`. The complete replay is ignored at
`datasets/evaluations/qwen3.5-4b-other-nine-failures-after-repeat-prompt.json`.

#### Occupied-percentage response experiment

The `occupied_floor_area` runtime response was changed from:

```text
The occupied floor area is 9.710 square meters.
```

to:

```text
The occupied floor area is 9.710 square meters (53.235% of total area).
```

The percentage uses the unrounded occupied and room-polygon areas and is
reported to three decimal places. A controlled before/after replay contained
the nine original 300-run failures plus five controls sampled with
`random.Random(0)` from the 28 Free Space records that were correct in that run:
`free-space-bedroom-305`, `free-space-bedroom-196`,
`free-space-bedroom-354`, `free-space-kitchen-356`, and
`free-space-bedroom-123`. Both 14-question runs used the same current system
prompt, shortened `occupied_floor_area` tool description, question order, model,
seed, temperature, thinking setting, turn limit, retry count, and token budget.
The runtime response was the only experimental change.

| Metric | Area only | Area plus percentage |
|---|---:|---:|
| Overall correct | 10/14 | **10/14** |
| Original-failure subset | 7/9 | **7/9** |
| Control subset | 3/5 | **3/5** |
| Formatting failures | 4 | 4 |
| Runtime | 410.507 s | 438.910 s |
| Generated tokens | 9,336 | 9,639 |
| Model calls | 73 | 72 |
| Tool calls | 63 | 62 |

The percentage caused no scored improvement or regression. Correctness was
identical on every record, and the ordered tool sequence was identical on 13 of
14. On `free-space-living_room-352`, the model dropped one redundant calculator
call and remained correct. On `free-space-kitchen-413`, it added one correct
calculator call but then continued with five irrelevant entity inspections and
remained incorrect.

The experiment also exposed a separate tool-selection effect that was already
present in the area-only baseline. Controls `free-space-bedroom-123` and
`free-space-bedroom-196`, plus former failure `free-space-hssd-101`, never
called `occupied_floor_area`; each called `inspect_room` followed by seven
`inspect_entity` calls and exhausted the turn limit. The percentage cannot
affect those traces because it is visible only after the tool is selected.
Between the preceding nine-question replay and this baseline, the relevant
model-input change was shortening the tool description from explicit
inclusion/exclusion semantics to the opaque phrase `it correctly excluded`.
That removed the terms matching the Free Space question and is therefore the
cause supported by the controlled evidence for the changed routing.

The ignored reports are
`datasets/evaluations/qwen3.5-4b-free-space-14-before-occupied-percent.json`
and
`datasets/evaluations/qwen3.5-4b-free-space-14-after-occupied-percent.json`.

A fresh 20-layout, seed-7 generation emitted all 160 task records, retained the
exact 10/10 Placement balance, and passed schema, prompt, reproducibility,
geometry-witness, placement-certificate, path, Max Box convergence, and
byte-for-byte determinism checks at 100%. Complete-layout yield was 20/22
(90.9%), below the 95% quality gate solely because two released bedroom layouts
contain a degenerate `mirror` polygon; this is an input-data validation issue,
not a geometry-seed failure. Directly replaying the generated visible arguments
through the agent runtime matched all 20/20 Placement answers and all 20/20
rounded Max Box answers.

On the same 20-layout seed-7 sample, the contact-event/SHGO solver changed 9 of
20 Max Box references while retaining valid witnesses for all 20. Seven changes
were larger than 2%; the largest improvements were HSSD 34 (+79.3%), HSSD 88
(+73.0%), kitchen 523 (+32.8%), living room 117 (+12.9%), living room 146
(+127.9%), bedroom 207 (+4.6%), and HSSD 106 (+5.3%). SHGO found a global
minimizer pool directly on 19 layouts; bedroom 390 had a narrow feasible set,
so its already-valid baseline witness was locally refined instead. Generation
plus all quality checks completed in about 23 seconds. Every gate passed at
100% except the pre-existing complete-layout-yield gate caused by the two
degenerate `mirror` polygons. Repeating the complete generation produced a
byte-identical JSONL file and a 100% deterministic-match score.

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
