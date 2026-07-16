# FloorplanQA question generation

## Findings

FloorplanQA does not use an LLM to write its questions. An LLM generated the
synthetic room layouts, but fixed prompt templates create the questions and
deterministic geometry algorithms calculate their reference answers. The
original benchmark poses one question of each of the eight task types for every
layout, producing 16,000 questions from 2,000 layouts.

Sources consulted:

- [FloorplanQA paper, version 4](https://arxiv.org/html/2507.07644v4)
- [Official FloorplanQA repository](https://github.com/OldDeLorean/FloorplanQA)
- [FloorplanQA layouts on Hugging Face](https://huggingface.co/datasets/OldDelorean/FloorplanQA-Layouts)
- The paper's TeX source, taxonomy, supplementary examples, and case studies
- The released prompt templates and eight task-generation modules under
  `fpqa-tooling/src/`

The Hugging Face release contains layouts only. It does not include the
generated questions, reference-answer CSV files, or the exact artifacts used
for the published evaluation.

## Published examples

The paper's taxonomy and Appendix J provide examples for all eight task types.

| Type | Published example or case | Answer format |
| --- | --- | --- |
| Pair distance | Fridge to stove in the taxonomy; sink to shower in HSSD bathroom 14 | Scalar meters |
| Free space | Total unoccupied area of HSSD game room 147 | Scalar square meters |
| View angle | Sofa to TV in the taxonomy; `chair_2` to `window_1` in kitchen 50 | Scalar degrees |
| Repositioning | Ottoman moving left in the taxonomy; `bin_2` moving left until it reaches the bathtub in HSSD bathroom 26 | Scalar meters |
| Max box | Largest rotated empty rectangle; the case illustration uses living-room layout 36 | Scalar square meters |
| Placement | A 2 by 3 meter desk in the taxonomy; a 2.5 by 1.0 meter antique storage chest in living-room layout 0 | Boolean |
| Shortest path | Stove to door in the taxonomy; TV to `armchair_2` with 15 cm clearance in living-room layout 26 | Waypoint sequence |
| Visibility | Window to fireplace in the taxonomy; window to bin in an office, with table and armchair reported as intersections | Object-label set or list |

Two case-study layouts can be matched directly to the downloaded dataset. A
fresh computation from the released polygons gives:

- HSSD `room_14`, sink to `shower_2`: **1.514 m**
- Kitchen `room_50`, `chair_2` to `window_1`: **100.081 degrees**

These two values are independently recomputed results, not numbers printed in
the paper.

The supplementary material also contains two numeric failure cases that are
useful as regression tests:

- A Max Box example has a reference area of **7.57 m2**, while model-written
  code found only 5.57 m2.
- A Repositioning example allows a dishwasher to move downward **1.96 m**,
  while model-written code incorrectly returned zero.

## Overall generation algorithm

Use a stable, seeded pipeline with one parameter selector and one reference
solver per task:

```text
for each layout in stable layout order:
    geometry = normalize_and_validate(layout)

    for each of the eight task types:
        rng = RNG(SHA256(global_seed, layout_id, task_type))
        parameters = select_task_parameters(geometry, rng)
        answer = solve_task(geometry, parameters)
        question = render_fixed_template(task_type, layout, parameters)
        emit(question, answer, metadata)
```

Do not seed with Python's `hash(layout_id)`, as the released code does in
several places. Python randomizes `hash()` between interpreter processes.
SHA-256 or BLAKE2 over the global seed, layout ID, and task name gives stable
results across machines and runs.

The original benchmark generates every task type for every layout. A smaller
sample should therefore be stratified by task type and room source. Sampling
task types without stratification produces a different distribution from the
benchmark.

The generator interface should select layouts, not individual task types. Each
selected layout should emit all eight task records. For example, requesting 20
layouts should produce 160 questions. If a caller needs a fixed number of
questions instead, that should be a separate explicitly stratified sampling
operation over an already generated eight-task-per-layout collection.

## Shared geometry preparation

For every layout:

1. Build the room polygon from `room_boundary`, falling back to polygonized
   walls when necessary.
2. Repair invalid polygons with a deterministic equivalent of Shapely's
   `make_valid`.
3. Clip object polygons to the room where appropriate.
4. Build an entity collection containing furniture, doors, and windows.
5. Classify labels into task-specific groups:
   - Soft coverings: rug, carpet, mat, doormat, and runner.
   - Ceiling-only fixtures: light, chandelier, fan, and pendant.
   - Openings: doors and windows.
6. Compute area-weighted polygon centroids with the shoelace formula. Use
   vertex averaging only for a genuinely degenerate polygon.

The filters are task-specific. For example, rugs count as occupied area in the
Free Space task but are nonblocking in Max Box, Placement, Repositioning, and
Shortest Path.

## Task algorithms

### Pair distance

1. Select two distinct entities, including doors and windows.
2. Compute each entity's polygon centroid.
3. Return the Euclidean distance between the two centroids.

### Free space

1. Build the room polygon.
2. Collect all object polygons except ceiling-only fixtures.
3. Do not treat doors or windows as occupied floor area.
4. Form the geometric union of occupied polygons before measuring it, so
   overlapping objects are not double-counted.
5. Return:

   ```text
   area(room - union(occupied objects))
   ```

Rugs and other floor coverings count as occupied for this task because the
released prompt says to ignore openings and ceiling fixtures but consider all
other objects.

### View angle

1. Select two distinct entities.
2. Compute their centroids `c1` and `c2`.
3. Form and normalize `d = c2 - c1`.
4. Compute `clip(dot(d, (0, 1)), -1, 1)`.
5. Return `acos(dot)`, converted to degrees in the range 0 through 180.

### Repositioning

1. Select movable furniture, excluding openings, rugs, and ceiling fixtures.
2. Select one of up, down, left, or right.
3. Translate the complete object polygon along that axis.
4. Find the first distance at which the translated polygon would leave the
   room or overlap a blocking object.
5. Return the last valid translation distance.

The released solver advances in 0.01 meter increments. A cleaner solver can
perform continuous collision detection, or scan to find the first collision
interval and then refine it with bisection. It must find the first collision,
not merely test whether a later pose is free, because an object cannot pass
through an intervening obstacle.

The new generator should not use the 0.01 meter stepping algorithm for its
reference answer. Compute the earliest collision distance continuously from
the moving polygon against the room boundary and blocking polygons. If a fully
analytic solution is impractical for a geometry, bracket the first collision
and refine it deterministically with bisection to a declared distance
tolerance.

### Max box

1. Form the blocking union from objects and openings, excluding rugs and
   ceiling fixtures.
2. Compute free space as `room - blocking_union`.
3. Generate candidate orientations from room and obstacle edge directions,
   supplemented by a uniform angular grid over `[0, pi)`.
4. For each orientation, rotate free space by the negative angle and find the
   largest axis-aligned rectangle contained in it.
5. Rotate the candidate rectangle back and retain the maximum area.

The released implementation is approximate: it uses 300 random starting
points, 12 angles, coordinate-ascent growth, seed 42, and a 30-second budget.
A sparse center-and-angle sample can miss a larger valid rectangle between
samples and therefore underestimate the reference answer.

The exact target is the contact-event algorithm in
[Maximum-Area Rectangles in a Simple Polygon](https://arxiv.org/abs/1910.08686),
which handles arbitrary orientations and polygonal domains with holes. The
current implementation uses the following deterministic hybrid:

1. Enumerate Type-A contact events by treating every pair of boundary vertices
   as opposite corners of a square and retaining only fully contained squares.
2. Parameterize remaining rectangles by center, width, height, and rotation.
3. Seed the search with connected-component representatives, polygon
   triangulation representatives, and a fixed Halton sequence.
4. Use SciPy SHGO with the rectangle's outside area as a nonlinear feasibility
   constraint to search continuous Type B-F contact configurations.
5. Use exact Shapely containment and collision checks as hard feasibility
   tests for every proposed candidate.
6. Stop at a documented convergence tolerance and record the best valid lower
   bound, tolerance, iteration count, and convergence status. A layout or
   generation seed must never affect the search.

The result is still a numerical lower bound rather than a symbolic proof of the
global maximum because the full Type B-F staircase and ray-shooting event map
has not been ported. The tool description and provenance therefore record
`global_optimum_certified: false` and must not claim greater precision than the
solver's convergence tolerance.

### Placement

1. Select a named rectangle with fixed width and depth from a room-specific
   object catalog.
2. Form free space by removing blocking objects and openings while ignoring
   rugs and ceiling fixtures.
3. For each candidate rotation `theta`, construct the rotated query rectangle
   `Q(theta)`.
4. Compute the valid-center region using configuration-space erosion:

   ```text
   valid_centers(theta) = free_space minus Q(theta)
   ```

   Here `minus` denotes geometric erosion or Minkowski difference, not ordinary
   polygon subtraction.
5. Return `True` if the valid-center region is nonempty at any angle; otherwise
   return `False`.

Use adaptive rotation refinement rather than a fixed set of center and angle
samples. Shapely should provide the final containment and collision decision
for every candidate pose. A deterministic global optimizer may be used as a
fallback to search `(x, y, theta)`, but configuration-space feasibility is
preferred because a nonempty valid-center region directly witnesses a valid
placement. A sampled solver may safely establish `True` by finding a valid
pose, but it cannot justify `False` merely because its finite samples failed.
The implementation should report its angular and geometric tolerances and
retain a witness pose for every `True` answer.

The exact deterministic target is the largest-similar-convex-polygon algorithm
in
[Largest similar copies of convex polygons amidst polygonal obstacles](https://arxiv.org/abs/2012.06978).
For a requested rectangle it computes the largest feasible scale `s`; the
rectangle fits exactly when `s >= 1`. The current implementation removes RNG
from its candidate enumeration and validates every positive witness, but it
does not treat an unsuccessful finite search as a negative certificate.

The paper reports a nearly balanced placement target distribution of 49.9%
`True`. The local generator enforces an exact 50/50 `True`/`False` split for
every registered Boolean task. It creates a seeded answer schedule, then uses
deterministic catalog rejection sampling to select a witnessed positive or a
certified negative for the class assigned to each emitted layout. When no fixed
catalog rectangle provides an area-based negative certificate, the largest
catalog rectangle is scaled deterministically until its three-decimal
dimensions have more area than the largest connected free-space component.
Exact balance requires an even requested layout count.

### Shortest path

1. Select two eligible entities, excluding rugs.
2. Compute their centroids.
3. Buffer every blocking obstacle by the required clearance and shrink the
   room boundary by the same amount.
4. Remove the source and target entities from the blocking union, while still
   connecting their centroids safely into navigable space.
5. Find the shortest collision-free waypoint sequence.

The paper specifies 0.15 meter clearance and A* over a navigable grid. The
released code instead uses 0.10 meter clearance and a visibility graph followed
by Dijkstra. The new generator should use the paper definition by default:
0.15 meter clearance and grid A*. Grid resolution, connectivity, centroid
connection rules, and path simplification must be deterministic and recorded
in provenance. A visibility-graph solver may remain as an optional
non-benchmark experiment, but it must not generate the default reference path.

### Visibility

1. Select two distinct entities.
2. Form the line segment between their polygon centroids.
3. Test the segment against every other entity polygon.
4. Exclude the source and target and ignore intersections consisting only of a
   segment endpoint.
5. Sort hits by projected distance from the source for deterministic output.

The paper scores this answer using set equality, so ordering should not affect
correctness. The paper text refers to bounding boxes while the released code
uses the actual polygons; polygon intersections are more consistent with the
dataset's unified polygon representation. The new generator will intentionally
retain actual-polygon intersection rather than add a paper-style bounding-box
mode. This is a documented semantic choice: for irregular HSSD objects, a line
can cross empty space inside an object's bounding box without intersecting the
object itself.

## Released-layout validation

The paper reports filtering roughly one-third of synthetic candidates for
overlaps, blocked doors, inadequate clearance, and improper attachment. The
released corpus contains only the 2,000 layouts that survived that process, so
the question generator does not need to reproduce the missing candidate
generation and selection pipeline.

It should nevertheless validate the released input before generating
questions. At minimum, check that:

1. The room boundary is a valid, nonempty polygon.
2. Entity polygons are valid and remain within the declared room to a numeric
   tolerance.
3. Doors and windows are attached to a room boundary or declared wall.
4. Labels used by a question identify unambiguous entities.
5. Blocking-object overlaps are either allowed semantic pairs or are reported.
6. Path endpoints are connectable to navigable free space.

These checks audit the fixed released corpus; they do not claim to reproduce
the authors' unpublished validity filter or infer whole-house connectivity.
Validation results and warnings should be included in record provenance, and a
layout with an error relevant to a task should not produce that task's record.

## Prompt and output construction

The original fixed prompt templates live in
[`fpqa-tooling/src/evaluation/questions.py`](fpqa-tooling/src/evaluation/questions.py).
The local generator keeps their task-specific variables such as object labels,
dimensions, direction, and clearance, but separates the serialized layout from
the question. The layout file is emitted beside `questions.jsonl` with a
collision-safe `<source>-<layout_id>-<split>.json` name. The question contains
`Room layout can be found in file : <filename>` and does not embed the layout
JSON.

Every generated record should contain at least:

```json
{
  "id": "<task>-<source>-<layout_id>",
  "layout_id": "<layout id>",
  "room_type": "<room type>",
  "split": "train",
  "layout_file": "<source>-<layout_id>-train.json",
  "task": "<task name>",
  "parameters": {},
  "question": "<rendered prompt>",
  "answer": "<typed reference answer>",
  "messages": [],
  "provenance": {
    "global_seed": 0,
    "solver_version": "<version>",
    "compatibility_mode": "paper"
  }
}
```

Prompts require the final response on a line beginning with `*Final answer*:`.
Numeric reference answers should be stored at full useful precision and
rendered to three decimal places in prompts and expected model responses.

## Scoring

The paper's evaluation protocol uses:

- Scalar results: relative error at most 2%.
- Free-space area: relative error at most 5%.
- Placement: exact Boolean equality.
- Visibility: set equality.
- Shortest path: collision-free under the required clearance and Frechet
  distance no greater than 0.6 meters from the reference path.

The evaluator should parse the final-answer line strictly but retain a separate
diagnostic showing whether a semantically correct answer failed only because of
formatting.

## Release inconsistencies to resolve

A clean generator should document its choices for the following discrepancies:

- The paper's shortest-path clearance and A* algorithm differ from the
  released code's clearance and visibility-graph solver.
- Pair-distance code rounds to two decimals while the prompts require three.
- The paper describes Visibility using bounding boxes while the code checks
  actual polygons. The new generator intentionally uses actual polygons.
- The Placement case says 2.5 by 1.0 meters, but its ground-truth paragraph
  discusses a 2 by 1.5 meter rectangle.
- View Angle, Visibility, and Repositioning load an unreleased
  `swapped_labels` dataset.
- The office-42 Visibility case does not match the labels in released HSSD
  `room_42`.
- The Max Box case caption describes layout 36 as a bedroom even though the
  filename and released layout identify it as a living room.

## Recommended implementation strategy

Build a new self-contained generator rather than patching the upstream scripts
in place. It should provide:

1. Stable SHA-based sampling.
2. One shared geometry normalization layer.
3. All eight task records for every selected layout.
4. Eight independently testable reference solvers.
5. Paper-style 0.15 meter grid A* for default shortest-path references.
6. Continuous first-collision computation for Repositioning.
7. Configuration-space placement and deterministic global optimization for Max
   Box, both backed by exact collision checks and declared tolerances.
8. Actual-polygon Visibility as an intentional documented divergence.
9. Sanity validation of the fixed released layouts without attempting to
   recreate their candidate-selection pipeline.
10. Solver parameters, convergence data, validation results, and version
    provenance in every record.
11. Golden tests using the concrete cases above.
12. Property tests covering polygon vertex order, overlaps, boundary contact,
    soft-covering filters, solver witnesses, and deterministic regeneration.

These corrections are implemented by the self-contained `paper-v2` generator
under `src/floorplan_qa/`.

## Quality metrics and validation

### Acceptance thresholds

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

### Distribution measurements

The observed room-source distribution is compared with the released corpus:
30% bedroom, 10% HSSD, 30% kitchen, and 30% living room. Placement `True` rate
is also reported. These are advisory because layouts use a uniform shuffle and
Placement uses a uniform catalog order with rejection only when the solver
cannot produce a witnessed or certified answer; neither is selected to hit a
target balance.

### Evaluation result

Attempt 4 evaluated 800 records from 100 uniformly sampled layouts using seed
`20260712`. The same sample was generated twice and the evaluator independently
regenerated all records.

All hard metrics reached 100%, except complete layout yield, which was 99.0%
(100 emitted layouts from 101 considered layouts) and exceeded its 95%
threshold. Max Box convergence used a 2% relative-improvement threshold.

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
