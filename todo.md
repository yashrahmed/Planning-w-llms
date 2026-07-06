## Goal - [Career - How agent models/datasets are built] - Planning with LLMs.

Design an agent harness + fine tune a model for that harness to beat or match some of the benchmarks below.
TBD - Narrow down the choice of datasets.

### Version 1
- [ ] Build a harness for a model to beat [FloorplanQA].
  - [Layouts](https://huggingface.co/datasets/OldDelorean/FloorplanQA-Layouts).
  - See generator for [Questions](https://github.com/OldDeLorean/FloorplanQA).
- [ ] Learn about building [reasoning LLMs](https://www.manning.com/books/build-a-reasoning-model-from-scratch).
- [ ] See [SpatialClaw](https://spatialclaw.github.io/?linkId=100000426730902)

### Things to check

### Text-Only LLM Spatial Datasets

| Dataset Name [1, 2, 3] | Primary Data Format | Spatial Concepts Tested | Key Purpose / Feature |
|---|---|---|---|
| PlanQA[](https://olddelorean.github.io/PlanQA/) | JSON / XML structural files | Room layouts, sightlines, distance | Evaluates furniture fitting and geometric constraints using textual logs. |
| SnorkelSpatial[](https://snorkel.ai/blog/introducing-snorkelspatial/) | Alphanumeric coordinate strings | Grid placement, egocentric perspective | Uses narrative movement logs to test if a model can track changing positions. |
| SpatialEval (TQA)[](https://github.com/jiayuww/SpatialEval) | Descriptive matrix strings | Maze tracking, text coordinates | Isolates spatial thinking by stripping away all visual hints. |

### Multimodal LLM (MLLM) Spatial Datasets

| Dataset Name [4, 5, 6, 7, 8] | Primary Data Format | Spatial Concepts Tested | Key Purpose / Feature |
|---|---|---|---|
| SpatialBench[](https://huggingface.co/datasets/RussRobin/SpatialBench) | 2D images + Text prompts | Visual paths, object sizes, metrics | Organizes tests into five distinct tiers based on cognitive map theory. |
| SpatialEval (VQA)[](https://huggingface.co/datasets/MilaWang/SpatialEval) | Images / Matrices + Text | Counting, route navigation, placement | Mixes pure visual tasks with combined text-vision problems. |
| SpatialQA[](https://huggingface.co/datasets/RussRobin/SpatialQA) | RGB-D (Depth) photos + Text | True depth perception, distances | Merges 750,000 real-world computer vision images to check foreground/background awareness. |
| Surprise3D[](https://mbzuai-liziwen.github.io/Surprise3D/) | 3D scenes + Text prompts | Volume space, absolute coordinates | Hides object names in prompts to force models to rely only on physical geometry. |
| CA-VQA[](https://github.com/apple/ml-cubifyanything) | Environment images + Text | 3D bounding boxes, object interactions | Checks if an MLLM can accurately predict the outer boundary edges of physical shapes. |


[1] [https://olddelorean.github.io](https://olddelorean.github.io/PlanQA/)
[2] [https://snorkel.ai](https://snorkel.ai/blog/introducing-snorkelspatial/)
[3] [https://github.com](https://github.com/jiayuww/SpatialEval)
[4] [https://huggingface.co](https://huggingface.co/datasets/RussRobin/SpatialBench)
[5] [https://huggingface.co](https://huggingface.co/datasets/MilaWang/SpatialEval)
[6] [https://huggingface.co](https://huggingface.co/datasets/RussRobin/SpatialQA)
[7] [https://mbzuai-liziwen.github.io](https://mbzuai-liziwen.github.io/Surprise3D/)
[8] [https://github.com](https://github.com/apple/ml-cubifyanything)
