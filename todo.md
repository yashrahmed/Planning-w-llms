### Next steps
- Complete the tutorial on "training small reasoning models".
- Run an experiment to build a simple planning dataset (see travel blogs).
- Research fine tuning models for agentic travel planning.
- Integrating classical planners into the LLM's toolkit (or build a workflow to that effect).
- See [Harness 1](https://arxiv.org/pdf/2606.02373)

### What already exists

**Travel-planning benchmarks:**
- [TravelPlanner](https://arxiv.org/abs/2402.01622) (Xie et al. 2024 — [GitHub/site](https://osu-nlp-group.github.io/TravelPlanner/), [dataset](https://huggingface.co/datasets/osunlp/TravelPlanner)) — the foundational one. A sandbox of ~4M crawled records behind six tools (flights, accommodation, restaurants, attractions, distance), 1,225 queries, each with a combination of environment, commonsense, and hard constraints checked programmatically. **This is the architecture to study first.**
- [Flex-TravelPlanner](https://arxiv.org/abs/2506.04649) — extends it with constraints introduced sequentially across turns and explicitly prioritized competing constraints.
- [GroupTravelBench](https://arxiv.org/abs/2605.25200) — multi-user, multi-turn travel planning.
- [DeepPlanning](https://qwenlm.github.io/Qwen-Agent/en/benchmarks/deepplanning/) (Qwen — [dataset](https://huggingface.co/datasets/Qwen/DeepPlanning), [paper](https://huggingface.co/papers/2601.18137)) — closest to the long-horizon framing: multi-day travel and multi-product shopping requiring proactive information acquisition plus local and global constrained optimization.

**General long-tool-chain datasets:**
- [TOUCAN](https://arxiv.org/abs/2510.01179) ([GitHub](https://github.com/TheAgentArk/Toucan), [dataset](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)) — 1.5M trajectories synthesized from ~500 real MCP servers / 2,000+ tools.
- Synthesis lineage: [APIGen](https://arxiv.org/abs/2406.18518) and [ToolACE](https://arxiv.org/abs/2409.00920).
- Step-level long-horizon evals like [LongCLI-Bench](https://huggingface.co/papers/2602.14337) ([GitHub](https://github.com/finyorko/longcli-bench)).