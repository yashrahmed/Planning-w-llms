# LLM Grounded Planning assistant.

## Goal
Build a scene that given a natural langauge context and a planning query can interview a user and generate a complete planning that is grounded in 3D physical constraints e.g. when asked to plan a camping trip, the system must be able to plan sub tasks like driving, packing, refueling, activities, camp set up while accounting for weather etc.

## Additional Goal
Develop a planning library that supports multiple classical planning algorithms, with variant Typescript and possibly Rust.

## High level idea
Given a natural langauge context and a planning query, retrieve the relevant frames for all grounded "scenes" and try slot filling with a human in the loop. Repeat until all constraints are satisfied.

Reference summary: [Frame-Grounded-Hierarchical-Planning-Summary.docx](./Frame-Grounded-Hierarchical-Planning-Summary.docx)

## Question and Answer.

### How would one generate an exhaustive list of scene frames and their slots?
- Using Raw videos as a source, implement the following pipelne.
     ```
     Video
     -> VLM API call
     -> structured JSON with:
          start, end, scene_description
     -> add records to Scene Description Dataset

     Scene Description Dataset
     -> call LLM with scene_description and extraction prompt
     -> LLM outputs a list of semantic scene frames
     -> add records to Scene Frame Dataset
     ```
- For starters, the second half of the above pipeline can be tested using the following *half* pipeline
     ```
     Existing dataset text
     -> select step or scene descriptions
     -> LLM extraction prompt
     -> scene labels / semantic frames
     -> normalization
     -> evaluation
     ```
     Here are some datasets. Extract the annotations from them to build the text dataset.
     | Dataset | Link | Notes |
     |---|---|---|
     | DeScript | [ACL Anthology](https://aclanthology.org/L16-1556/) | Crowdsourced event-sequence descriptions for everyday scenarios. |
     | proScript | [AllenAI project page](https://proscript.allenai.org/) | Partially ordered scripts for everyday activities. |
     | CrossTask | [GitHub repository](https://github.com/DmZhukov/CrossTask) | Instructional video dataset with ordered step lists for 83 tasks. |
     | HowTo100M | [Project page](https://www.di.ens.fr/willow/research/howto100m/) | Large-scale narrated instructional video dataset with clips and captions. |
     | WikiHow dataset | [arXiv paper](https://arxiv.org/abs/1810.09305) | Large-scale procedural text dataset built from WikiHow article-summary pairs. |
          

### What does the retrieval mechanism for these scene frames look like?
- I have no idea. One approach is to dump the input and potential scene /w descriptions into an LLM and yolo it.
- Same ideas as the previous one but use some refined RAG approach.


## Todo List

### What to do next?
- Find ways to extract activity annotations (labels) from the above datasets.
- Think on the side about how to build game like RL enviroments for agents.