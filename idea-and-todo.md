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

__________

### Idea scratchpad

Plan a camping trip at Athelstane, WI.

When are we going there?
Who are the people traveling?
When are they traveling?
How long is the trip? Is the duration of stay the same for everyone?
What is each person carrying?
How will each person get there?
What vehicles (if any) are they using?
How will the things being carried by each person be packed?
What route will the vehicles take?
Are there any driving constraints like the fuel and EV charging time?
What food will be required/be available to each of them?
How will that food be cooked?
...
...
...

Meal Prep Frame
- Cuisine/Food name
- Ingredients (with qty).
- Prep conditions.
- Steps.

Cooking frame
- Cuisine/Food name
- Ref<Meal Prep Frame>
- Utensils/Tools

Utensil prep frame
- Utensil name
- Precondition
- Post condition
  
Accessible tools frame -- use to ensure that tools are easy to get to e.g. a ladle required for prep cannot be a hundred miles away.
- Accessing agent (who needs it)
  - Location
- Tools
  - Ref<Utensil Prep Frame>
  - Location


  ....

There are a large number of frames for any scenario.
To truly ground a scene, many frames have to be filled out.
I may have to brute force a list of these frames based on 
 - Life experience.
 - Action datasets.

- Perhaps a decomposition worklow would make it easy to constrain the relations. Let's start with a procedural approach and then gradually move over to a declarative approach.
- LLM coding agents can help generate and decompose multiple workflows.
- Workflows encode some aspect of cognitive processes instead of forcing the LLM to figure out the cognitive processes.
Let's start with camping trip planning


--- An example of code placement plan
1. Describe the logic change. 
2. What code flows would be affected?
3. In each code flow what components would be affected?
4. Will the resulting change lead to duplicated code?
5. If so, factor it out.
6. Refactor.
7. Update the flow structure.
