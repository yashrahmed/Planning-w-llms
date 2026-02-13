# IFF - [Identification Friend or Foe](https://en.wikipedia.org/wiki/Identification_friend_or_foe)

An LLM-powered search engine for querying aircraft information using natural language.

## Overview

IFF enables users to search and discover aircraft details through conversational queries. The system leverages large language models to understand natural language inputs and performs intelligent, multi-step searches to find and synthesize relevant information.

## Key Features

- **Natural Language Queries**: Search using conversational language with full conversation context
- **Agentic Search**: Automatically performs multiple searches to comprehensively answer complex queries
- **Information Synthesis**: Combines data from multiple sources into coherent responses
- **Rich Output Formats**: Presents results as tables, graphs, timelines, and other visual formats where appropriate
- **Hybrid Search**: Combines embedding similarity and semantic parsing for accurate results
- **Visual Function Blocks**: Built on a modular, visual programming paradigm

## Technology Stack

- **Search Engine**: Typesense or SQLite
- **Vector Store**: TBD
- **Prompt Optimization**: DSPy
- **Architecture**: Visual function blocks

## Notes

- Create notes on React internals for later use
- See the [langextract library](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)

## Planned Platforms

- Web application
- Mobile application
- ChatGPT app integration

## Development Roadmap

1. Prototype search system with structured queries
2. Natural language to search query translation
3. Integration testing and iteration
4. DSPy pipeline for prompt optimization
5. Apply optimized prompts (if applicable) and perform benchmarking
6. Build the IFF web app and integrate with the API, then iterate
7. Build the IFF mobile app
