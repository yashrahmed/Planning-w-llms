# IFF QA Pipeline (Simple, Authoritative)

## Goal
Build a QA pipeline that answers aircraft questions with high precision using:
- Structured facts as the primary source of truth
- BM25 for exact lexical evidence retrieval
- Embedding-based RAG for semantic fallback and synthesis

## Pipeline Design

1. Ingest and normalize source docs
- Input: `resources/corpus/*.txt`
- Normalize whitespace and split into chunks (with overlap)
- Keep provenance for each chunk: `file`, `chunk_index`, offsets

2. Extract structured facts
- Use an LLM with a strict JSON schema to extract explicit facts only
- Example fact fields:
  - `aircraft`
  - `fact_type` (operator_country, role, service_entry, etc.)
  - `value`
  - `start_year` / `end_year` (optional)
  - `source_file`
  - `source_chunk`
  - `evidence_quote`
  - `confidence`
- Validate and canonicalize entities (country names, aircraft variants)
- Persist facts in SQLite

3. Build retrieval indexes
- BM25 index over raw chunks (`title/file + chunk text`)
- Embedding matrix over chunks (`text-embedding-3-large`)
- Persist:
  - `embeddings.npy`
  - `embeddings_index.json`
  - BM25 index artifacts

4. Query routing
- Parse intent and entities from query
- If query maps to structured fields (e.g. "used in France"):
  - Run structured SQL lookup first
- Else:
  - Run hybrid retrieval (BM25 + vector)

5. Hybrid retrieval and ranking
- Retrieve top-N BM25 chunks
- Retrieve top-N semantic chunks
- Merge and rerank with weighted score:
  - `final_score = a * bm25 + b * cosine + c * source_trust`
- Require lexical grounding for entity-specific questions

6. Answer synthesis with citations
- Generate answer using only selected evidence and/or structured facts
- Return:
  - direct answer
  - supporting citations (`file`, chunk, quote)
  - confidence
  - "insufficient evidence" when thresholds fail

7. Quality gates
- Rule-based checks before final output:
  - minimum evidence count
  - evidence-source diversity
  - contradiction detection
- Maintain a small eval set of canonical questions and expected cited answers

## Implementation Tasks

1. Data + indexing
- [ ] Add chunker utility shared by embed/search/fact extraction
- [ ] Add BM25 index builder for chunk corpus
- [ ] Keep current embedding flow and persist metadata

2. Structured fact layer
- [ ] Add `scripts/extract_facts.py` for schema-constrained LLM extraction
- [ ] Add SQLite schema and upsert pipeline
- [ ] Add canonicalization map for aircraft/country aliases

3. Search + QA API
- [ ] Extend `scripts/search.py` to:
  - structured lookup first
  - BM25 + vector fallback
  - merged reranking
- [ ] Add `scripts/answer.py` to generate cited final responses

4. Reliability
- [ ] Add thresholds for "insufficient evidence"
- [ ] Add regression checks with 10-20 benchmark questions
- [ ] Log query traces (retrieved chunks, scores, selected evidence)

## Minimal Success Criteria
- Query: "Which aircraft was used in France?"
  - returns aircraft list from structured facts when available
  - includes at least one explicit France-linked citation
  - avoids unsupported claims when evidence is weak
