# GREAT Experiment Checkpoint — v0.2

## Status

This repository checkpoint freezes the GREAT green-retrofit RAG evaluation setup before the formal repeated-run experiment.

The current benchmark and evaluation pipeline should NOT be modified to improve model scores.

---

## 1. Frozen benchmark

Benchmark file:

`evaluation/qa/standards_questions_48.jsonl`

Total questions: 48

Distribution:

- BREEAM: 12
- Cost: 12
- Energy: 12
- WLCA: 12

Difficulty:

- Easy: 16
- Medium: 16
- Difficult: 16

Question types include:

- Retrieval
- Concept
- Process
- Compliance
- Trade-off
- Domain reasoning
- Scope
- Cross-domain synthesis

Cross-domain synthesis questions: 12.

The benchmark contains 48 unique question IDs.

The benchmark has been manually reviewed and validated programmatically.

---

## 2. Knowledge corpus

Local corpus:

- 18 PDF documents
- Approximately 58 MB

Domains:

- BREEAM: 4 PDFs
- Cost: 4 PDFs
- Energy: 6 PDFs
- WLCA: 4 PDFs

The PDF files are intentionally NOT committed to Git because of copyright/licensing considerations.

They are protected by `.gitignore`.

The local corpus should be restored before rebuilding the RAG database.

---

## 3. RAG corpus version

Current frozen corpus configuration:

- Corpus version: v0.2
- Documents: 18
- Total chunks: 4,856
- Chunk size: 1,200 characters
- Chunk overlap: 200 characters
- Embedding model: `text-embedding-3-large`
- Vector database: ChromaDB
- Local database directory: `.chroma_great`

Current local Chroma database size is approximately 210 MB.

`.chroma_great` is generated data and is intentionally NOT committed to Git.

---

## 4. RAG configuration

Specialist retrieval:

- Specialist top-k: 6

Single-agent retrieval:

- Single-agent top-k: 8

Evaluation temperature:

- `0.2`

GREAT currently evaluates each benchmark question using the four specialist agents:

1. BREEAM
2. Cost
3. Energy
4. WLCA

The planner then synthesises the specialist outputs.

---

## 5. Experimental conditions

Three controlled systems:

### A. Generic LLM

Generic LLM without RAG.

### B. Single-agent RAG

One RAG agent operating over the combined knowledge corpus.

### C. GREAT multi-agent RAG

Planner + four domain-specialist agents.

The objective is to compare retrieval grounding and multi-agent decomposition under controlled conditions.

---

## 6. Pilot experiment

Pilot benchmark:

48 questions × 3 systems = 144 responses.

Semantic evaluation has been completed for the pilot.

Pilot output files:

- `logs/qa_semantic_metrics_run_01.csv`
- `logs/qa_semantic_outputs_run_01.jsonl`

The pilot is diagnostic and is NOT the final repeated-run experiment.

The pilot must not be treated as 144 independent observations.

---

## 7. Formal experiment

Formal experiment planned:

48 questions × 3 systems × 5 repeated runs

= 720 generated responses.

Repeated runs are intended to assess stochastic consistency.

They should NOT be treated as 720 independent statistical observations.

The formal experiment has NOT yet been completed at this checkpoint.

---

## 8. Evaluation principles

The benchmark is frozen before the formal experiment.

Do not modify questions, expected facts, forbidden facts, or source assignments after observing formal experiment results merely to improve system performance.

Semantic fact evaluation uses:

- 0 = absent/incorrect
- 1 = partially supported/incomplete
- 2 = clearly supported

Forbidden-fact violations are recorded separately.

Retrieval traces are retained where supported by the evaluator.

---

## 9. Important reproducibility note

The PDF corpus and Chroma database are local generated/restricted data and are not committed to Git.

To reproduce the experiment in a new Codespace:

1. Restore the 18 authorised PDF files locally.
2. Verify the filenames and corpus inventory.
3. Rebuild `.chroma_great` using the repository's ingestion code.
4. Run the RAG sanity checks.
5. Run the frozen 48-question benchmark.

The uppercase `.PDF` extension must be handled explicitly during ingestion.

---

## 10. Current Git checkpoint

The repository should contain:

- Source code
- Frozen 48-question benchmark
- Semantic evaluation code
- Requirements/environment configuration
- Experiment documentation

The repository should NOT contain:

- API keys
- `.env` secrets
- Copyright-restricted PDF files
- `.chroma_great`
- Python cache files
- Temporary backup scripts

---

## 11. Research interpretation

The pilot indicates that retrieval grounding and multi-agent decomposition should be investigated separately.

The formal experiment should test whether:

1. RAG improves evidence-grounded factual performance over an ungrounded LLM;
2. multi-agent decomposition provides additional benefit over single-agent RAG;
3. the effect varies across retrofit knowledge domains and task types;
4. multi-agent synthesis introduces identifiable failure patterns.

No claim of universal multi-agent superiority should be made before the formal experiment is completed.