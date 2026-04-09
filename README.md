# Research Intelligence Agent

A multi-source retrieval and reasoning system for AI/ML research papers.

It answers complex cross-paper questions and evaluates whether its own answer is sufficiently grounded in the retrieved evidence.

---

## The Problem

Most LLM-based systems generate answers that sound convincing even when the retrieved evidence is incomplete or only partially relevant.

When dealing with research papers, this becomes especially risky because systems may:

* flatten differences across papers
* over-generalize from a single source
* rely on context that is related but not actually sufficient

There is usually no built-in way to know:

> *“Can I trust this answer?”*

This project was built to address that gap.

---

## System Architecture

![Architecture](docs/architecture.png)

---

## What the System Does

Given a research question, the system:

1. rewrites the query for better retrieval
2. retrieves evidence across multiple papers
3. enforces source diversity across retrieved chunks
4. generates an answer grounded in retrieved context
5. evaluates the answer for:

   * faithfulness
   * retrieval sufficiency
   * source coverage
6. returns both the answer and a structured reliability signal

---

## Key Design Decisions

* **Query rewriting**
  Improves retrieval precision by making implicit intent more explicit.

* **Source diversity enforcement**
  Limits chunk dominance from any one paper to encourage cross-paper reasoning.

* **Source-aware prompting**
  Encourages the model to compare, contrast, and synthesize across papers rather than summarize one source.

* **LLM-as-judge evaluation**
  Measures faithfulness and sufficiency of answers beyond retrieval relevance alone.

* **Deterministic source coverage**
  Avoids circularity by computing paper diversity directly from metadata.

---

## Evaluation Results

* Confidence distribution: **1 HIGH, 4 MEDIUM, 0 LOW**
* Evaluator agreement with manual labels: **87% (13/15)**

### Core Design Insight

Standard RAG systems have no mechanism to distinguish between retrieving *related* content and retrieving *sufficient* content.

This system’s sufficiency evaluator was specifically designed to surface that gap. When validated against manual labels, even the automated judge conflated the two in borderline cases — which confirms how subtle this distinction is in practice.

> **Relevance ≠ Sufficiency**

---

## How to Run

1. Clone the repository

```bash
git clone <your-repo-url>
cd research-intelligence-agent
```

2. Create a `.env` file and add your Groq API key

```env
GROQ_API_KEY=your_key_here
```

3. Start Qdrant and the API

```bash
docker compose up
```

4. In a separate terminal, ingest the papers

```bash
docker compose run --rm api python src/ingest.py
```

5. Open the API docs

```text
http://localhost:8000/docs
```

---

## Sample API Usage

### Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "How do different papers approach reducing hallucination in RAG?",
  "evaluate": true
}'
```

### Response

```json
{
  "question": "How do different papers approach reducing hallucination in RAG?",
  "rewritten_query": "What techniques and strategies do various research papers employ to mitigate hallucination in Retrieval-Augmented Generative (RAG) models?",
  "answer": "Based on the provided sources, different papers reduce hallucination in RAG through better retrieval structure, finer-grained chunking, and graph-based retrieval approaches. One paper frames RAG as a way to reduce hallucination by grounding responses in external evidence, while another suggests that GraphRAG can improve factual reasoning by retrieving more structured and relevant context. The system also correctly surfaced that the retrieved evidence was only partially sufficient for a fully comprehensive answer.",
  "sources": [
    "chunking_1.pdf",
    "hallucination_2.pdf",
    "chunking_2.pdf",
    "hallucination_1.pdf"
  ],
  "retrieved_chunks": 6,
  "top_relevance_score": 0.825,
  "evaluation": {
    "overall_confidence": "MEDIUM",
    "faithfulness": {
      "faithfulness_score": "LOW"
    },
    "retrieval_sufficiency": {
      "sufficiency_score": "SUFFICIENT"
    },
    "source_coverage": {
      "coverage_score": "GOOD",
      "unique_papers_used": 4
    },
    "recommendation": "Answer is partially grounded. Treat with moderate caution."
  }
}
```

---

## Tech Stack

* FastAPI
* Qdrant
* LlamaIndex
* HuggingFace Embeddings
* Groq API
* Docker Compose

---

## Why This Project Matters

This project is about making AI systems more honest about what they do and do not know.

That makes them more useful for research, analysis, and decision-making.