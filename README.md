# Research Intelligence Agent

A multi-source retrieval and reasoning system built over AI/ML research papers.

It answers complex cross-paper questions and evaluates whether its own answers are grounded, sufficient, and reliable.

---

## The Problem

Most LLM-based systems generate answers that sound confident but may not be grounded in actual sources.

When dealing with research papers, this becomes worse — answers often:
- flatten differences across papers
- over-generalize from a single source
- rely on context that is related but not actually sufficient

There is no built-in way to know:

> *“Can I trust this answer?”*

---

## System Architecture

![Architecture](docs/architecture.png)

---

## Key Design Decisions

- **Query rewriting**  
  Improves retrieval precision by making implicit information needs explicit.

- **Source diversity enforcement**  
  Limits number of chunks per paper to avoid over-reliance on a single source.

- **Source-aware prompting**  
  Forces the model to reason across papers instead of merging them into one narrative.

- **LLM-as-judge evaluation**  
  Measures faithfulness and sufficiency of answers using a separate evaluation step.

- **Deterministic source coverage**  
  Avoids circularity by computing source diversity without using an LLM.

---

## Evaluation Results

- Confidence distribution: **1 HIGH, 4 MEDIUM, 0 LOW**
- Evaluator agreement with manual labels: **87% (13/15)**

### Key Insight

The evaluator shows a consistent bias:

> It tends to treat *related context* as *sufficient context*, even when the retrieved evidence is not strong enough to fully answer the question.

This highlights a critical distinction:

> **Relevance ≠ Sufficiency**

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
