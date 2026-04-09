# Research Intelligence Agent

A multi-source retrieval and reasoning system built over AI/ML research papers.

It answers complex cross-paper questions and evaluates whether its own answer is sufficiently grounded in the retrieved evidence.

---

## The Problem

Most LLM-based systems generate answers that sound convincing even when the retrieved evidence is incomplete or only partially relevant.

When dealing with research papers, this becomes especially risky because systems may:

- flatten differences across papers
- over-generalize from a single source
- rely on context that is related but not actually sufficient

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
   - faithfulness  
   - retrieval sufficiency  
   - source coverage  
6. returns both the answer and a structured reliability signal  

---

## Key Design Decisions

- **Query rewriting**  
  Improves retrieval precision by making implicit intent more explicit.

- **Source diversity enforcement**  
  Limits chunk dominance from any one paper to encourage cross-paper reasoning.

- **Source-aware prompting**  
  Encourages the model to compare, contrast, and synthesize across papers rather than summarize one source.

- **LLM-as-judge evaluation**  
  Measures faithfulness and sufficiency of answers beyond retrieval relevance alone.

- **Deterministic source coverage**  
  Avoids circularity by computing paper diversity directly from metadata.

---

## Evaluation Results

- Confidence distribution: **1 HIGH, 4 MEDIUM, 0 LOW**
- Evaluator agreement with manual labels: **87% (13/15)**

### Key Insight

Standard RAG systems have no mechanism to distinguish between retrieving **related content** and retrieving **sufficient content**.

This system was specifically designed to surface that gap through a dedicated sufficiency evaluator.

During validation, we found that even the evaluator itself sometimes conflated the two in borderline cases — which reinforced the central insight behind the project:

> **Relevance ≠ Sufficiency**

That is not just an implementation detail. It is the main design idea behind the system.

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd research-intelligence-agent
