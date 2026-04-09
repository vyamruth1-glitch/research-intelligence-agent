# Research Intelligence Agent

A multi-source retrieval and reasoning system built over AI/ML research papers.

It answers complex cross-paper questions and evaluates whether its own 
answers are grounded, sufficient, and reliable.

---

## The Problem

Most LLM-based systems generate answers that sound confident but may 
not be grounded in actual sources.

When dealing with research papers, this becomes worse — answers often:

- flatten differences across papers into one smooth narrative
- over-generalize from a single highly-ranked source
- rely on context that is related but not actually sufficient to answer

There is no built-in way to know: **"Can I trust this answer?"**

This system is designed to answer that question explicitly.

---

## System Architecture

![Architecture](docs/architecture.png)

**Pipeline:**
User Query → Query Rewriter → Diverse Retriever (Qdrant) → 
Source-Aware Prompt → LLM Generator (Groq) → 
Evaluation Layer → Structured Response with Confidence Score

---

## Key Design Decisions

**Query rewriting**
Rewrites vague queries into retrieval-precise forms before 
hitting the vector DB. Includes guardrails to preserve 
technical terms like "RAG" from semantic drift.

**Source diversity enforcement**
Retrieves a larger candidate pool, then caps chunks per paper 
at 2. Forces the system to draw from multiple papers rather 
than dominating with one highly-similar source.

**Source-aware prompting**
Explicitly instructs the LLM to treat each chunk as a distinct 
source and surface disagreements rather than synthesising 
into one smooth answer.

**LLM-as-judge evaluation**
Faithfulness and sufficiency scored by a separate model call 
at temperature=0.0 for consistency. Validated against manual 
labels to understand where the judge is reliable.

**Deterministic source coverage**
Source diversity computed directly from retrieved node metadata — 
no LLM involved. Avoids circularity in the evaluation pipeline.

---

## Evaluation Results

| Metric | Result |
|--------|--------|
| Confidence distribution | 1 HIGH, 4 MEDIUM, 0 LOW |
| Evaluator vs manual labels | 87% agreement (13/15) |

### Core Design Insight

Standard RAG systems have no mechanism to distinguish between 
retrieving *related* content and retrieving *sufficient* content.

This system was specifically designed to surface that gap. When 
validated against manual labels, even the automated judge conflated 
the two in borderline cases — confirming how subtle this distinction 
is in practice.

> **Relevance ≠ Sufficiency**

The 13% disagreement in evaluator validation clusters entirely 
around borderline sufficiency cases — where context is topically 
relevant but evidence density is too low to fully answer the question.

---

## How to Run

**Prerequisites:** Docker, Docker Compose, Groq API key (free at console.groq.com)

```bash
# 1. Clone the repo
git clone https://github.com/vyamruth1-glitch/research-intelligence-agent
cd research-intelligence-agent

# 2. Set up environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 3. Start Qdrant and the API
docker-compose up

# 4. In a separate terminal, ingest papers
docker exec -it research-intelligence-agent-api-1 python src/ingest.py

# 5. API is live at http://localhost:8000
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
  "rewritten_query": "What techniques do research papers propose to reduce hallucination in retrieval-augmented generation systems?",
  "answer": "Papers take three main approaches to hallucination reduction in RAG. First, retrieval grounding — ensuring answers are explicitly tied to retrieved chunks rather than model priors. Second, post-generation verification — using a separate model pass to check claim-level faithfulness. Third, context sufficiency filtering — identifying when retrieved context is too weak to answer reliably and flagging this rather than generating a confident but unsupported answer.",
  "sources": ["hallucination_survey.pdf", "rag_evaluation.pdf", "chunking_strategies.pdf"],
  "retrieved_chunks": 6,
  "top_relevance_score": 0.74,
  "evaluation": {
    "overall_confidence": "MEDIUM",
    "faithfulness": {
      "faithfulness_score": "HIGH",
      "reasoning": "All major claims are supported by retrieved context"
    },
    "retrieval_sufficiency": {
      "sufficiency_score": "PARTIAL",
      "what_is_missing": "Specific implementation details for post-generation verification"
    },
    "source_coverage": {
      "coverage_score": "GOOD",
      "unique_papers_used": 3
    },
    "recommendation": "Answer is partially grounded. Core claims are supported but some details lack sufficient evidence."
  }
}
```

---

## Limitations and Future Directions

**Known limitations:**
- Evaluator overestimates sufficiency in borderline cases 
  (observed in 2/15 manual validation judgements)
- Cross-paper comparison surfaces sources but does not 
  deeply contrast positions
- System quality is bounded by corpus size — currently 10 papers

**What I would build next:**
- Cross-paper disagreement detection: identify when papers take 
  genuinely different positions and surface that explicitly
- Tighter sufficiency metric: require evidence density, 
  not just topical relevance
- Automated corpus expansion via ArXiv API on a schedule

---

## Stack

| Component | Tool | Reason |
|-----------|------|--------|
| Vector DB | Qdrant | Production-grade, hybrid search, local Docker |
| Embeddings | bge-small-en-v1.5 | Fast, local, strong on technical text |
| Orchestration | LlamaIndex | Clean RAG abstractions |
| LLM | Groq / Llama 3.1 | Fast inference, free tier |
| Reranking | Diversity-enforced retrieval | Avoids single-paper dominance |
| Evaluation | LLM-as-judge + deterministic | Validated against manual labels |
| Backend | FastAPI | Production-ready API layer |
| Containerisation | Docker Compose | One-command reproducibility |