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

**ArXiv-based corpus expansion**
Papers are fetched automatically via the ArXiv API across six topic 
queries (RAG, LLM evaluation, agents, embeddings, hallucination, RAG 
evaluation). Each paper is chunked and stored with structured metadata 
— arxiv_id, title, authors, year, topic, abstract — directly in the 
Qdrant point payload. This makes metadata available at retrieval time 
with no separate lookup. Deduplication is by canonical ArXiv ID 
(version suffix stripped) so v1 and v2 of the same paper are treated 
as one. The ingestion script supports `--resume` to continue after 
failures without re-downloading already-ingested papers.

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

## Corpus

49 papers ingested across six topics via the ArXiv API, producing ~2,155 chunks.

| Topic | Papers |
|-------|--------|
| RAG | 14 |
| LLM Evaluation | 13 |
| Agents | 6 |
| Embeddings | 6 |
| Hallucination | 6 |
| RAG Evaluation | 4 |

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

# 4. In a separate terminal, ingest papers from ArXiv
docker exec -it research-intelligence-agent-api-1 pip install "arxiv>=2.1.0"
docker exec -it research-intelligence-agent-api-1 python -m src.arxiv_ingest

# Re-run after partial failures without losing already-ingested papers:
docker exec -it research-intelligence-agent-api-1 python -m src.arxiv_ingest --resume

# 5. API is live at http://localhost:8000
```

---

## Sample API Usage

This example demonstrates retrieval, query rewriting, evaluator scoring, 
and source-grounded answer generation across the live 49-paper corpus.

### Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "How do different evaluation frameworks measure RAG faithfulness, and where do they disagree?",
  "evaluate": true
}'
```

### Response

```json
{
  "question": "How do different evaluation frameworks measure RAG faithfulness, and where do they disagree?",
  "rewritten_query": "What evaluation frameworks for Retrieval-Augmented Generation (RAG) models assess faithfulness, and how do their metrics and methodologies diverge?",
  "answer": "Ragas measures faithfulness by checking whether claims in the answer can be inferred from retrieved context using an LLM. VERA focuses on factual consistency between generated text and grounded sources. FAIR-RAG uses a structured, gap-aware iterative refinement loop — deconstructing the query into a checklist and auditing evidence against it, requiring no model fine-tuning. SELF-RAG fine-tunes an LLM to emit inline reflection tokens that critique relevance and factual support during generation. The key disagreement is methodological: Ragas and VERA treat faithfulness as a post-hoc metric, while FAIR-RAG and SELF-RAG make it part of the generation process itself. SELF-RAG's reliance on fine-tuning is a further point of divergence from the others.",
  "sources": [
    "VERA: Validation and Evaluation of Retrieval-Augmented Systems",
    "FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation",
    "Ragas: Automated Evaluation of Retrieval Augmented Generation",
    "Retrieval-Augmented Generation in Industry: An Interview Study"
  ],
  "retrieved_chunks": 6,
  "top_relevance_score": 0.877,
  "evaluation": {
    "overall_confidence": "MEDIUM",
    "faithfulness": {
      "faithfulness_score": "LOW",
      "supported_claims": [
        "Ragas measures faithfulness by estimating whether claims can be inferred from context."
      ],
      "unsupported_claims": [
        "VERA explicitly mentions faithfulness as a separate metric.",
        "FAIR-RAG and SELF-RAG disagree on the approach to measuring faithfulness."
      ],
      "reasoning": "The answer does not accurately represent all disagreements between sources. SELF-RAG's approach is not accurately framed as disagreeing with the others, and VERA does not explicitly define faithfulness as a standalone metric."
    },
    "retrieval_sufficiency": {
      "sufficiency_score": "SUFFICIENT",
      "what_is_present": "Multiple evaluation frameworks discussed with their core mechanisms and limitations.",
      "what_is_missing": "A direct side-by-side comparison of frameworks is not explicitly stated in any single source."
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

## Limitations and Future Directions

**Known limitations:**
- Evaluator overestimates sufficiency in borderline cases 
  (observed in 2/15 manual validation judgements)
- Cross-paper comparison surfaces sources but does not 
  deeply contrast positions
- ArXiv PDF rate limiting caps practical corpus size per run 
  (~50 papers reliably; `--resume` recovers partial failures)

**What I would build next:**
- Cross-paper disagreement detection: identify when papers take 
  genuinely different positions and surface that explicitly
- Tighter sufficiency metric: require evidence density, 
  not just topical relevance
- Scheduled corpus refresh: re-run ArXiv ingestion on a cron 
  schedule to keep the corpus current with new publications

---

## Stack

| Component | Tool | Reason |
|-----------|------|--------|
| Vector DB | Qdrant | Production-grade, payload filtering, local Docker |
| Embeddings | bge-small-en-v1.5 | Fast, local, strong on technical text |
| Orchestration | LlamaIndex | Clean RAG abstractions |
| LLM | Groq / Llama 3.1 | Fast inference, free tier |
| Corpus ingestion | ArXiv API + arxiv library | Automated, metadata-rich, deduplicated |
| Reranking | Diversity-enforced retrieval | Avoids single-paper dominance |
| Evaluation | LLM-as-judge + deterministic | Validated against manual labels |
| Backend | FastAPI | Production-ready API layer |
| Containerisation | Docker Compose | One-command reproducibility |
