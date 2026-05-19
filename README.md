# Research Intelligence Agent

A multi-source retrieval and reasoning system built over AI/ML research papers.

It answers complex cross-paper questions, evaluates whether its own answers are 
grounded and sufficient, and explicitly surfaces when papers disagree rather than 
smoothing contradictions into a unified narrative.

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
Deduplicated Numbered Context → LLM Generator (Groq) →
Disagreement Detector (independent LLM pass) →
Evaluation Layer → Structured Response with Confidence Score + Conflict Analysis

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
Retrieved papers are deduplicated by title and numbered `[Paper 1]`, 
`[Paper 2]`, etc. before being passed to the generator. The prompt 
requires every factual claim to be attributed to a specific label 
and prohibits using position words ("the first paper") that break 
when the same source appears multiple times. Contradictory positions 
must be presented as explicit conflicts, not merged.

**Cross-paper disagreement detection**
A dedicated second LLM pass runs on the raw retrieved context — 
independently of the generated answer — and identifies genuine 
cross-paper conflicts. Running it independently matters: the generation 
LLM can soften disagreements even with strict instructions, so the 
detector's findings are not contaminated by how the answer was phrased.

Disagreements are classified along two axes:

- **conflict_type**: `methodological` (different approaches to the same problem), `empirical` (conflicting results or measurements), `definitional` (incompatible definitions of a term)
- **severity**: `direct_contradiction` (mutually exclusive claims — both cannot be true), `methodological_difference` (different approaches that could coexist), `tension` (different priorities or trade-offs in practice)

A same-paper filter rejects false positives where both positions come 
from the same source (e.g. one paper reporting results across two 
different models). The JSON parser has multi-stage fallback recovery: 
markdown fence stripping, targeted repair for a common LLM-generated 
malformed brace pattern, and a brace-scan extraction for embedded JSON.

The result appears as `disagreement_analysis` — a top-level field in 
every response, always populated, not buried inside the evaluation block.

**LLM-as-judge evaluation**
Faithfulness and sufficiency scored by a separate model call 
at temperature=0.0 for consistency. Validated against manual 
labels to understand where the judge is reliable.

**Deterministic source coverage**
Source diversity computed directly from retrieved node metadata — 
no LLM involved. Avoids circularity in the evaluation pipeline.

---

## Corpus

61 papers ingested across six topics via the ArXiv API, producing ~2,700 chunks.

| Topic | Papers |
|-------|--------|
| RAG | 15 |
| LLM Evaluation | 13 |
| Agents | 8 |
| Embeddings | 13 |
| Hallucination | 6 |
| RAG Evaluation | 6 |

---

## Evaluation Results

| Metric | Result |
|--------|--------|
| Confidence distribution | 1 HIGH, 4 MEDIUM, 0 LOW |
| Evaluator vs manual labels | 87% agreement (13/15) |

### Core Design Insights

**Relevance ≠ Sufficiency**

Standard RAG systems have no mechanism to distinguish between 
retrieving *related* content and retrieving *sufficient* content.

This system was specifically designed to surface that gap. When 
validated against manual labels, even the automated judge conflated 
the two in borderline cases — confirming how subtle this distinction 
is in practice.

The 13% disagreement in evaluator validation clusters entirely 
around borderline sufficiency cases — where context is topically 
relevant but evidence density is too low to fully answer the question.

**Disagreement ≠ Different Wording**

Detecting genuine cross-paper disagreement requires distinguishing 
between papers contradicting each other and papers simply presenting 
different angles — most RAG systems flatten both into smooth summaries.

A paper that says "late chunking is computationally efficient" and a 
paper that says "chunk size should balance context and processing limits" 
are not contradicting each other — they are addressing different aspects 
of the same problem. Treating this as a `direct_contradiction` would 
be wrong; classifying it as a `methodological_difference` is accurate. 
The severity taxonomy (`direct_contradiction` / `methodological_difference` 
/ `tension`) exists specifically to encode this distinction in a 
machine-readable way.

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

This example demonstrates retrieval, query rewriting, disagreement detection,
evaluator scoring, and source-grounded answer generation across the live corpus.

### Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "How do different papers approach chunking strategies in RAG?",
  "evaluate": true
}'
```

### Response

```json
{
  "question": "How do different papers approach chunking strategies in RAG?",
  "rewritten_query": "What chunking strategies are employed in Retrieval-Augmented Generation (RAG) models as described in various research papers?",
  "answer": "According to [Paper 1], late chunking offers a more computationally efficient solution by leveraging the natural capabilities of embedding models, in contrast to contextual retrieval which incurs higher computational expenses. [Paper 2] takes a different view, arguing that chunks should be large enough to contain sufficient context for answering questions accurately, but not so large that they overwhelm the generator or exceed processing limits. Both [Paper 1] and [Paper 2] agree that chunking strategy is crucial for RAG performance, but they differ in their specific recommendations.",
  "sources": [
    "Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation",
    "Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation"
  ],
  "retrieved_chunks": 2,
  "top_relevance_score": 0.921,
  "disagreement_analysis": {
    "disagreements_found": true,
    "conflict_count": 1,
    "conflicts": [
      {
        "topic": "chunking strategy for RAG systems",
        "positions": [
          {
            "paper": "Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation",
            "position": "Late chunking offers a more computationally efficient solution by leveraging the natural capabilities of embedding models."
          },
          {
            "paper": "Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation",
            "position": "Chunks should be large enough to contain sufficient context for answering questions accurately, but not so large that they overwhelm the generator or exceed processing limits."
          }
        ],
        "conflict_type": "methodological",
        "severity": "methodological_difference"
      }
    ],
    "summary": "The two papers disagree on the optimal chunking strategy for RAG, with one advocating for late chunking for computational efficiency and the other recommending a balanced approach to avoid overwhelming the generator."
  },
  "evaluation": {
    "overall_confidence": "MEDIUM",
    "faithfulness": {
      "faithfulness_score": "MEDIUM",
      "supported_claims": [
        "Late chunking offers a more computationally efficient solution.",
        "Chunks should balance sufficient context against processing limits."
      ],
      "unsupported_claims": [],
      "reasoning": "Claims are grounded in the retrieved context and attributed to specific papers."
    },
    "retrieval_sufficiency": {
      "sufficiency_score": "SUFFICIENT",
      "what_is_present": "Two papers discussing chunking trade-offs with different priorities.",
      "what_is_missing": "Empirical benchmarks comparing the strategies head-to-head."
    },
    "source_coverage": {
      "coverage_score": "MODERATE",
      "unique_papers_used": 2
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
- ArXiv PDF rate limiting caps practical corpus size per run 
  (~60 papers reliably; `--resume` recovers partial failures)
- Disagreement detector relies on the LLM correctly identifying 
  cross-paper conflicts; may miss subtle theoretical disagreements 
  that require deep domain knowledge to recognise

**What I would build next:**
- Tighter sufficiency metric: require evidence density, 
  not just topical relevance
- Scheduled corpus refresh: re-run ArXiv ingestion on a cron 
  schedule to keep the corpus current with new publications
- Disagreement trending: track whether a conflict between two 
  papers is resolved by more recent work in the corpus

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
| Disagreement detection | Independent LLM pass on context | Not contaminated by answer generation |
| Evaluation | LLM-as-judge + deterministic | Validated against manual labels |
| Backend | FastAPI | Production-ready API layer |
| Containerisation | Docker Compose | One-command reproducibility |
