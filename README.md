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

> “Can I trust this answer?”

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

- Query rewriting improves retrieval precision  
- Source diversity prevents over-reliance on one paper  
- Source-aware prompting enforces cross-paper reasoning  
- LLM-as-judge evaluates faithfulness and sufficiency  
- Deterministic coverage avoids circular evaluation  

---

## Evaluation Results

- Confidence distribution: 1 HIGH, 4 MEDIUM, 0 LOW  
- Evaluator agreement: 87% (13/15)  

### Key Insight

Standard RAG systems cannot distinguish between **related** and **sufficient** context.

This system explicitly surfaces that gap:

> Relevance ≠ Sufficiency

---

## How to Run

1. Clone repo  
   git clone <your-repo-url>  
   cd research-intelligence-agent  

2. Add API key  
   Create a `.env` file in the root folder and add:  
   GROQ_API_KEY=your_key_here  

3. Start system  
   docker compose up  

4. Ingest data  
   docker compose run --rm api python src/ingest.py  

5. Open API  
   http://localhost:8000/docs  

---

## Sample Request

curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "How do different papers approach reducing hallucination in RAG?",
  "evaluate": true
}'

---

## Sample Response (trimmed)

{
  "answer": "...",
  "sources": ["chunking_1.pdf", "chunking_2.pdf"],
  "evaluation": {
    "overall_confidence": "MEDIUM",
    "faithfulness_score": "LOW",
    "coverage": "GOOD"
  }
}

---

## Tech Stack

FastAPI, Qdrant, LlamaIndex, HuggingFace, Groq, Docker

---

## Why This Project Matters

This project makes AI systems more honest about what they know vs don’t know.
