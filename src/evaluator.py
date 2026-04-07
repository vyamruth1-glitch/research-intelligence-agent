from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluate_faithfulness(question: str, answer: str, context: str) -> dict:
    """
    Checks whether claims in the answer are grounded in retrieved context.
    Returns a score and explanation.
    """
    prompt = f"""You are an evaluation assistant checking whether an AI answer 
is faithful to its source context.

Task: For each major claim in the answer, check whether it is:
- SUPPORTED: clearly present in the context
- PARTIALLY SUPPORTED: implied but not explicitly stated
- UNSUPPORTED: not present in the context at all

Then give an overall faithfulness score:
- HIGH: all major claims are supported
- MEDIUM: most claims supported, minor gaps
- LOW: significant claims have no grounding in context

Respond in this exact JSON format:
{{
  "faithfulness_score": "HIGH/MEDIUM/LOW",
  "supported_claims": ["claim 1", "claim 2"],
  "unsupported_claims": ["claim 1"],
  "reasoning": "brief explanation"
}}

Context:
{context}

Question: {question}

Answer: {answer}

Evaluation (JSON only):"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()

    try:
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "faithfulness_score": "PARSE_ERROR",
            "raw_response": raw
        }


def evaluate_retrieval_sufficiency(question: str, context: str) -> dict:
    """
    Checks whether retrieved context is actually sufficient to answer
    the question — independent of what the LLM said.
    """
    prompt = f"""You are evaluating whether retrieved research paper excerpts 
contain sufficient information to answer a question.

This is independent of any generated answer — just evaluate the context itself.

Score as:
- SUFFICIENT: context clearly contains what's needed to answer
- PARTIAL: context is related but missing key aspects
- INSUFFICIENT: context is too weak, off-topic, or shallow to answer reliably

Respond in this exact JSON format:
{{
  "sufficiency_score": "SUFFICIENT/PARTIAL/INSUFFICIENT",
  "what_is_present": "what relevant information exists in context",
  "what_is_missing": "what would be needed for a complete answer",
  "reasoning": "brief explanation"
}}

Question: {question}

Retrieved Context:
{context}

Evaluation (JSON only):"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "sufficiency_score": "PARSE_ERROR",
            "raw_response": raw
        }


def evaluate_source_coverage(nodes: list) -> dict:
    """
    Measures whether the answer drew from multiple papers.
    Does not need an LLM — computed directly from retrieved nodes.
    """
    sources = [node.metadata.get('file_name', 'unknown') for node in nodes]
    unique_sources = list(set(sources))

    if len(unique_sources) >= 3:
        coverage = "GOOD"
    elif len(unique_sources) == 2:
        coverage = "MODERATE"
    else:
        coverage = "WEAK"

    return {
        "coverage_score": coverage,
        "unique_papers_used": len(unique_sources),
        "papers": unique_sources
    }


def _get_recommendation(confidence: str, sufficiency: str) -> str:
    if confidence == "HIGH":
        return "Answer is well-grounded. Safe to use."
    elif sufficiency == "INSUFFICIENT":
        return "Retrieval is weak for this question. Consider rephrasing or expanding paper set."
    elif confidence == "MEDIUM":
        return "Answer is partially grounded. Treat with moderate caution."
    else:
        return "Low confidence. Answer may contain unsupported claims. Verify manually."


def evaluate_response(question: str, answer: str, context: str, nodes: list) -> dict:
    """
    Full evaluation pipeline — runs all three evaluators and
    produces a combined confidence assessment.
    """
    faithfulness = evaluate_faithfulness(question, answer, context)
    sufficiency = evaluate_retrieval_sufficiency(question, context)
    coverage = evaluate_source_coverage(nodes)

    # Derive overall confidence
    faith_score = faithfulness.get("faithfulness_score", "LOW")
    suff_score = sufficiency.get("sufficiency_score", "INSUFFICIENT")
    cov_score = coverage.get("coverage_score", "WEAK")

    score_map = {
        "HIGH": 3, "MEDIUM": 2, "LOW": 1,
        "SUFFICIENT": 3, "PARTIAL": 2, "INSUFFICIENT": 1,
        "GOOD": 3, "MODERATE": 2, "WEAK": 1,
        "PARSE_ERROR": 0
    }

    total = (
        score_map.get(faith_score, 0) +
        score_map.get(suff_score, 0) +
        score_map.get(cov_score, 0)
    )

    if total >= 8:
        confidence = "HIGH"
    elif total >= 5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "overall_confidence": confidence,
        "faithfulness": faithfulness,
        "retrieval_sufficiency": sufficiency,
        "source_coverage": coverage,
        "recommendation": _get_recommendation(confidence, suff_score)
    }
