from groq import Groq
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _parse_json_response(raw: str) -> dict | None:
    """
    Extract and parse a JSON object from an LLM response.
    Handles markdown fences and recovers when the JSON is embedded in surrounding text
    by scanning for the outermost balanced brace pair.
    """
    # Direct parse — fast path for clean responses
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    # Targeted repair: the LLM sometimes omits the closing } on a positions
    # array element, producing `"position": "value", {` instead of
    # `"position": "value"}, {`. Fix before attempting the brace-scan.
    repaired = re.sub(
        r'("position"\s*:\s*"(?:[^"\\]|\\.)*")\s*,(\s*\{)',
        r'\1},\2',
        raw,
    )
    if repaired != raw:
        try:
            return json.loads(repaired.strip())
        except json.JSONDecodeError:
            pass

    # Brace-scan: find the outermost { ... } pair and try to parse that slice.
    # Handles cases where the LLM adds preamble or trailing commentary.
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


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
    sources = [node.metadata.get('title', node.metadata.get('file_name', 'unknown')) for node in nodes]
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


def detect_disagreements(question: str, context: str, nodes: list) -> dict:
    """
    Identifies genuine cross-paper disagreements from the retrieved context.

    Runs as an independent pass on the raw context — not on the generated answer —
    so its findings are not contaminated by how the LLM chose to synthesise things.

    Only called when 2+ distinct papers are in the retrieved set; a single-source
    context cannot produce a disagreement by definition.
    """
    unique_papers = set(
        node.metadata.get('title', node.metadata.get('file_name', 'unknown'))
        for node in nodes
    )
    if len(unique_papers) < 2:
        return {
            "disagreements_found": False,
            "conflict_count": 0,
            "conflicts": [],
            "summary": "Only one paper retrieved — cross-paper comparison not possible."
        }

    prompt = f"""You are analyzing whether research papers in the provided context take genuinely different positions.

GENUINE DISAGREEMENT requires ALL of the following:
- Two papers making claims about the SAME specific aspect (same metric, technique, or phenomenon)
- Their claims are contradictory or mutually incompatible — not just different in scope
- The difference is grounded in what the papers explicitly say, not inferred

NOT a disagreement (do NOT report these):
- Papers discussing different aspects of the same broad topic
- Same concept expressed with different terminology
- One paper is more detailed; the other is more general
- A newer paper extends or builds on an older one without contradicting it

For each genuine disagreement found, identify:
- topic: the precise aspect they disagree on (specific, not broad)
- positions: each paper's exact position, attributed to the PAPER label from the context
- conflict_type: "methodological" (different approaches to same problem), "empirical" (conflicting results or findings), or "definitional" (incompatible definitions)
- severity: "direct_contradiction" (mutually exclusive claims), "tension" (incompatible emphasis or priority), or "complementary_different" (genuinely different but not mutually exclusive)

Only report disagreements grounded in the provided context. Do not infer or extrapolate.

Context:
{context}

Question: {question}

Respond in this exact JSON format only:
{{
  "disagreements_found": true,
  "conflict_count": 1,
  "conflicts": [
    {{
      "topic": "specific aspect they disagree on",
      "positions": [
        {{"paper": "exact PAPER label from context", "position": "their specific claim"}},
        {{"paper": "exact PAPER label from context", "position": "their specific claim"}}
      ],
      "conflict_type": "methodological",
      "severity": "direct_contradiction"
    }}
  ],
  "summary": "one sentence describing the key conflict(s), or confirming papers are aligned"
}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
    result = _parse_json_response(raw)
    if result is None:
        return {
            "disagreements_found": False,
            "conflict_count": 0,
            "conflicts": [],
            "summary": "PARSE_ERROR",
            "raw_response": raw
        }

    # Remove conflicts where every position cites the same paper — those are
    # within-paper comparisons (e.g. one paper reporting results on two models),
    # not genuine cross-paper disagreements.
    valid = [
        c for c in result.get("conflicts", [])
        if len({p.get("paper", "") for p in c.get("positions", [])}) >= 2
    ]
    result["conflicts"] = valid
    result["conflict_count"] = len(valid)
    result["disagreements_found"] = len(valid) > 0
    if not result["disagreements_found"] and not result.get("summary", "").startswith("PARSE"):
        result["summary"] = "No genuine cross-paper disagreements found — papers address different aspects or are broadly aligned."
    return result


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
