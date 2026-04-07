import json
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query import query_papers

TEST_QUESTIONS = [
    "How do different papers approach reducing hallucination in RAG?",
    "How do the approaches to chunking differ across papers?",
    "What techniques reduce hallucination in LLMs without fine-tuning?",
    "How do reranking methods improve retrieval quality compared to basic similarity search?",
    "What are the main trade-offs between retrieval quality and response generation quality in RAG pipelines?"
]


def run_evaluation_suite():
    results = []

    print("\nRunning evaluation suite...\n")

    for question in TEST_QUESTIONS:
        print(f"Evaluating: {question[:60]}...")

        result = query_papers(question, evaluate=True)

        eval_data = result.get("evaluation", {})

        summary = {
            "question": question,
            "overall_confidence": eval_data.get("overall_confidence", "N/A"),
            "faithfulness": eval_data.get("faithfulness", {}).get("faithfulness_score", "N/A"),
            "sufficiency": eval_data.get("retrieval_sufficiency", {}).get("sufficiency_score", "N/A"),
            "source_coverage": eval_data.get("source_coverage", {}).get("coverage_score", "N/A"),
            "papers_used": eval_data.get("source_coverage", {}).get("unique_papers_used", 0),
            "recommendation": eval_data.get("recommendation", "N/A")
        }

        results.append(summary)

        print(
            f"→ Confidence: {summary['overall_confidence']} | "
            f"Faithful: {summary['faithfulness']} | "
            f"Sufficient: {summary['sufficiency']} | "
            f"Coverage: {summary['source_coverage']}"
        )

    # Save results
    with open("eval/evaluation_suite_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary ---")

    confidence_counts = {}
    for r in results:
        c = r["overall_confidence"]
        confidence_counts[c] = confidence_counts.get(c, 0) + 1

    print(f"Confidence distribution: {confidence_counts}")

    return results


if __name__ == "__main__":
    run_evaluation_suite()
