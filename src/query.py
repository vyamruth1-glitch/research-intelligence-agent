from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv
from src.query_rewriter import rewrite_query
from src.evaluator import evaluate_response, detect_disagreements
load_dotenv()

# Must match exactly what was used during ingestion
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None  # we handle LLM calls manually via Groq


def build_filters(topic=None, year_from=None, year_to=None):
    # Translate optional request params into a LlamaIndex MetadataFilters object.
    # LlamaIndex passes this to QdrantVectorStore, which translates it into
    # Qdrant's native filter format (MatchValue for equality, Range for comparisons).
    # The filter runs BEFORE the ANN vector search — Qdrant restricts the candidate
    # set to matching points first, then finds the most similar among those.
    # Returning None (no conditions) tells LlamaIndex to search the full collection.
    conditions = []
    if topic:
        conditions.append(MetadataFilter(key="topic", value=topic, operator=FilterOperator.EQ))
    if year_from:
        conditions.append(MetadataFilter(key="year", value=year_from, operator=FilterOperator.GTE))
    if year_to:
        conditions.append(MetadataFilter(key="year", value=year_to, operator=FilterOperator.LTE))
    return MetadataFilters(filters=conditions) if conditions else None


def get_diverse_retriever(question: str, top_k_per_source: int = 2,
                           topic=None, year_from=None, year_to=None):
    client = QdrantClient(host="qdrant", port=6333)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="research_papers"
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

    # Retrieve more candidates than needed, then enforce diversity.
    # filters=None means no payload restriction — full collection search.
    filters = build_filters(topic=topic, year_from=year_from, year_to=year_to)
    retriever = index.as_retriever(similarity_top_k=15, filters=filters)
    all_nodes = retriever.retrieve(question)

    # Enforce source diversity — max 2 chunks per paper
    seen_sources = {}
    diverse_nodes = []

    for node in all_nodes:
        source = node.metadata.get('title', node.metadata.get('file_name', 'unknown'))
        count = seen_sources.get(source, 0)
        if count < top_k_per_source:
            diverse_nodes.append(node)
            seen_sources[source] = count + 1
        if len(diverse_nodes) >= 6:
            break

    return diverse_nodes

def query_papers(question: str, evaluate: bool = True,
                 topic=None, year_from=None, year_to=None) -> dict:
    rewritten_question = rewrite_query(question)

    # Retrieve diverse chunks using rewritten query, with optional payload filters
    nodes = get_diverse_retriever(rewritten_question, topic=topic,
                                   year_from=year_from, year_to=year_to)

    # Deduplicate nodes by title so the same paper never appears under two
    # different positional labels in the context block.
    seen_titles: set[str] = set()
    unique_nodes = []
    for node in nodes:
        title = node.metadata.get('title', node.metadata.get('file_name', 'unknown'))
        if title not in seen_titles:
            seen_titles.add(title)
            unique_nodes.append(node)

    # Number each unique paper so the prompt can reference stable labels
    # instead of relying on position words like "the first paper."
    numbered_context = "\n\n---\n\n".join([
        f"[Paper {i + 1}] {node.metadata.get('title', node.metadata.get('file_name', 'unknown'))}\n"
        f"Relevance: {round(node.score, 3)}\n"
        f"Content: {node.text}"
        for i, node in enumerate(unique_nodes)
    ])

    # Also build the plain context string (used by evaluators) from the same
    # deduplicated set so evaluation and generation see the same sources.
    context = "\n\n---\n\n".join([
        f"PAPER: {node.metadata.get('title', node.metadata.get('file_name', 'unknown'))}\n"
        f"Relevance: {round(node.score, 3)}\n"
        f"Content: {node.text}"
        for node in unique_nodes
    ])

    # Send to LLM via Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a research assistant whose primary job is to surface how multiple AI/ML papers agree AND disagree on a question.

The context below lists each paper with a stable label: [Paper 1], [Paper 2], etc.

Rules — follow all of them:
1. Refer to papers by their assigned label ([Paper 1], [Paper 2], ...) — never by position words like "the first paper" or "the second paper."
2. Every factual claim must be attributed: write "According to [Paper 1]..." or "[Paper 2] argues that..."
3. Do not attribute the same claim to two different labels if they refer to the same source — each label is a distinct paper.
4. When two papers take different positions on the same aspect, present the conflict explicitly:
   "[Paper A] argues [position]. [Paper B] takes a different view, arguing [position]."
   Never merge contradictory positions into a single smoothed claim.
5. When papers genuinely agree, say so: "Both [Paper 1] and [Paper 2] agree that..."
6. If a paper only touches a topic indirectly, say so rather than overstating its position.
7. If the retrieved context is too thin to answer reliably, say:
   "Retrieved context is insufficient — this question needs broader coverage."
8. Answer in plain prose only — no JSON, no bullet lists, no headers.

Context:
{numbered_context}

Question: {question}

Answer (use [Paper N] labels, attribute every claim, surface every disagreement explicitly):"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    sources = [
        node.metadata.get('title', node.metadata.get('file_name', 'unknown'))
        for node in unique_nodes
    ]

    # Include only the filters that were actually set so the response is clean
    # when no filters are used (the common case)
    active_filters = {k: v for k, v in
                      {"topic": topic, "year_from": year_from, "year_to": year_to}.items()
                      if v is not None}

    result = {
        "question": question,
        "rewritten_query": rewritten_question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(unique_nodes),
        "top_relevance_score": round(unique_nodes[0].score, 3) if unique_nodes else 0,
    }
    if active_filters:
        result["filters_applied"] = active_filters

    # Disagreement detection always runs when 2+ papers are in the retrieved set.
    # It's a content feature of the response, not an optional eval layer — callers
    # need it to understand whether the answer reflects genuine consensus or conflict.
    result["disagreement_analysis"] = detect_disagreements(
        question=question,
        context=context,
        nodes=unique_nodes
    )

    if evaluate:
        evaluation = evaluate_response(
            question=question,
            answer=answer,
            context=context,
            nodes=unique_nodes
        )
        result["evaluation"] = evaluation

    return result


if __name__ == "__main__":
    # Test with one real question first
    question = "How do different papers approach reducing hallucination in RAG?"
    result = query_papers(question, evaluate=True)

    print(f"\nOriginal Question: {result['question']}")
    print(f"Rewritten Query: {result['rewritten_query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Chunks retrieved: {result['retrieved_chunks']}")
    print(f"Top relevance score: {result['top_relevance_score']}")

    if "evaluation" in result:
        print("\n--- Evaluation ---")
        print(f"Overall Confidence: {result['evaluation']['overall_confidence']}")
        print(f"Faithfulness: {result['evaluation']['faithfulness'].get('faithfulness_score', 'N/A')}")
        print(f"Sufficiency: {result['evaluation']['retrieval_sufficiency'].get('sufficiency_score', 'N/A')}")
        print(f"Source Coverage: {result['evaluation']['source_coverage'].get('coverage_score', 'N/A')}")
        print(f"Recommendation: {result['evaluation']['recommendation']}")
