from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv
from src.query_rewriter import rewrite_query
from src.evaluator import evaluate_response
load_dotenv()

# Must match exactly what was used during ingestion
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None  # we handle LLM calls manually via Groq


def get_diverse_retriever(question: str, top_k_per_source: int = 2):
    client = QdrantClient(host="localhost", port=6333)
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

    # Retrieve more candidates than needed, then enforce diversity
    retriever = index.as_retriever(similarity_top_k=15)
    all_nodes = retriever.retrieve(question)

    # Enforce source diversity — max 2 chunks per paper
    seen_sources = {}
    diverse_nodes = []

    for node in all_nodes:
        source = node.metadata.get('file_name', 'unknown')
        count = seen_sources.get(source, 0)
        if count < top_k_per_source:
            diverse_nodes.append(node)
            seen_sources[source] = count + 1
        if len(diverse_nodes) >= 6:
            break

    return diverse_nodes


def query_papers(question: str, evaluate: bool = True) -> dict:
    rewritten_question = rewrite_query(question)

    # Retrieve diverse chunks using rewritten query
    nodes = get_diverse_retriever(rewritten_question)

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"PAPER: {node.metadata.get('file_name', 'unknown')}\n"
        f"Relevance: {round(node.score, 3)}\n"
        f"Content: {node.text}"
        for node in nodes
    ])

    # Send to LLM via Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a research assistant analyzing multiple AI/ML papers.

Your job is to answer the question by reasoning ACROSS the provided sources.

Instructions:
- Treat each PAPER section as a distinct source with potentially different positions
- If papers agree, say so explicitly
- If papers disagree or take different approaches, surface that difference clearly
- If a paper only mentions the topic indirectly, say that explicitly instead of overstating it
- If retrieved context is insufficient to answer confidently, say:
  "Retrieved context is insufficient — this question needs broader coverage"
- Never synthesize a smooth answer that hides disagreement between sources
- Always attribute claims to specific papers

Context:
{context}

Question: {question}

Answer (reason across sources explicitly):"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    sources = list(set([
        node.metadata.get('file_name', 'unknown')
        for node in nodes
    ]))

    result = {
        "question": question,
        "rewritten_query": rewritten_question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(nodes),
        "top_relevance_score": round(nodes[0].score, 3) if nodes else 0
    }

    # Run evaluation if requested
    if evaluate:
        evaluation = evaluate_response(
            question=question,
            answer=answer,
            context=context,
            nodes=nodes
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
