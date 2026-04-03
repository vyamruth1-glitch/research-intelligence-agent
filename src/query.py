from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Must match exactly what was used during ingestion
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None  # we handle LLM calls manually via Groq

def get_retriever():
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
    return index.as_retriever(similarity_top_k=5)


def query_papers(question: str) -> dict:
    retriever = get_retriever()

    # Retrieve relevant chunks
    nodes = retriever.retrieve(question)

    # Build context from retrieved chunks
    context = "\n\n".join([
        f"Source: {node.metadata.get('file_name', 'unknown')}\n"
        f"Relevance Score: {round(node.score, 3)}\n"
        f"{node.text}"
        for node in nodes
    ])

    # Send to LLM via Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a research assistant helping analyze AI/ML papers.

Answer the question using ONLY the provided context.
- If the context contains enough information, answer clearly and cite sources.
- If the context is insufficient, say explicitly: 
  "The retrieved context does not contain enough information to answer this."
- Never invent information not present in the context.

Context:
{context}

Question: {question}

Answer:"""

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

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(nodes),
        "top_relevance_score": round(nodes[0].score, 3) if nodes else 0
    }


if __name__ == "__main__":
    question = "What are the main trade-offs between retrieval quality and response generation quality in RAG pipelines?"
    result = query_papers(question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Chunks retrieved: {result['retrieved_chunks']}")
    print(f"Top relevance score: {result['top_relevance_score']}")
