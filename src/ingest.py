from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None


def ingest_papers(papers_dir: str = "data/papers"):
    # Load PDFs
    documents = SimpleDirectoryReader(papers_dir).load_data()
    print(f"Loaded {len(documents)} documents")

    # Chunking
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")

    # Connect to Qdrant
    client = QdrantClient(host="qdrant", port=6333)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="research_papers"
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Build index
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context
    )

    print("Ingestion complete. Papers stored in Qdrant.")
    return index


if __name__ == "__main__":
    ingest_papers()
