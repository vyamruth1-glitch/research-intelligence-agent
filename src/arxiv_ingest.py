"""
ArXiv-based corpus builder for the Research Intelligence Agent.

Replaces the manual PDF drop-in pipeline (ingest.py) with automated fetching
from ArXiv across six research topics, consistent metadata tagging, and
deduplication by canonical ArXiv ID.

Usage:
  python -m src.arxiv_ingest           # full run — wipes collection first
  python -m src.arxiv_ingest --resume  # adds to existing collection; skips already-ingested papers
"""

import sys
import tempfile
import time

import arxiv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Must match ingest.py exactly — both pipelines write to the same Qdrant
# collection, so vectors must be produced by the same model to be comparable
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

COLLECTION_NAME = "research_papers"
MAX_PER_TOPIC = 20          # 20 × 6 topics = 120 candidates; dedup reduces this slightly
PDF_DOWNLOAD_DELAY = 5      # seconds between PDF downloads — separate from the API query
                            # delay; ArXiv's PDF server rate-limits bulk fetches independently

# Queries are deliberately specific — broader terms like "RAG" alone surface
# tangentially related papers; adding "knowledge grounding" or "evaluation metrics"
# anchors results to papers that actually engage with the concept
TOPICS = {
    "RAG":           "retrieval augmented generation knowledge grounding",
    "LLM_eval":      "large language model evaluation benchmark",
    "agents":        "LLM autonomous agents tool use planning",
    "embeddings":    "sentence embeddings semantic similarity dense retrieval",
    "hallucination": "hallucination detection mitigation language models",
    "RAG_eval":      "retrieval augmented generation evaluation metrics",
}


def wipe_collection(client: QdrantClient) -> None:
    # Delete before re-ingesting so every chunk in the collection shares the
    # same metadata schema (arxiv_id, title, authors, year, topic, abstract).
    # A partial wipe or soft-delete would leave schema-mismatched orphan points
    # that silently fail payload filters.
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[WIPE] Deleted '{COLLECTION_NAME}'")
    else:
        print(f"[WIPE] '{COLLECTION_NAME}' did not exist — nothing to delete")


def load_existing_arxiv_ids(client: QdrantClient) -> set[str]:
    # Scroll through every point in the collection and collect known arxiv_ids.
    # Used in --resume mode to pre-populate seen_ids so already-ingested papers
    # are skipped without querying Qdrant on every paper individually.
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        return set()

    ids: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["arxiv_id"],
            limit=256,
            offset=offset,
        )
        for record in records:
            aid = record.payload.get("arxiv_id")
            if aid:
                ids.add(aid)
        if offset is None:
            break
    return ids


def extract_arxiv_id(entry_id: str) -> str:
    # entry_id is a full URL: https://arxiv.org/abs/2310.11511v2
    # Strip the version suffix so v1 and v2 of the same paper are treated as
    # one entity — we want one canonical copy per paper, not one per revision
    return entry_id.split("/")[-1].split("v")[0]


def chunk_paper(paper: arxiv.Result, topic: str) -> list:
    """
    Download the PDF, chunk it, and attach structured metadata to every node.

    Why metadata lives in the Qdrant payload (not a separate store):
    LlamaIndex's QdrantVectorStore serializes node.metadata into the point
    payload. On retrieval, it deserializes it back into node.metadata. This
    means title, authors, year, and topic are automatically available at
    query time — zero extra lookups, no join across stores.

    Why abstract and authors are excluded from embedding:
    The embedding should represent the semantic content of the chunk text.
    Including metadata prose in the embedded text would dilute that signal
    and produce vectors that are partly anchored to author names or abstract
    phrasing rather than the paper's actual technical content.

    Why chunk_size=512 with overlap=50:
    Matches ingest.py exactly. Existing eval baselines were measured against
    this chunking strategy; changing it here would make the two ingestion
    paths produce non-comparable similarity scores in Qdrant.
    """
    arxiv_id = extract_arxiv_id(paper.entry_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Filename becomes the file_name metadata that LlamaIndex auto-attaches.
        # Using arxiv_id keeps it stable, unique, and human-readable in logs.
        # load_data() reads the full text into memory before the temp dir is removed.
        paper.download_pdf(dirpath=tmpdir, filename=f"{arxiv_id}.pdf")
        docs = SimpleDirectoryReader(tmpdir).load_data()

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    metadata = {
        "arxiv_id": arxiv_id,
        "title":    paper.title.replace("\n", " "),
        # Cap at 5 authors — papers with 20+ authors would bloat the payload
        "authors":  ", ".join(a.name for a in paper.authors[:5]),
        "year":     paper.published.year,
        # First-match topic: if this paper also appears in another topic's results,
        # the dedup check prevents re-ingestion, so this label is stable
        "topic":    topic,
        # Truncated to 500 chars — enough for display context without payload bloat
        "abstract": paper.summary.replace("\n", " ")[:500],
    }

    for node in nodes:
        node.metadata.update(metadata)
        node.excluded_embed_metadata_keys.extend(["abstract", "authors"])

    return nodes


def run_arxiv_ingest(resume: bool = False) -> None:
    client = QdrantClient(host="qdrant", port=6333)

    if resume:
        # Load arxiv_ids already in Qdrant so we skip them without re-downloading.
        # This lets a crashed run pick up where it left off without wiping the
        # papers that were successfully ingested before the crash.
        seen_ids = load_existing_arxiv_ids(client)
        print(f"[RESUME] Found {len(seen_ids)} existing papers in Qdrant — will skip these")
    else:
        wipe_collection(client)
        seen_ids = set()

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize with an empty node list — the Python index object is created
    # without touching Qdrant yet. The collection is created lazily on the first
    # insert_nodes call, at which point LlamaIndex infers vector dimensions from
    # the embedding model output (384-dim for bge-small-en-v1.5).
    # Incremental insertion (one paper at a time) means a mid-run crash loses
    # only the in-flight paper, not the entire batch.
    index = VectorStoreIndex([], storage_context=storage_context)

    stats = {"ingested": 0, "skipped": 0, "failed": 0}

    # One shared client instance across all topic searches.
    # delay_seconds=3 is required — ArXiv's API terms limit automated clients to
    # one request per 3 seconds. num_retries=3 handles transient network failures.
    arxiv_client = arxiv.Client(delay_seconds=3, num_retries=3)

    for topic, query in TOPICS.items():
        print(f"\n[TOPIC] {topic}  |  query: \"{query}\"")

        search = arxiv.Search(
            query=query,
            max_results=MAX_PER_TOPIC,
            # Relevance ranking surfaces the most cited / closely matched papers
            # first, which is more useful than recency for a research corpus
            sort_by=arxiv.SortCriterion.Relevance,
        )

        # Wrap the entire topic iteration so an API-level 429 or 503 (which the
        # arxiv library raises after exhausting num_retries) doesn't crash the
        # whole run — we log the failure and move to the next topic instead.
        try:
            for paper in arxiv_client.results(search):
                arxiv_id = extract_arxiv_id(paper.entry_id)

                if arxiv_id in seen_ids:
                    print(f"  [SKIP] {arxiv_id} — already ingested")
                    stats["skipped"] += 1
                    continue

                try:
                    title_preview = paper.title.replace("\n", " ")[:55]
                    print(f"  [GET]  {arxiv_id}  {title_preview}...")
                    nodes = chunk_paper(paper, topic)
                    index.insert_nodes(nodes)
                    seen_ids.add(arxiv_id)
                    stats["ingested"] += 1
                    print(f"  [OK]   {arxiv_id}  {len(nodes)} chunks stored")

                except Exception as e:
                    # A single bad PDF (scanned, corrupted, rate-limited) should
                    # not abort the topic — log and continue to the next paper
                    print(f"  [FAIL] {arxiv_id}  {e}")
                    stats["failed"] += 1

                finally:
                    # Throttle PDF downloads independently of the API query delay.
                    # ArXiv's PDF server enforces its own rate limit; without this
                    # sleep the bulk of downloads trigger 429s on the PDF endpoint.
                    time.sleep(PDF_DOWNLOAD_DELAY)

        except Exception as e:
            # API-level failure (429/503 after all retries) — skip this topic
            # entirely rather than crashing and losing all progress so far
            print(f"  [TOPIC ERROR] {topic} — API error after retries: {e}")

    print(f"\n{'=' * 42}")
    print(f"  Ingested : {stats['ingested']} papers")
    print(f"  Skipped  : {stats['skipped']} duplicates / already present")
    print(f"  Failed   : {stats['failed']} errors")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    run_arxiv_ingest(resume=resume)
