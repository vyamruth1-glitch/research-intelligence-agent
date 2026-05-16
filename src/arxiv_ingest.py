"""
ArXiv-based corpus builder for the Research Intelligence Agent.

Replaces the manual PDF drop-in pipeline (ingest.py) with automated fetching
from ArXiv across six research topics, consistent metadata tagging, and
deduplication by canonical ArXiv ID.

Usage:
  python -m src.arxiv_ingest           # full run — wipes collection first
  python -m src.arxiv_ingest --resume  # skips already-ingested papers; adds new ones
  python -m src.arxiv_ingest --retry   # retries only papers that previously failed (data/failed_ids.txt)

Environment:
  QDRANT_HOST  Qdrant hostname (default: "qdrant" for Docker; set to "localhost" to run outside Docker)
"""

import os
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
PDF_RETRY_DELAY = 15        # longer delay for --retry mode; failed papers are more likely
                            # to hit 429 again if retried at the same pace
FAILED_IDS_FILE = "data/failed_ids.txt"  # tab-separated: arxiv_id \t topic

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


def get_qdrant_client() -> QdrantClient:
    # QDRANT_HOST defaults to "qdrant" (the Docker service name) so the script
    # works inside the container without any config. Set QDRANT_HOST=localhost
    # to run the script directly outside Docker against the exposed port.
    host = os.getenv("QDRANT_HOST", "qdrant")
    return QdrantClient(host=host, port=6333)


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


def load_failed_ids() -> list[tuple[str, str]]:
    # Read failed_ids.txt and return (arxiv_id, topic) pairs.
    # Each line is tab-separated: "2310.11511\tRAG"
    if not os.path.exists(FAILED_IDS_FILE):
        print(f"[RETRY] No failed IDs file found at {FAILED_IDS_FILE}")
        return []
    entries = []
    with open(FAILED_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
    return entries


def record_failure(arxiv_id: str, topic: str) -> None:
    # Append this arxiv_id+topic to the failures file so --retry can target it.
    # Appending (not overwriting) means multiple runs accumulate into one list;
    # the dedup in seen_ids prevents double-entries from causing double ingestion.
    os.makedirs(os.path.dirname(FAILED_IDS_FILE), exist_ok=True)
    with open(FAILED_IDS_FILE, "a") as f:
        f.write(f"{arxiv_id}\t{topic}\n")


def clear_failed_ids_file() -> None:
    if os.path.exists(FAILED_IDS_FILE):
        os.remove(FAILED_IDS_FILE)
        print(f"[RETRY] Cleared {FAILED_IDS_FILE} — starting fresh failure log")


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


def ingest_paper(paper: arxiv.Result, topic: str, index: VectorStoreIndex,
                 seen_ids: set[str], stats: dict, delay: int) -> None:
    """Attempt to ingest one paper; update seen_ids, stats, and failures file."""
    arxiv_id = extract_arxiv_id(paper.entry_id)

    if arxiv_id in seen_ids:
        print(f"  [SKIP] {arxiv_id} — already ingested")
        stats["skipped"] += 1
        return

    try:
        title_preview = paper.title.replace("\n", " ")[:55]
        print(f"  [GET]  {arxiv_id}  {title_preview}...")
        nodes = chunk_paper(paper, topic)
        index.insert_nodes(nodes)
        seen_ids.add(arxiv_id)
        stats["ingested"] += 1
        print(f"  [OK]   {arxiv_id}  {len(nodes)} chunks stored")

    except Exception as e:
        # Log, record to file for --retry, and continue to the next paper.
        # A single bad PDF (scanned, corrupted, rate-limited) should not
        # abort the topic or the run.
        print(f"  [FAIL] {arxiv_id}  {e}")
        stats["failed"] += 1
        record_failure(arxiv_id, topic)

    finally:
        # Throttle PDF downloads independently of the API query delay.
        # ArXiv's PDF server enforces its own rate limit; without this
        # sleep the bulk of downloads trigger 429s on the PDF endpoint.
        time.sleep(delay)


def run_arxiv_ingest(resume: bool = False, retry: bool = False) -> None:
    client = get_qdrant_client()

    if retry:
        # --retry: load only the previously-failed papers and attempt them
        # again with a longer inter-download delay to give ArXiv time to recover.
        failed = load_failed_ids()
        if not failed:
            return
        seen_ids = load_existing_arxiv_ids(client)
        print(f"[RETRY] {len(failed)} failed papers to retry | "
              f"{len(seen_ids)} already in Qdrant")
        # Clear the file now; failures during this retry run will re-populate it
        clear_failed_ids_file()
        delay = PDF_RETRY_DELAY
    elif resume:
        # --resume: skip everything already in Qdrant, add what's missing.
        # Used to continue after a crashed run without re-downloading successes.
        seen_ids = load_existing_arxiv_ids(client)
        print(f"[RESUME] Found {len(seen_ids)} existing papers in Qdrant — will skip these")
        delay = PDF_DOWNLOAD_DELAY
    else:
        wipe_collection(client)
        seen_ids = set()
        # Clear stale failures from a previous full run so --retry stays accurate
        clear_failed_ids_file()
        delay = PDF_DOWNLOAD_DELAY

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

    if retry:
        # Fetch metadata for specific arxiv_ids using ArXiv's id_list parameter.
        # Group into batches of 20 to avoid oversized API requests.
        print(f"\n[RETRY] Fetching {len(failed)} papers by ID...")
        batch_size = 20
        for i in range(0, len(failed), batch_size):
            batch = failed[i:i + batch_size]
            id_list = [aid for aid, _ in batch]
            topic_map = {aid: topic for aid, topic in batch}

            try:
                search = arxiv.Search(id_list=id_list)
                for paper in arxiv_client.results(search):
                    arxiv_id = extract_arxiv_id(paper.entry_id)
                    topic = topic_map.get(arxiv_id, "unknown")
                    ingest_paper(paper, topic, index, seen_ids, stats, delay)
            except Exception as e:
                print(f"  [BATCH ERROR] {e}")

    else:
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
                    ingest_paper(paper, topic, index, seen_ids, stats, delay)
            except Exception as e:
                # API-level failure after all retries — skip topic, preserve progress
                print(f"  [TOPIC ERROR] {topic} — API error after retries: {e}")

    print(f"\n{'=' * 42}")
    print(f"  Ingested : {stats['ingested']} papers")
    print(f"  Skipped  : {stats['skipped']} duplicates / already present")
    print(f"  Failed   : {stats['failed']} errors")
    if stats["failed"] > 0:
        print(f"  → Run with --retry to attempt failed papers at a slower pace")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    retry  = "--retry"  in sys.argv
    resume = "--resume" in sys.argv and not retry
    run_arxiv_ingest(resume=resume, retry=retry)
