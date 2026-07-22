"""
Embedding generation and vector store indexing using Chroma.

Uses `pritamdeka/S-PubMedBert-MS-MARCO` — a biomedical BERT model
specifically FINE-TUNED FOR RETRIEVAL on MS-MARCO (the standard passage-
retrieval benchmark dataset). This replaces an earlier version that used
raw SciBERT with manual mean-pooling.

WHY THIS CHANGE: SciBERT is a base encoder pretrained on scientific text,
but never trained with a retrieval objective — mean-pooling its token
embeddings is a reasonable fallback, but a genuinely weaker signal than a
model actually trained to place semantically-similar query/passage pairs
close together in embedding space. This was flagged as a real, testable
design uncertainty when SciBERT was first used — validated empirically by
manually inspecting retrieval quality on real PubMed data (478 papers):
SciBERT's semantic search surfaced some clearly off-topic results (e.g. an
aflatoxin-prediction paper ranking for a nitrogen-timing query). Switching
to a retrieval-tuned model should measurably improve this.

Distributed via `sentence-transformers`, which handles the correct pooling
strategy internally (as trained), rather than requiring manual mean-pooling
code — simpler and more correct than the previous approach.

Usage:
    python src/embeddings/embed.py
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class RetrievalEmbedder:
    """
    Thin wrapper around sentence-transformers for a retrieval-tuned model.
    Kept as a class (rather than calling SentenceTransformer directly
    everywhere) so retriever.py's usage doesn't need to change regardless
    of which underlying embedding approach is used.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading retrieval-tuned embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )


def build_vector_store(chunks: list[dict], embedder: RetrievalEmbedder, cfg: dict):
    import chromadb

    vs_cfg = cfg["vector_store"]
    client = chromadb.PersistentClient(path=vs_cfg["persist_dir"])

    # Delete and recreate the collection fresh each time — prevents stale
    # entries from a previous embedding model/corpus lingering alongside
    # new ones (this exact issue was hit and fixed during real-data testing:
    # leftover demo-data chunks caused a KeyError during retrieval after
    # switching to real PubMed data, because upsert() doesn't remove IDs
    # that aren't in the new batch).
    try:
        client.delete_collection(name=vs_cfg["collection_name"])
        log.info(f"Cleared existing collection '{vs_cfg['collection_name']}' before rebuilding.")
    except Exception:
        pass  # collection didn't exist yet — fine on first run

    collection = client.get_or_create_collection(name=vs_cfg["collection_name"])

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"pmid": c["pmid"], "title": c["title"], "journal": c.get("journal", ""), "year": str(c.get("year", ""))}
        for c in chunks
    ]

    log.info(f"Embedding {len(texts)} chunks ...")
    embeddings = embedder.embed(texts, batch_size=cfg["embeddings"]["batch_size"])

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    log.info(f"Indexed {len(ids)} chunks into Chroma collection '{vs_cfg['collection_name']}'")
    return collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    chunks_path = Path(cfg["data"]["processed_dir"]) / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found. Run src/ingestion/chunking.py first.")

    with open(chunks_path) as f:
        chunks = json.load(f)

    embed_cfg = cfg["embeddings"]
    embedder = RetrievalEmbedder(embed_cfg["model_name"])

    build_vector_store(chunks, embedder, cfg)


if __name__ == "__main__":
    main()
