"""
Hybrid retrieval combining BM25 (keyword/lexical matching) and semantic
search (Chroma vector similarity), fused via Reciprocal Rank Fusion (RRF).

Why hybrid: BM25 catches exact terminology matches (e.g. specific gene names,
chemical compounds, precise agronomic terms) that dense embeddings can miss
or fuzzy-match incorrectly. Semantic search catches conceptually related
content phrased differently. RRF combines both rankings without needing to
tune a weighting hyperparameter between two different score scales.

Usage:
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever(cfg)
    results = retriever.search("How does nitrogen timing affect corn yield?")
"""

import json
import logging
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists of document IDs into a single fused ranking.
    RRF score for a doc = sum over each list of 1 / (k + rank_in_that_list).
    k=60 is the standard default from the original RRF paper — dampens the
    influence of any single list's top rank being overly decisive.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retrieval_cfg = cfg["retrieval"]

        # Load chunks for BM25 (BM25 operates over raw text, not vectors)
        chunks_path = Path(cfg["data"]["processed_dir"]) / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} not found. Run chunking.py first.")

        with open(chunks_path) as f:
            self.chunks = json.load(f)

        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_ids = [c["chunk_id"] for c in self.chunks]

        # Connect to the same Chroma collection built by embed.py
        import chromadb
        vs_cfg = cfg["vector_store"]
        client = chromadb.PersistentClient(path=vs_cfg["persist_dir"])
        self.collection = client.get_or_create_collection(name=vs_cfg["collection_name"])

        # Lazy-load the embedder only when semantic search is actually called
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from src.embeddings.embed import SciBERTEmbedder
            embed_cfg = self.cfg["embeddings"]
            self._embedder = SciBERTEmbedder(embed_cfg["model_name"], pooling=embed_cfg["pooling_strategy"])
        return self._embedder

    def bm25_search(self, query: str, top_k: int) -> list[str]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.bm25_ids[i] for i in ranked_indices]

    def semantic_search(self, query: str, top_k: int) -> list[str]:
        embedder = self._get_embedder()
        query_embedding = embedder.embed([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["ids"][0] if results["ids"] else []

    def search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or self.retrieval_cfg["top_k_final"]

        bm25_ids = self.bm25_search(query, self.retrieval_cfg["top_k_bm25"])
        semantic_ids = self.semantic_search(query, self.retrieval_cfg["top_k_semantic"])

        fused = reciprocal_rank_fusion([bm25_ids, semantic_ids])
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]

        return [self.chunk_by_id[doc_id] for doc_id in top_ids if doc_id in self.chunk_by_id]


def main():
    """Quick manual test — run a sample query and print retrieved chunks."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retriever = HybridRetriever(cfg)
    results = retriever.search(args.query)

    print(f"\nQuery: {args.query}\n")
    for i, chunk in enumerate(results, 1):
        print(f"[{i}] {chunk['title']} ({chunk['year']})")
        print(f"    {chunk['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
