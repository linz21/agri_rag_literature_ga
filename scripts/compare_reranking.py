"""
Compares semantic-only retrieval vs. semantic + cross-encoder reranking,
across the SAME 10 queries used to test hybrid (BM25+RRF) vs semantic-only.
Same evidence-based approach — don't assume reranking helps, test it.

Usage:
    python scripts/compare_reranking.py
"""

import sys
sys.path.insert(0, '.')
import yaml
from src.retrieval.retriever import HybridRetriever

# Same 10 queries used for the hybrid vs. semantic-only comparison —
# keeps the evidence directly comparable across both experiments.
TEST_QUERIES = [
    "How does nitrogen timing affect corn yield?",
    "How is remote sensing used to predict crop yield?",
    "What soil health indicators matter for precision agriculture?",
    "How does drought stress affect corn physiology?",
    "Can machine learning models predict corn yield accurately?",
    "What is the function of the ZmWRKY74 gene in maize?",
    "How does Aspergillus flavus cause aflatoxin contamination in maize?",
    "What are the applications of UAV imaging in crop monitoring?",
    "How is CRISPR used to improve crop stress tolerance?",
    "What precision agriculture technologies improve crop management?",
]


def compare_one_query(retriever: HybridRetriever, query: str, top_k: int):
    semantic_only = retriever.search(query, top_k=top_k, use_hybrid=False, use_reranker=False)
    reranked = retriever.search(query, top_k=top_k, use_hybrid=False, use_reranker=True)

    semantic_ids = {c["chunk_id"] for c in semantic_only}
    reranked_ids = {c["chunk_id"] for c in reranked}

    only_in_reranked = reranked_ids - semantic_ids
    only_in_semantic = semantic_ids - reranked_ids

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    print(f"\nSemantic-only top {top_k}:")
    for i, c in enumerate(semantic_only, 1):
        print(f"  [{i}] {c['title']}")

    print(f"\nReranked top {top_k}:")
    for i, c in enumerate(reranked, 1):
        print(f"  [{i}] {c['title']}")

    print(f"\nAdded by reranking (in reranked, not in semantic-only): {len(only_in_reranked)}")
    for cid in only_in_reranked:
        print(f"  + {retriever.chunk_by_id[cid]['title']}")
    print(f"Dropped by reranking (in semantic-only, not in reranked): {len(only_in_semantic)}")
    for cid in only_in_semantic:
        print(f"  - {retriever.chunk_by_id[cid]['title']}")


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    retriever = HybridRetriever(cfg)
    top_k = cfg["retrieval"]["top_k_final"]

    print(f"\nTesting {len(TEST_QUERIES)} queries — comparing semantic-only vs.")
    print("semantic + cross-encoder reranking. Review each query's added/dropped")
    print("results and judge manually whether reranking was a net improvement.\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}/{len(TEST_QUERIES)}]", end="")
        compare_one_query(retriever, query, top_k)

    print(f"\n{'='*70}")
    print(f"ALL {len(TEST_QUERIES)} QUERIES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
