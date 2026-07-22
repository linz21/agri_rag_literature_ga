"""
Compares semantic-only retrieval vs. hybrid (BM25 + semantic + RRF) side by
side, to make an evidence-based decision about whether BM25 is net helpful
or net harmful for a given query style — rather than assuming either way.

Usage:
    python scripts/compare_retrieval_modes.py --query "your question here"
"""

import argparse
import yaml

import sys
sys.path.insert(0, '.')
from src.retrieval.retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retriever = HybridRetriever(cfg)
    top_k = cfg["retrieval"]["top_k_final"]

    semantic_only_ids = retriever.semantic_search(args.query, top_k=top_k)
    hybrid_chunks = retriever.search(args.query, top_k=top_k)

    print(f"\nQuery: {args.query}\n")

    print(f"=== SEMANTIC-ONLY (top {top_k}) ===")
    for i, cid in enumerate(semantic_only_ids, 1):
        title = retriever.chunk_by_id[cid]["title"]
        print(f"[{i}] {title}")

    print(f"\n=== HYBRID (BM25 + semantic, RRF-fused, top {top_k}) ===")
    for i, chunk in enumerate(hybrid_chunks, 1):
        print(f"[{i}] {chunk['title']}")

    semantic_set = set(semantic_only_ids)
    hybrid_set = {c["chunk_id"] for c in hybrid_chunks}
    only_in_hybrid = hybrid_set - semantic_set
    only_in_semantic = semantic_set - hybrid_set

    print(f"\n=== DIFFERENCES ===")
    print(f"Results only in HYBRID (added by BM25): {len(only_in_hybrid)}")
    for cid in only_in_hybrid:
        print(f"  + {retriever.chunk_by_id[cid]['title']}")
    print(f"Results only in SEMANTIC-ONLY (dropped by adding BM25): {len(only_in_semantic)}")
    for cid in only_in_semantic:
        print(f"  - {retriever.chunk_by_id[cid]['title']}")


if __name__ == "__main__":
    main()
