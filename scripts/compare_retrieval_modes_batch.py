"""
Runs the semantic-only vs. hybrid comparison across MULTIPLE representative
queries (one per configured search topic), so a decision about whether to
keep BM25 in the fusion is based on a representative sample, not a single
query. This directly addresses the concern that one query isn't a valid
basis for a general conclusion.

Usage:
    python scripts/compare_retrieval_modes_batch.py
"""

import sys
sys.path.insert(0, '.')
import yaml
from src.retrieval.retriever import HybridRetriever

# One representative query per configured PubMed search topic —
# covers the actual topical range of the corpus, not just one angle.
TEST_QUERIES = [
    "How does nitrogen timing affect corn yield?",
    "How is remote sensing used to predict crop yield?",
    "What soil health indicators matter for precision agriculture?",
    "How does drought stress affect corn physiology?",
    "Can machine learning models predict corn yield accurately?",
    # Second batch — added to broaden the sample beyond 5 queries, and to
    # specifically include query styles where BM25's theoretical strength
    # (exact keyword/technical-term matching) should have the best chance
    # to show a real advantage over semantic search, rather than only
    # testing broad conceptual questions.
    "What is the function of the ZmWRKY74 gene in maize?",          # exact gene name — BM25's ideal case
    "How does Aspergillus flavus cause aflatoxin contamination in maize?",  # exact species/compound names
    "What are the applications of UAV imaging in crop monitoring?",  # specific acronym (UAV)
    "How is CRISPR used to improve crop stress tolerance?",          # exact technical term (CRISPR)
    "What precision agriculture technologies improve crop management?",  # broad, semantic-favoring (control query)
]


def compare_one_query(retriever: HybridRetriever, query: str, top_k: int):
    semantic_only_ids = retriever.semantic_search(query, top_k=top_k)
    hybrid_chunks = retriever.search(query, top_k=top_k)

    semantic_set = set(semantic_only_ids)
    hybrid_set = {c["chunk_id"] for c in hybrid_chunks}
    only_in_hybrid = hybrid_set - semantic_set
    only_in_semantic = semantic_set - hybrid_set

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    print(f"\nSemantic-only top {top_k}:")
    for i, cid in enumerate(semantic_only_ids, 1):
        print(f"  [{i}] {retriever.chunk_by_id[cid]['title']}")

    print(f"\nHybrid top {top_k}:")
    for i, chunk in enumerate(hybrid_chunks, 1):
        print(f"  [{i}] {chunk['title']}")

    print(f"\nAdded by BM25 (in hybrid, not in semantic-only): {len(only_in_hybrid)}")
    for cid in only_in_hybrid:
        print(f"  + {retriever.chunk_by_id[cid]['title']}")

    print(f"Dropped by adding BM25 (in semantic-only, not in hybrid): {len(only_in_semantic)}")
    for cid in only_in_semantic:
        print(f"  - {retriever.chunk_by_id[cid]['title']}")

    return len(only_in_hybrid), len(only_in_semantic)


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    retriever = HybridRetriever(cfg)
    top_k = cfg["retrieval"]["top_k_final"]

    print(f"\nTesting {len(TEST_QUERIES)} queries — review each one's added/dropped")
    print("results and judge manually whether each addition/removal was a net")
    print("improvement or a net harm to relevance.\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}/{len(TEST_QUERIES)}]", end="")
        compare_one_query(retriever, query, top_k)

    print(f"\n{'='*70}")
    print(f"ALL {len(TEST_QUERIES)} QUERIES COMPLETE")
    print("Next: manually tally how many queries BM25 helped vs. hurt, using")
    print("the added/dropped lists above, then decide whether to keep hybrid")
    print("fusion as default, semantic-only as default, or something in between.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
