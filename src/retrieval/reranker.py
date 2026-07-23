"""
Cross-encoder reranking — a second-stage relevance scorer applied AFTER
initial retrieval, distinct from both the bi-encoder embeddings used for
retrieval and the RRF fusion previously tested and disabled.

WHY THIS IS ARCHITECTURALLY DIFFERENT from what's been tried so far:
  - Bi-encoder (current retrieval): embeds query and each document
    SEPARATELY, then compares vectors. Fast (documents can be pre-embedded
    once), but each embedding has no knowledge of the other side.
  - RRF fusion (tested, disabled): merges two INDEPENDENTLY produced
    rankings based on rank position — never looks at query and document
    together.
  - Cross-encoder (this module): takes the query and a candidate document
    CONCATENATED TOGETHER as a single input, and directly outputs a
    relevance score. Much more accurate because it can model interactions
    between specific query and document tokens — but slower, since it
    can't precompute anything in advance; every candidate must be
    reprocessed per query.

Standard two-stage pattern: bi-encoder retrieves a wider candidate set
(fast, broad recall) → cross-encoder reranks those candidates down to the
final top-k (slow but precise, only run on a small candidate set).

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — a small, fast, widely-used
general-purpose reranker trained on MS-MARCO passage ranking. No biomedical-
specific cross-encoder is as well-established/available as the biomedical
bi-encoder used for retrieval — this is a real, stated limitation, not a
hidden one.

Usage:
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(model_name)
    reranked_chunks = reranker.rerank(query, candidate_chunks, top_k=5)
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        log.info(f"Loading cross-encoder reranker: {model_name} ...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Score each (query, candidate_text) pair jointly, then return the
        top_k candidates sorted by the cross-encoder's relevance score
        (highest first) — a full re-sort, not just a filter.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, score in scored[:top_k]]
