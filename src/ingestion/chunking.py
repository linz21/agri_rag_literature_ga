"""
Chunking for ingested papers.

IMPORTANT DESIGN NOTE: PubMed abstracts (unlike full papers) rarely have
explicit section headers (Methods/Results/Discussion) — they're typically
one continuous paragraph. "Section-based chunking" as originally envisioned
mainly applies to full-text papers (e.g. from PMC open access). Since this
project's corpus is abstract-only (see ingest.py's docstring), chunking here
is realistically PARAGRAPH/SENTENCE-GROUP based, with a fallback structure
that would support true section-based chunking if full-text PMC papers are
added later. This is flagged explicitly rather than silently doing something
different from what "section-based" implies — see README Known Gaps.

Usage:
    python src/ingestion/chunking.py
"""

import argparse
import json
import logging
import re
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def naive_token_count(text: str) -> int:
    """Rough token estimate (~0.75 words per token for English) — good enough
    for chunk-size decisions without pulling in a full tokenizer here."""
    return int(len(text.split()) / 0.75)


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter. Not as robust as spaCy/nltk, but avoids an
    extra heavy dependency for this straightforward use case."""
    # Split on '. ' followed by a capital letter — a reasonable heuristic
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_abstract(paper: dict, chunk_size_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Chunk a single paper's abstract into overlapping windows of sentences,
    targeting `chunk_size_tokens` per chunk. Short abstracts (the common
    case) will produce a single chunk.
    """
    abstract = paper.get("abstract", "")
    if not abstract:
        return []

    sentences = split_into_sentences(abstract)
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = naive_token_count(sentence)

        if current_tokens + sentence_tokens > chunk_size_tokens and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            # Overlap: keep the last sentence(s) that fit within overlap_tokens
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                if overlap_count + naive_token_count(s) > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += naive_token_count(s)
            current_chunk_sentences = overlap_sentences
            current_tokens = overlap_count

        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return [
        {
            "chunk_id": f"{paper['pmid']}_chunk{i}",
            "pmid": paper["pmid"],
            "title": paper["title"],
            "text": chunk_text,
            "journal": paper.get("journal", ""),
            "year": paper.get("year", ""),
            "authors": paper.get("authors", []),
        }
        for i, chunk_text in enumerate(chunks)
    ]


def chunk_all_papers(papers: list[dict], cfg: dict) -> list[dict]:
    chunk_size = cfg["chunking"]["chunk_size_tokens"]
    overlap = cfg["chunking"]["chunk_overlap_tokens"]

    all_chunks = []
    for paper in papers:
        chunks = chunk_abstract(paper, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["data"]["raw_dir"]) / "papers.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run src/data/ingest.py first.")

    with open(raw_path) as f:
        papers = json.load(f)

    chunks = chunk_all_papers(papers, cfg)

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.json"

    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    log.info(f"Chunked {len(papers)} papers into {len(chunks)} chunks → {out_path}")
    avg_chunks_per_paper = len(chunks) / len(papers) if papers else 0
    log.info(f"Average chunks per paper: {avg_chunks_per_paper:.1f} "
             f"(low number expected — abstracts are short; see module docstring)")


if __name__ == "__main__":
    main()
