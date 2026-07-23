"""
Interactive helper for building the golden Q&A dataset needed for RAGAS
evaluation. Retrieves real chunks from YOUR corpus for each candidate
question, shows you the actual abstract text, and prompts you to write a
ground-truth answer based on what you actually read — rather than either
guessing or auto-generating answers that might not reflect the real corpus.

This keeps human curation in the loop deliberately (see ragas_eval.py's
docstring: "a poor-quality golden dataset makes all downstream RAGAS
scores meaningless"). It speeds up curation without skipping the judgment
step that actually matters.

Usage:
    python scripts/build_golden_dataset.py
    (run from the project root, with an existing Chroma index already built)

Each run appends to data/processed/golden_qa_pairs.json — safe to run
multiple times across multiple sessions to build up the dataset gradually.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
import yaml
from src.retrieval.retriever import HybridRetriever

# Candidate questions to curate — same style as the 10 used for retrieval
# testing, covering the corpus's actual topical range. Edit/add your own.
CANDIDATE_QUESTIONS = [
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

GOLDEN_PATH = Path("data/processed/golden_qa_pairs.json")


def load_existing() -> list[dict]:
    if GOLDEN_PATH.exists():
        with open(GOLDEN_PATH) as f:
            return json.load(f)
    return []


def save(pairs: list[dict]):
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(pairs, f, indent=2)


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    retriever = HybridRetriever(cfg)
    existing = load_existing()
    done_questions = {p["question"] for p in existing}

    print(f"\n{len(existing)} question(s) already curated.")
    print("For each question below, you'll see the top retrieved passages")
    print("from YOUR real corpus. Read them, then type a ground-truth answer")
    print("based on what they actually say. Press Enter with no text to skip")
    print("a question (e.g. if the corpus doesn't have a good answer for it).\n")

    for question in CANDIDATE_QUESTIONS:
        if question in done_questions:
            print(f"[Already curated, skipping]: {question}")
            continue

        chunks = retriever.search(question, top_k=5)

        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print(f"{'='*70}")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] {c['title']} ({c.get('year', 'n.d.')}) — PMID {c['pmid']}")
            print(f"    {c['text'][:400]}")

        print(f"\n{'-'*70}")
        print("Type your ground-truth answer below (based on the passages above).")
        print("Press Enter twice when done, or just Enter once to skip this question.")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
                if not lines:
                    break
            lines.append(line)
        ground_truth = " ".join(l for l in lines if l).strip()

        if not ground_truth:
            print(f"Skipped: {question}")
            continue

        # Ask which PMIDs the answer actually draws from
        print("\nWhich source numbers ([1]-[5] above) does your answer draw from?")
        print("(e.g. '1,3' — comma separated, or Enter for 'all shown')")
        pmid_input = input().strip()
        if pmid_input:
            indices = [int(x.strip()) - 1 for x in pmid_input.split(",") if x.strip().isdigit()]
            reference_pmids = [chunks[i]["pmid"] for i in indices if 0 <= i < len(chunks)]
        else:
            reference_pmids = [c["pmid"] for c in chunks]

        existing.append({
            "question": question,
            "ground_truth": ground_truth,
            "reference_pmids": reference_pmids,
        })
        save(existing)  # save incrementally — don't lose progress if interrupted
        print(f"Saved. ({len(existing)} total)")

    print(f"\n{'='*70}")
    print(f"DONE. {len(existing)} question(s) in {GOLDEN_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
