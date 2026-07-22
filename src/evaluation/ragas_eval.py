"""
RAGAS evaluation — measures RAG pipeline quality using an LLM-as-judge
approach across four standard metrics:
  - faithfulness: does the answer avoid claims not supported by the context?
  - answer_relevancy: does the answer actually address the question asked?
  - context_precision: are the retrieved chunks actually relevant?
  - context_recall: did retrieval find the chunks needed to answer correctly?

Uses a LOCAL open-source model as the judge (see local_llm.py) — no API key,
no external service, fully free. See local_llm.py's docstring for the
honest trade-off: a small local model is a weaker judge than GPT-4/Claude,
which most RAGAS documentation assumes. Treat scores as a directional
signal, not a publication-grade absolute measurement.

REQUIRES a golden dataset (question + ground-truth answer pairs) that does
NOT yet exist — this must be manually curated (see GOLDEN_DATASET_TEMPLATE
below for the expected structure).

Usage (once golden dataset exists):
    python src/evaluation/ragas_eval.py
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


GOLDEN_DATASET_TEMPLATE = """
Expected structure for data/processed/golden_qa_pairs.json — NOT YET CREATED:

[
  {
    "question": "How does nitrogen application timing affect corn yield?",
    "ground_truth": "Split nitrogen application, particularly side-dress "
                     "timing at V6 growth stage, improves nitrogen use "
                     "efficiency and can increase yield by 8-12 bushels "
                     "per acre under drought conditions compared to single "
                     "pre-plant application.",
    "reference_pmids": ["12345678"]  # papers the answer should be grounded in
  },
  ...
]

This must be manually curated (or carefully LLM-assisted with human review)
since it defines the "correct" answers evaluation is measured against. A
poor-quality golden dataset makes all downstream RAGAS scores meaningless —
this is not something to auto-generate carelessly.
"""


def load_golden_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n{GOLDEN_DATASET_TEMPLATE}"
        )
    with open(path) as f:
        return json.load(f)


def run_ragas_evaluation(golden_pairs: list[dict], cfg: dict) -> dict:
    """
    Run the full pipeline (retrieve + generate) for each golden question,
    then score against RAGAS metrics using a local LLM judge.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    from src.retrieval.retriever import HybridRetriever
    from src.generation.generator import generate_answer
    from src.evaluation.local_llm import get_local_ragas_llm, get_local_ragas_embeddings

    retriever = HybridRetriever(cfg)

    questions, answers, contexts, ground_truths = [], [], [], []

    for pair in golden_pairs:
        question = pair["question"]
        retrieved_chunks = retriever.search(question)
        result = generate_answer(question, retrieved_chunks, cfg)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in retrieved_chunks])
        ground_truths.append(pair["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics_to_run = [metric_map[m] for m in cfg["evaluation"]["metrics"] if m in metric_map]

    log.info("Loading local judge model (this may take a minute on first run) ...")
    local_llm = get_local_ragas_llm()
    local_embeddings = get_local_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=metrics_to_run,
        llm=local_llm,
        embeddings=local_embeddings,
    )
    return result.to_pandas().to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    golden_path = Path(cfg["evaluation"]["golden_dataset_path"])
    golden_pairs = load_golden_dataset(golden_path)  # raises FileNotFoundError with template if missing

    log.info(f"Running RAGAS evaluation on {len(golden_pairs)} golden Q&A pairs ...")
    results = run_ragas_evaluation(golden_pairs, cfg)

    out_path = Path("reports/ragas_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved: {out_path}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
