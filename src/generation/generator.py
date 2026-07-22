"""
Answer generation — GENERATIVE approach using a local, open-source LLM.
No API key, no external service, no per-query cost.

Uses the same small local model already set up for RAGAS's judge role
(src/evaluation/local_llm.py) — Qwen2.5-1.5B-Instruct by default — running
entirely via HuggingFace transformers on your own machine. This restores
proper synthesized answers (the model reads the retrieved passages and
writes a coherent response, rather than just returning raw excerpts) while
keeping the whole project free of any paid API.

HONEST TRADE-OFF: a 1.5B-parameter local model produces noticeably less
polished, less reliable synthesis than a frontier model like GPT-4 or
Claude — expect occasional awkward phrasing, minor repetition, or
imperfect citation formatting. It's a genuine, usable generative RAG system,
just not a frontier-quality one. Swap `model_name` in config.yaml to a
larger local model (e.g. Qwen2.5-7B-Instruct, or microsoft/Phi-3-mini-
4k-instruct) if you have more compute/RAM available and want higher
quality — the interface here doesn't change.

Usage:
    from src.generation.generator import generate_answer
    result = generate_answer(question, retrieved_chunks, cfg)
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are an agricultural research assistant. Answer the question using ONLY the information in the research excerpts below. If the excerpts don't contain enough information to answer confidently, say so explicitly rather than guessing.

Cite sources using [1], [2], etc. matching the excerpt numbers. Keep the answer concise (2-4 sentences).

Research excerpts:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['title']} ({chunk.get('year', 'n.d.')})\n{chunk['text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


class LocalGenerator:
    """
    Lazily loads the local model once and reuses it across calls — avoids
    reloading a multi-GB model from disk on every single question, which
    would make the API/frontend unusably slow.
    """
    _instance = None

    def __init__(self, model_name: str, max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        log.info(f"Loading local generation model: {model_name} (first run downloads it) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.01,  # near-deterministic; 0.0 exactly can error on some HF pipelines
            do_sample=False,
            top_k=None,
            return_full_text=False,
        )

    @classmethod
    def get_instance(cls, model_name: str, max_new_tokens: int = 300) -> "LocalGenerator":
        if cls._instance is None:
            cls._instance = cls(model_name, max_new_tokens)
        return cls._instance

    def generate(self, prompt: str) -> str:
        outputs = self.pipe(prompt)
        return outputs[0]["generated_text"].strip()


def generate_answer(question: str, retrieved_chunks: list[dict], cfg: dict) -> dict:
    """
    Generate a synthesized answer grounded in the retrieved chunks, using a
    local LLM — no API key required.
    """
    t0 = time.time()

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "No relevant passages were found in the corpus for this question.",
            "sources": [],
            "model": cfg["generation"]["model_name"] if cfg else "none",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    gen_cfg = cfg["generation"]
    generator = LocalGenerator.get_instance(
        model_name=gen_cfg["model_name"],
        max_new_tokens=gen_cfg.get("max_new_tokens", 300),
    )

    prompt = build_prompt(question, retrieved_chunks)
    answer_text = generator.generate(prompt)

    sources = [
        {"pmid": c["pmid"], "title": c["title"], "year": c.get("year", "")}
        for c in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "model": gen_cfg["model_name"],
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


def main():
    """Quick manual test."""
    import argparse
    import yaml
    from src.retrieval.retriever import HybridRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retriever = HybridRetriever(cfg)
    chunks = retriever.search(args.question)

    result = generate_answer(args.question, chunks, cfg)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources: {[s['title'] for s in result['sources']]}")
    print(f"Model: {result['model']}  |  Latency: {result['latency_ms']}ms")


if __name__ == "__main__":
    main()
