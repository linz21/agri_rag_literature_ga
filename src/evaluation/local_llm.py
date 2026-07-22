"""
Local LLM wrapper for RAGAS evaluation — no API key, no external service.

RAGAS's metrics (faithfulness, answer_relevancy, etc.) work by using an LLM
as a judge internally — this is inherent to how RAGAS is designed, not
something specific to this project. Rather than requiring a paid API key
(Anthropic, OpenAI, etc.), this wraps a small open-source instruction model
running locally via HuggingFace transformers, so evaluation stays fully
free and offline — consistent with the rest of this project.

HONEST TRADE-OFF: a small local model (1-2B parameters) is a meaningfully
weaker judge than GPT-4 or Claude, which is what RAGAS's documentation and
most published examples assume. Judge quality directly affects how much you
should trust the resulting scores — treat them as a rough, directional
signal (e.g. "did faithfulness get worse after a change?") rather than an
absolute, publication-grade quality measurement. This is a deliberate,
stated trade-off in exchange for zero cost and zero API key setup — not a
hidden limitation.

Usage:
    from src.evaluation.local_llm import get_local_ragas_llm, get_local_ragas_embeddings
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def get_local_ragas_llm(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 512):
    """
    Build a RAGAS-compatible LLM wrapper around a local HuggingFace model.

    Model choice: Qwen2.5-1.5B-Instruct is small enough to run on a laptop
    CPU in reasonable time (a few seconds per judgment call) while still
    following instructions well enough to be a workable judge. Swap to a
    larger model (e.g. Qwen2.5-7B-Instruct, or microsoft/Phi-3-mini-4k-instruct)
    via the model_name argument if you have more compute available and want
    higher judge quality.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
    from ragas.llms import LangchainLLMWrapper

    log.info(f"Loading local judge model: {model_name} (first run downloads the model) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU-safe; use torch.float16/bfloat16 if you have a GPU
        device_map="cpu",
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
    )

    langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return LangchainLLMWrapper(langchain_llm)


def get_local_ragas_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Some RAGAS metrics (e.g. context_precision, context_recall) also need an
    embedding model internally, separate from this project's own SciBERT
    retrieval embeddings. Using a small, fast, well-established sentence-
    embedding model here — this is purely for RAGAS's internal similarity
    calculations, not related to the project's actual retrieval pipeline.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    langchain_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return LangchainEmbeddingsWrapper(langchain_embeddings)
