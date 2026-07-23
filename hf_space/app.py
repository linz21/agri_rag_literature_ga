"""
Hugging Face Spaces entry point — Agricultural Research Assistant.

IMPORTANT ARCHITECTURAL NOTE: locally, src/frontend/app.py (Gradio) calls
src/api/main.py (FastAPI) over HTTP on localhost — two separate processes.
Hugging Face Spaces (Gradio SDK) runs a SINGLE process per Space, so that
two-server pattern doesn't directly translate. This file instead imports
the retriever and generator DIRECTLY and calls them in-process — the
same underlying RAG logic, just without the network hop between two
servers that only makes sense in a local multi-process setup.

The FastAPI backend (src/api/main.py) remains useful for local development,
testing, and any deployment where a separate API is genuinely needed (e.g.
if other services need to call this system programmatically) — it's not
replaced, just not used for this particular single-process deployment.

To deploy:
    1. Create a new Space at huggingface.co/new-space (SDK: Gradio)
    2. Copy this hf_space/ folder's contents to the Space repo root
       (this file becomes the Space's app.py)
    3. Also copy: ../src/, ../configs/, ../data/chroma_db/,
       ../data/processed/chunks.json (see hf_space/README.md for exact steps)
    4. Push to the Space's git remote
"""

import sys
from pathlib import Path

# Allows `from src...` imports to resolve when this file is at the Space
# root (same reasoning as the -m invocation needed to run scripts locally —
# see src/retrieval/retriever.py and src/generation/generator.py usage notes)
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import gradio as gr

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

_retriever = None
_generation_cfg = cfg["generation"]


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever(cfg)
    return _retriever


def ask_question(question: str, history: list) -> tuple[str, list]:
    if not question.strip():
        return "", history

    try:
        from src.generation.generator import generate_answer

        retriever = get_retriever()
        chunks = retriever.search(question, top_k=cfg["retrieval"]["top_k_final"])

        if not chunks:
            response = "No relevant passages were found in the corpus for this question."
        else:
            result = generate_answer(question, chunks, cfg)
            sources_text = "\n".join(
                f"  - {s['title']} ({s['year']})" for s in result["sources"]
            )
            response = f"{result['answer']}\n\n**Sources:**\n{sources_text}"

    except Exception as e:
        response = f"⚠ Error processing question: {e}"

    history.append((question, response))
    return "", history


with gr.Blocks(title="Agricultural Research Assistant") as demo:
    gr.Markdown("# 🌾 Agricultural Research Assistant")
    gr.Markdown(
        "Ask questions about corn yield, precision agriculture, and crop science. "
        "Answers are grounded in 478 real PubMed research paper abstracts, generated "
        "by a local open-source LLM (no API key), with citations.\n\n"
        "**Note:** generation takes ~60-90 seconds per question on CPU — this is a "
        "known, stated trade-off for running a fully free, local model instead of a "
        "paid API. Please be patient after submitting a question."
    )

    chatbot = gr.Chatbot(height=450)
    question_box = gr.Textbox(
        placeholder="e.g. How does nitrogen application timing affect corn yield?",
        label="Your question",
    )
    submit_btn = gr.Button("Ask", variant="primary")

    submit_btn.click(ask_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    question_box.submit(ask_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])

if __name__ == "__main__":
    demo.launch()
