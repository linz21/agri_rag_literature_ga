"""
Hugging Face Spaces entry point — Agricultural Research Assistant.

IMPORTANT ARCHITECTURAL NOTE: locally, src/frontend/app.py (Gradio) calls
src/api/main.py (FastAPI) over HTTP on localhost — two separate processes.
Hugging Face Spaces (Gradio SDK) runs a SINGLE process per Space, so that
two-server pattern doesn't directly translate. This file instead imports
the retriever and generator DIRECTLY and calls them in-process — the
same underlying RAG logic, just without the network hop between two
servers that only makes sense in a local multi-process setup.

ZERO-GPU NOTE (as of mid-2026): Hugging Face changed free-tier Spaces
policy so that new free accounts can only create Gradio Spaces on
ZeroGPU hardware, not CPU Basic (CPU Basic now requires a PRO
subscription). ZeroGPU provides free, on-demand access to a shared GPU
pool (H200-class hardware), which is actually a good fit here — GPU
inference is much faster than the ~90s/question measured on local CPU.

The standard ZeroGPU pattern (per Hugging Face's own documentation):
load models ONCE at startup on CPU/meta device, then move them to
'cuda' INSIDE a function decorated with @spaces.GPU — this decorator
allocates a GPU to the process only for the duration of that function
call. Moving an already-loaded model to GPU is a fast device-transfer
operation, not a reload from disk, so this is cheap on every call.

REAL BUGS FOUND AND FIXED DURING DEPLOYMENT (documented here since they
were non-obvious and worth remembering):

1. Models are now pre-loaded at Space STARTUP (see the module-level
   get_retriever()/get_generator() calls below), not lazily on the first
   question. The very first call previously needed to download ~3.5GB of
   models over the network from inside the 60-second @spaces.GPU budget,
   which sometimes wasn't enough time for a fresh download — moving the
   download to startup avoids competing with the GPU time budget.

2. transformers Pipeline objects track device SEPARATELY from the model
   itself. Moving only `pipe.model.to('cuda')` left the pipeline's own
   input-tensor preprocessing still targeting CPU, causing
   "Expected all tensors to be on the same device" errors. Both
   `pipe.model` AND `pipe.device` must be updated together — see
   run_rag_pipeline() below.

3. Gradio's experimental SSR mode conflicted with Hugging Face's proxy,
   causing SvelteKit routing errors. Fixed with `ssr_mode=False` in
   demo.launch().

4. A gradio_client schema bug (TypeError: argument of type 'bool' is not
   iterable — a known pydantic-version-related issue reported across
   multiple Gradio versions) was resolved by using README.md's sdk_version
   field to control the Gradio version (the correct HF Spaces mechanism)
   instead of also pinning gradio in requirements.txt, which was redundant
   and likely contributing to the version conflict.

The FastAPI backend (src/api/main.py) remains useful for local
development and any deployment where a separate API is genuinely
needed — it's not replaced, just not used for this single-process,
ZeroGPU-based deployment.

To deploy: see hf_space/DEPLOY.md for full step-by-step instructions.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
import gradio as gr
import spaces  # provided by the `spaces` package — required for ZeroGPU Spaces

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

_retriever = None
_generator = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever(cfg)
        # Ensure the embedder is loaded now (on CPU) so later GPU moves
        # are just a device transfer, not a fresh model load
        _retriever._get_embedder()
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        from src.generation.generator import LocalGenerator
        gen_cfg = cfg["generation"]
        _generator = LocalGenerator.get_instance(
            model_name=gen_cfg["model_name"],
            max_new_tokens=gen_cfg.get("max_new_tokens", 300),
        )
    return _generator


# Pre-load models at Space STARTUP (CPU, no GPU needed for downloading/
# loading weights) — rather than lazily on the first question, which would
# otherwise eat into the @spaces.GPU decorator's time budget below with
# a one-time ~3.5GB model download over the network. See bug #1 in the
# module docstring above.
print("Pre-loading models at startup ...")
get_retriever()
get_generator()
print("Models loaded and ready.")


@spaces.GPU(duration=120)
def run_rag_pipeline(question: str) -> dict:
    """
    The actual GPU-using work, isolated in one function so the @spaces.GPU
    decorator allocates a GPU for exactly this call. Moves the already-
    loaded embedding and generation models to 'cuda' for this call, runs
    retrieval + generation, and returns the result. duration=120 requests
    up to 2 minutes of GPU time per call as a safety margin.
    """
    import torch
    from src.generation.generator import build_prompt

    retriever = get_retriever()
    generator = get_generator()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move models to GPU for this call (cheap — weights already in RAM/loaded).
    # IMPORTANT: the transformers Pipeline object has its own separate
    # `.device` attribute used to decide where to place INPUT tensors
    # (tokenized text) — moving only `.model` leaves inputs on CPU while
    # weights are on GPU, causing "Expected all tensors to be on the same
    # device" errors. Both must be updated together. See bug #2 above.
    retriever._embedder.model.to(device)
    generator.pipe.model.to(device)
    generator.pipe.device = torch.device(device)

    chunks = retriever.search(question, top_k=cfg["retrieval"]["top_k_final"])

    if not chunks:
        return {
            "answer": "No relevant passages were found in the corpus for this question.",
            "sources": [],
        }

    prompt = build_prompt(question, chunks)
    answer_text = generator.generate(prompt)

    # The model sometimes appends its own "Sources: [1][2]..." line since
    # the prompt asks it to cite sources — strip that here since we add a
    # fuller, properly-formatted source list with titles/years below,
    # avoiding a confusing duplicate "Sources" section in the output.
    answer_text = re.sub(r"\n*Sources?:\s*(\[\d+\]\s*)+\s*$", "", answer_text, flags=re.IGNORECASE).strip()

    sources = [
        {"pmid": c["pmid"], "title": c["title"], "year": c.get("year", "")}
        for c in chunks
    ]
    return {"answer": answer_text, "sources": sources}


def ask_question(question: str, history: list) -> tuple[str, list]:
    if not question.strip():
        return "", history

    try:
        result = run_rag_pipeline(question)
        sources_text = "\n".join(
            f"  - {s['title']} ({s['year']})" for s in result["sources"]
        )
        response = f"{result['answer']}\n\n**Sources:**\n{sources_text}" if result["sources"] else result["answer"]

    except Exception as e:
        response = f"⚠ Error processing question: {e}"

    history.append((question, response))
    return "", history


with gr.Blocks(title="Agricultural Research Assistant") as demo:
    gr.Markdown("# 🌽 Agricultural Research Assistant")
    gr.Markdown(
        "Ask questions about corn yield, precision agriculture, and crop science. "
        "Answers are grounded in 478 real PubMed research paper abstracts, generated "
        "by a local open-source LLM (no API key), with citations.\n\n"
        "**Note:** this Space runs on free ZeroGPU hardware — each question briefly "
        "allocates a shared GPU for generation. Please be patient if a question takes "
        "a little while, especially the first one after the Space wakes up."
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
    demo.launch(ssr_mode=False)
