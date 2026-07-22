# 🌾 Agricultural Research Assistant — RAG over Scientific Literature

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A retrieval-augmented generation (RAG) system that answers questions about
corn yield, precision agriculture, and crop science by grounding responses
in real agronomic research paper abstracts, with citations.

## Architecture

```
PubMed E-utilities API (abstracts)
        ↓
chunking.py  →  sentence-group chunks
        ↓
embed.py  →  SciBERT (mean-pooled) embeddings  →  Chroma vector store
        ↓
retriever.py  →  Hybrid search: BM25 + semantic  →  Reciprocal Rank Fusion
        ↓
generator.py  →  local LLM (Qwen2.5-1.5B-Instruct)  →  synthesized answer + citations
        ↓
FastAPI /query  ←→  Gradio chat frontend
        ↓
ragas_eval.py  →  same local LLM as judge  →  faithfulness / relevancy / precision / recall
```

Fully local and free — no API keys, no per-query cost, no external LLM
service of any kind. Both generation and evaluation reuse the same small
open-source model running via HuggingFace transformers on your own machine.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/linz21/agri_rag_system.git
cd agri_rag_system
pip install -r requirements.txt

# 2. Get papers (demo — no PubMed API needed, works immediately)
python src/data/ingest.py --demo

# 3. Chunk the abstracts
python src/ingestion/chunking.py

# 4. Build the embedding index (downloads SciBERT on first run, ~440MB)
python src/embeddings/embed.py

# 5. Test retrieval directly (returns ranked chunks, no generation yet)
#    (uses -m so Python can resolve the src.* package imports correctly —
#    running the .py file directly causes ModuleNotFoundError: no module named 'src')
python -m src.retrieval.retriever --query "How does nitrogen timing affect corn yield?"

# 6. Generate a synthesized answer (downloads Qwen2.5-1.5B-Instruct on first run, ~3GB)
python -m src.generation.generator --question "How does nitrogen timing affect corn yield?"

# 7. Serve the API
uvicorn src.api.main:app --reload --port 8002   # → http://localhost:8002/docs

# 8. (Optional) Launch the chat frontend
python src/frontend/app.py   # → http://localhost:7860
```

> **Fully local and free, no API keys required.** Answers are synthesized
> by a small open-source LLM (Qwen2.5-1.5B-Instruct) running on your own
> machine via HuggingFace transformers — not an extractive fallback, and
> not a paid API. First run downloads the model (~3GB); after that,
> everything runs offline. Expect noticeably less polished output than a
> frontier model like GPT-4/Claude — a genuine, usable trade-off for zero
> cost (see `src/generation/generator.py` docstring for details).

## Using Real PubMed Data

To move beyond the demo corpus and pull real agronomic research abstracts:

```bash
# Edit configs/config.yaml — set pubmed.email to your real email
# (required by NCBI's usage policy to identify API traffic)

python src/data/ingest.py           # fetches real papers via NCBI E-utilities
python src/ingestion/chunking.py
python src/embeddings/embed.py
```

No API key needed for PubMed itself — it's a free public API, rate-limited
to 3 requests/second (10/second if you register a free NCBI API key).

## Tech Stack

`transformers` (SciBERT + Qwen2.5-1.5B-Instruct) · `ChromaDB` · `rank-bm25` ·
`FastAPI` · `Gradio` · `Docker` · `GitHub Actions` · `RAGAS` (local judge)

Fully local and API-key-free throughout — both answer generation and RAGAS
evaluation are powered by the same small open-source model running on your
own machine, not a paid API.

## Results

Validated end-to-end on the demo corpus (30 synthetic papers). Real PubMed
data and a golden evaluation dataset are the next steps — RAGAS metrics
below remain TBD until both exist.

| Metric | Value |
|--------|-------|
| Corpus size | 30 papers → 30 chunks (demo data) |
| Retrieval | Working — correctly ranks topically relevant papers (validated manually) |
| Generation | Working — coherent, grounded, cited answers confirmed by manual review |
| Generation latency (CPU, Qwen2.5-1.5B-Instruct) | ~90 seconds per question |
| Faithfulness | TBD — requires golden dataset |
| Answer relevancy | TBD — requires golden dataset |
| Context precision | TBD — requires golden dataset |
| Context recall | TBD — requires golden dataset |

**Honest note on latency:** ~90s/question on CPU is slow, a real trade-off
for zero-cost local inference. A GPU, a smaller/quantized model, or batching
would bring this down significantly — worth stating plainly rather than
hiding behind a vague "TBD."

## Project Structure

```
agri_rag_system/
├── src/
│   ├── data/ingest.py                 # PubMed E-utilities ingestion + demo generator
│   ├── ingestion/chunking.py          # Sentence-group chunking
│   ├── embeddings/embed.py            # SciBERT mean-pooled embeddings + Chroma indexing
│   ├── retrieval/retriever.py         # Hybrid BM25 + semantic search, RRF fusion
│   ├── generation/generator.py        # Local LLM (Qwen2.5) — synthesized answer + citations, no API key
│   ├── evaluation/ragas_eval.py       # RAGAS metrics
│   ├── api/main.py                    # FastAPI /query endpoint
│   └── frontend/app.py                # Gradio chat interface
├── docker/Dockerfile                  # API container
├── tests/test_pipeline.py             # Test suite
├── configs/config.yaml                # All settings — single source of truth
└── .github/workflows/ci.yml           # Tests on every push (demo data, no API key needed)
```
