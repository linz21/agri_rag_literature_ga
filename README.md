# 🌾 Agricultural Research Assistant — RAG over Scientific Literature

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A retrieval-augmented generation (RAG) system that answers questions about
corn yield, precision agriculture, and crop science by grounding responses
in real agronomic research paper abstracts, with citations.

## Architecture

```
PubMed E-utilities API (478 real papers, 5 agronomic search terms)
        ↓
chunking.py  →  sentence-group chunks (483 chunks)
        ↓
embed.py  →  pritamdeka/S-PubMedBert-MS-MARCO (retrieval-tuned)  →  Chroma vector store
        ↓
retriever.py  →  Semantic search (default)
                  [hybrid BM25+RRF and cross-encoder reranking both available
                   via config — both validated to not improve results on this
                   corpus, off by default]
        ↓
generator.py  →  local LLM (Qwen2.5-1.5B-Instruct)  →  synthesized answer + citations
        ↓
FastAPI /query  ←→  Gradio chat frontend
        ↓
ragas_eval.py  →  same local LLM as judge  →  faithfulness / relevancy / precision / recall
```

Fully local and free — no API keys, no per-query cost, no external LLM
service of any kind (PubMed's API is free and public, not an LLM service).
Both generation and evaluation reuse the same small
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

Validated: this has been run successfully against 478 real papers across 5
agronomic search terms. To reproduce or pull fresh data:

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

`sentence-transformers` (PubMedBert-MS-MARCO) · `transformers` (Qwen2.5-1.5B-Instruct) ·
`ChromaDB` · `FastAPI` · `Gradio` · `Docker` · `GitHub Actions` · `RAGAS` (local judge)

`rank-bm25` is included for the optional hybrid retrieval path
(`retrieval.use_hybrid: true`) but is not part of the default pipeline.

Fully local and API-key-free throughout — both answer generation and RAGAS
evaluation are powered by the same small open-source model running on your
own machine, not a paid API.

## Results

Validated end-to-end on real PubMed data (478 papers, 483 chunks, fetched
2026-07-22 across 5 agronomic search terms).

| Metric | Value |
|--------|-------|
| Corpus size | 478 real papers → 483 chunks |
| Retrieval — embedding model | `pritamdeka/S-PubMedBert-MS-MARCO` (retrieval-tuned) |
| Generation | Working — coherent, grounded, cited answers confirmed by manual review |
| Generation latency (CPU, Qwen2.5-1.5B-Instruct) | ~90 seconds per question |
| Faithfulness | 0.847 (5/10 valid — local judge failed to parse output on 5/10 questions) |
| Answer relevancy | 0.752 (10/10 valid) |
| Context precision | 0.863 (10/10 valid) |
| Context recall | 1.000 (10/10 valid) |


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
