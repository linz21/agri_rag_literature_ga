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
retriever.py  →  Semantic search (retrieval-tuned embeddings)
                  [hybrid BM25+RRF available via config, off by default — see Results]
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

Validated: this has been run successfully against 478 real papers across 5
agronomic search terms (see Results above). To reproduce or pull fresh data:

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

Validated end-to-end on real PubMed data (478 papers, 483 chunks, fetched
2026-07-22 across 5 agronomic search terms).

| Metric | Value |
|--------|-------|
| Corpus size | 478 real papers → 483 chunks |
| Retrieval — embedding model | `pritamdeka/S-PubMedBert-MS-MARCO` (retrieval-tuned) |
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

### Validated finding: semantic-only retrieval outperforms hybrid (BM25 + semantic) on this corpus

**Embedding model choice** (tested first): SciBERT (mean-pooled) vs.
`pritamdeka/S-PubMedBert-MS-MARCO` (retrieval-tuned). The retrieval-tuned
model scored 5/5 on-topic vs. SciBERT's 4/5 on a single test query — a
promising early signal that led to switching the default embedding model.

**Hybrid vs. semantic-only** (tested properly, across 10 queries): rather
than assume RRF fusion of BM25 + semantic search was strictly better (the
original plan), retrieval was compared side-by-side across 10 queries —
5 broad conceptual questions matching the corpus's search topics, plus 5
queries specifically chosen to give BM25 its best possible chance (exact
gene names, species names, technical acronyms):

| Query type | BM25's effect on relevance |
|---|---|
| Nitrogen timing | Hurt — dropped 2 precise maize-nitrogen matches |
| Remote sensing yield | Hurt — dropped the most directly on-topic review paper |
| Soil health | Mixed — no clear winner either way |
| Drought stress physiology | Hurt — dropped a specific gene study for tangential papers |
| ML yield prediction | Hurt — dropped a systematic review for an unrelated processing paper |
| Exact gene name (ZmWRKY74) | Mixed/slight hurt — even BM25's ideal case showed no clean win |
| Species/compound names (Aspergillus, aflatoxin) | Neutral/slight help — one good addition, one unrelated addition |
| Acronym (UAV) | Neutral — swapped one good match for another |
| Technical term (CRISPR) | Hurt — dropped stress-relevant papers for tangential ones |
| Broad control query | Hurt — added an unrelated livestock/manure paper |

**Result: 6/10 queries hurt, 4/10 mixed or neutral, 0/10 a clean win for
hybrid** — including in the 4 queries specifically designed to favor BM25's
exact-match strength. Two specific papers recurred as false positives
across 4+ unrelated queries (a UAV/ResNet50 crop-classification paper and
a generic "computational biology" review), indicating BM25 was matching on
shared generic terms rather than genuinely relevant exact-term hits.

**Decision:** semantic-only is now the default retrieval mode
(`retrieval.use_hybrid: false` in `configs/config.yaml`). BM25/RRF fusion
code remains available (`use_hybrid: true`) since a different corpus or
query style could plausibly favor it — but it is not the default given
this corpus's measured behavior. Diagnostic scripts used to generate this
evidence are kept in `scripts/compare_retrieval_modes.py` and
`scripts/compare_retrieval_modes_batch.py` for future re-validation if the
corpus changes significantly (e.g. after adding many more papers).

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
