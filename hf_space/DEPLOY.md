# Deploying to Hugging Face Spaces

This folder contains everything specific to the Spaces deployment. A few
files/folders from the main project also need to be copied in before
pushing — see Step 3 below.

## Step 1 — Create the Space

1. Go to https://huggingface.co/new-space
2. Owner: your account. Space name: e.g. `agri-rag-assistant`
3. License: your choice (e.g. MIT)
4. **SDK: Gradio**
5. **Hardware: ZeroGPU** — as of a 2026 policy change, new free accounts
   can only create Gradio Spaces on ZeroGPU (free, shared GPU pool) or
   PRO-gated CPU Basic. ZeroGPU is selected automatically for free
   accounts and works well here — see app.py's docstring for the
   `@spaces.GPU` pattern this deployment uses.
6. Click **Create Space**

## Step 2 — Clone the new (empty) Space repo locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/agri-rag-assistant
cd agri-rag-assistant
```

## Step 3 — Copy in the required files

From your main project root (`agri_rag_literature_ga/`), copy:

```bash
# From this hf_space/ folder — becomes the Space's entry point
cp /path/to/agri_rag_literature_ga/hf_space/app.py .
cp /path/to/agri_rag_literature_ga/hf_space/requirements.txt .
cp /path/to/agri_rag_literature_ga/hf_space/README.md .

# From the main project — needed for the app to actually run
cp -r /path/to/agri_rag_literature_ga/src .
cp -r /path/to/agri_rag_literature_ga/configs .

# The pre-built vector index and chunks — REQUIRED so the Space doesn't
# need to re-embed 483 chunks on every restart (which would be slow and
# wasteful). This is public research-paper data, safe to include.
mkdir -p data/processed data/chroma_db
cp /path/to/agri_rag_literature_ga/data/processed/chunks.json data/processed/
cp -r /path/to/agri_rag_literature_ga/data/chroma_db/* data/chroma_db/
```

**Verify the copied structure looks like:**
```
agri-rag-assistant/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── retrieval/
│   ├── embeddings/
│   └── generation/
├── configs/
│   └── config.yaml
└── data/
    ├── processed/chunks.json
    └── chroma_db/  (Chroma's binary index files)
```

## Step 4 — Push to the Space

```bash
git add .
git commit -m "Initial deployment"
git push
```

Hugging Face will automatically build and launch the Space — watch the
"Building" logs in the Space's web UI. First build will take a few minutes
(installing torch/transformers) plus additional time on first request to
download the two models (SciBERT-retrieval-tuned embedder ~420MB, Qwen2.5
generator ~3GB).

## Known limitations of this deployment

- **ZeroGPU quota** — free tier provides a limited daily GPU-time budget
  shared across all your ZeroGPU Spaces (Hugging Face's PRO tier extends
  this quota if it becomes a constraint for demo purposes).
- **Cold starts** — if the Space goes to sleep after inactivity (free tier
  default), the first request after waking will be slower while models
  reload into memory and the first GPU allocation happens.
- **Chroma index is static** — this deployment ships the vector index
  built from the corpus at the time it was copied in. To update with new
  papers, rebuild the index locally (`python src/embeddings/embed.py`)
  and re-copy `data/chroma_db/` before pushing again.
