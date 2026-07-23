---
title: Agricultural Research Assistant
emoji: 🌽
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
---

# 🌽 Agricultural Research Assistant

A retrieval-augmented generation (RAG) system answering questions about
corn yield, precision agriculture, and crop science - grounded in 478 real
PubMed research paper abstracts, with citations.

Fully free - no API keys, runs on Hugging Face's free ZeroGPU tier.
Retrieval uses a biomedical retrieval-tuned embedding model
(pritamdeka/S-PubMedBert-MS-MARCO). Generation uses a small open-source
LLM (Qwen2.5-1.5B-Instruct) running directly in this Space on shared
GPU hardware, not a paid external API.

Note on speed: this Space uses ZeroGPU - a shared, on-demand GPU pool
Hugging Face provides free of charge. Each question briefly allocates a
GPU for generation, which is meaningfully faster than CPU-only inference.
The first question after the Space wakes from idle may take longer while
models load into memory.

Full project source, evaluation results, and design documentation:
github.com/linz21/agri_rag_literature_ga

## Note on this deployment

The emoji field above was generated programmatically via Python's Unicode
escape (\U0001F33D) rather than copy-pasted directly. Earlier attempts to
include the emoji via copy/paste between tools resulted in corrupted,
invalid UTF-8 bytes, which caused a genuine "Configuration error" on the
Space (Hugging Face's YAML parser failed on the malformed bytes). A plain
text placeholder like "corn" was also tried and explicitly REJECTED by
Hugging Face's own YAML validator ("emoji" must match Unicode's
Extended_Pictographic pattern - a real emoji character, not a word).
If editing this field again, generate the character programmatically and
verify with Python's `repr()` (not a raw byte viewer like `cat -v`, which
makes even valid multi-byte characters look suspicious).
