---
title: Agricultural Research Assistant
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🌾 Agricultural Research Assistant

A retrieval-augmented generation (RAG) system answering questions about
corn yield, precision agriculture, and crop science — grounded in 478 real
PubMed research paper abstracts, with citations.

**Fully local and free — no API keys.** Retrieval uses a biomedical
retrieval-tuned embedding model (`pritamdeka/S-PubMedBert-MS-MARCO`).
Generation uses a small open-source LLM (`Qwen2.5-1.5B-Instruct`) running
directly in this Space, not a paid external API.

**Note on speed:** generation takes roughly 60-90 seconds per question on
this Space's CPU hardware — a known, stated trade-off for a fully free,
local-inference system rather than a paid API. Please be patient after
asking a question.

Full project source, evaluation results, and design documentation:
[github.com/linz21/agri_rag_literature_ga](https://github.com/linz21/agri_rag_literature_ga)
