"""
FastAPI backend for the agricultural RAG system.

Run:
    uvicorn src.api.main:app --reload --port 8002
"""

from __future__ import annotations
import logging
import time

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Agricultural Research Assistant API",
    description=(
        "RAG-based Q&A over agronomic research papers. "
        "Built by Linlin Zhang — github.com/linz21/agri_rag_system"
    ),
    version="0.1.0",
)

_retriever = None
_cfg = None


def load_config() -> dict:
    global _cfg
    if _cfg is None:
        with open("configs/config.yaml") as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        cfg = load_config()
        _retriever = HybridRetriever(cfg)
        log.info("Retriever initialized.")
    return _retriever


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How does nitrogen timing affect corn yield?"])
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    pmid: str
    title: str
    year: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "retriever_loaded": _retriever is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        retriever = get_retriever()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retriever unavailable: {e}")

    try:
        chunks = retriever.search(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant papers found for this question.")

    try:
        from src.generation.generator import generate_answer
        cfg = load_config()
        result = generate_answer(req.question, chunks, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e}")

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        model=result["model"],
        latency_ms=result["latency_ms"],
    )
