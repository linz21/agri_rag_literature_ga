"""
Embedding generation and vector store indexing using Chroma.

IMPORTANT DESIGN NOTE on SciBERT: SciBERT (allenai/scibert_scivocab_uncased)
is a base BERT encoder pretrained on scientific text — it is NOT natively a
sentence-embedding model like models trained specifically for retrieval
(e.g. sentence-transformers' all-MiniLM, or retrieval-tuned models like
pritamdeka/S-PubMedBert-MS-MARCO). Using SciBERT for retrieval requires
mean-pooling over token embeddings, which is what this module does — but
this is a weaker retrieval signal than a model actually trained with a
contrastive/retrieval objective. This was the explicit original plan
("SciBERT"), so it's implemented as specified, but see README Known Gaps
for a stronger alternative if retrieval quality turns out to be weak
(this is exactly the kind of thing worth validating empirically, the same
way Project 2's LSTM assumptions were tested rather than assumed correct).

Usage:
    python src/embeddings/embed.py
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class SciBERTEmbedder:
    def __init__(self, model_name: str, pooling: str = "mean", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.pooling = pooling

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.pooling == "mean":
                    embeddings = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                else:  # cls
                    embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def build_vector_store(chunks: list[dict], embedder: SciBERTEmbedder, cfg: dict):
    import chromadb

    vs_cfg = cfg["vector_store"]
    client = chromadb.PersistentClient(path=vs_cfg["persist_dir"])
    collection = client.get_or_create_collection(name=vs_cfg["collection_name"])

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"pmid": c["pmid"], "title": c["title"], "journal": c.get("journal", ""), "year": str(c.get("year", ""))}
        for c in chunks
    ]

    log.info(f"Embedding {len(texts)} chunks with {cfg['embeddings']['model_name']} ...")
    embeddings = embedder.embed(texts, batch_size=cfg["embeddings"]["batch_size"])

    # Chroma upsert in batches to avoid overly large single calls
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    log.info(f"Indexed {len(ids)} chunks into Chroma collection '{vs_cfg['collection_name']}'")
    return collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    chunks_path = Path(cfg["data"]["processed_dir"]) / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found. Run src/ingestion/chunking.py first.")

    with open(chunks_path) as f:
        chunks = json.load(f)

    embed_cfg = cfg["embeddings"]
    embedder = SciBERTEmbedder(embed_cfg["model_name"], pooling=embed_cfg["pooling_strategy"])

    build_vector_store(chunks, embedder, cfg)


if __name__ == "__main__":
    main()
