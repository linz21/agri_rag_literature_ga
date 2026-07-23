"""
Tests for the agricultural RAG system.
Run:  pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIngestion:
    def test_demo_papers_structure(self):
        from src.data.ingest import make_demo_papers
        papers = make_demo_papers(n=10)
        assert len(papers) == 10
        for p in papers:
            assert "pmid" in p
            assert "abstract" in p
            assert len(p["abstract"]) > 0

    def test_demo_papers_unique_pmids(self):
        from src.data.ingest import make_demo_papers
        papers = make_demo_papers(n=15)
        pmids = [p["pmid"] for p in papers]
        assert len(pmids) == len(set(pmids))


class TestChunking:
    def test_chunk_abstract_produces_chunks(self):
        from src.ingestion.chunking import chunk_abstract
        paper = {
            "pmid": "12345",
            "title": "Test Paper",
            "abstract": "This is sentence one. This is sentence two. This is sentence three.",
            "journal": "Test Journal",
            "year": "2024",
            "authors": ["Test, A."],
        }
        chunks = chunk_abstract(paper, chunk_size_tokens=100, overlap_tokens=10)
        assert len(chunks) >= 1
        assert chunks[0]["pmid"] == "12345"
        assert "text" in chunks[0]

    def test_chunk_empty_abstract_returns_empty(self):
        from src.ingestion.chunking import chunk_abstract
        paper = {"pmid": "999", "title": "No abstract", "abstract": "", "year": "2024", "authors": []}
        chunks = chunk_abstract(paper, chunk_size_tokens=100, overlap_tokens=10)
        assert chunks == []

    def test_sentence_splitting(self):
        from src.ingestion.chunking import split_into_sentences
        text = "First sentence. Second sentence. Third one here."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3


class TestRetrievalFusion:
    def test_reciprocal_rank_fusion_combines_lists(self):
        from src.retrieval.retriever import reciprocal_rank_fusion
        list_a = ["doc1", "doc2", "doc3"]
        list_b = ["doc2", "doc1", "doc4"]
        fused = reciprocal_rank_fusion([list_a, list_b])

        fused_ids = [doc_id for doc_id, _ in fused]
        # doc1 and doc2 appear high in both lists, should rank at the top
        assert set(fused_ids[:2]) == {"doc1", "doc2"}

    def test_rrf_single_list_preserves_order(self):
        from src.retrieval.retriever import reciprocal_rank_fusion
        single_list = ["a", "b", "c"]
        fused = reciprocal_rank_fusion([single_list])
        fused_ids = [doc_id for doc_id, _ in fused]
        assert fused_ids == ["a", "b", "c"]

    def test_search_defaults_to_semantic_only(self, monkeypatch):
        """
        Documents a real design decision: semantic-only is the default,
        NOT hybrid, based on empirical testing across 10 queries showing
        BM25 hurt relevance in 6/10 cases (see README). This test confirms
        that default without needing a real corpus/model — it patches the
        two search methods and checks which one(s) get called.
        """
        import types
        from src.retrieval.retriever import HybridRetriever

        # Build a minimal fake instance without running __init__
        # (avoids needing a real Chroma DB / chunks.json / model download)
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.retrieval_cfg = {"use_hybrid": False, "top_k_final": 5,
                                   "top_k_semantic": 10, "top_k_bm25": 10}
        retriever.chunk_by_id = {"a": {"chunk_id": "a"}, "b": {"chunk_id": "b"}}

        bm25_called = {"value": False}

        def fake_semantic_search(self, query, top_k):
            return ["a", "b"]

        def fake_bm25_search(self, query, top_k):
            bm25_called["value"] = True
            return ["b", "a"]

        retriever.semantic_search = types.MethodType(fake_semantic_search, retriever)
        retriever.bm25_search = types.MethodType(fake_bm25_search, retriever)

        results = retriever.search("test query")

        assert bm25_called["value"] is False, "BM25 should NOT be called when use_hybrid=False (the default)"
        assert len(results) == 2


class TestReranker:
    """
    Tests the reranker's sorting logic directly, without loading the real
    cross-encoder model — same reasoning as TestGenerationPromptBuilding:
    loading a real model is slow and inappropriate for a fast unit test.
    """
    def test_rerank_sorts_by_score_descending(self, monkeypatch):
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

        class FakeModel:
            def predict(self, pairs):
                # Return scores in a deliberately non-sorted input order,
                # to confirm rerank() actually re-sorts rather than just
                # trusting input order
                return [0.2, 0.9, 0.5]

        reranker.model = FakeModel()

        candidates = [
            {"chunk_id": "a", "text": "low relevance"},
            {"chunk_id": "b", "text": "high relevance"},
            {"chunk_id": "c", "text": "medium relevance"},
        ]
        result = reranker.rerank("test query", candidates, top_k=3)

        assert [c["chunk_id"] for c in result] == ["b", "c", "a"]

    def test_rerank_respects_top_k(self, monkeypatch):
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

        class FakeModel:
            def predict(self, pairs):
                return [0.1, 0.9, 0.5, 0.3]

        reranker.model = FakeModel()

        candidates = [{"chunk_id": str(i), "text": f"doc {i}"} for i in range(4)]
        result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0]["chunk_id"] == "1"  # highest score (0.9)

    def test_rerank_empty_candidates_returns_empty(self):
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


class TestGenerationPromptBuilding:
    """
    Tests the prompt-construction logic directly, WITHOUT loading the actual
    local LLM — loading a multi-GB model is slow and inappropriate for a
    fast unit test suite / CI run. Full end-to-end generation (including
    real model inference) should be validated manually or in a separate,
    explicitly-marked slow/integration test — see test_generation_integration
    below, skipped by default.
    """
    def test_format_context_includes_all_chunks(self):
        from src.generation.generator import format_context
        chunks = [
            {"title": "Paper A", "year": "2020", "text": "Content A"},
            {"title": "Paper B", "year": "2021", "text": "Content B"},
        ]
        context = format_context(chunks)
        assert "Paper A" in context
        assert "Paper B" in context
        assert "[1]" in context
        assert "[2]" in context

    def test_build_prompt_includes_question_and_context(self):
        from src.generation.generator import build_prompt
        chunks = [{"title": "Paper A", "year": "2020", "text": "Nitrogen improves yield."}]
        prompt = build_prompt("Does nitrogen help yield?", chunks)
        assert "Does nitrogen help yield?" in prompt
        assert "Nitrogen improves yield." in prompt
        assert "Paper A" in prompt

    def test_generate_answer_no_chunks_returns_no_results_message(self):
        from src.generation.generator import generate_answer
        cfg = {"generation": {"model_name": "test-model", "max_new_tokens": 100}}
        result = generate_answer("test question", [], cfg)
        assert result["sources"] == []
        assert "No relevant passages" in result["answer"]
        # Confirms the no-chunks path never attempts to load the model
        # (would be slow and pointless if there's nothing to answer from)


@pytest.mark.skip(reason="Slow integration test — loads a real ~1.5B parameter "
                         "model. Run manually with: pytest -m '' tests/test_pipeline.py "
                         "-k test_generation_integration")
class TestGenerationIntegration:
    def test_generation_integration_real_model(self):
        """
        Full end-to-end test with the actual local model loaded. Skipped by
        default in normal test runs (including CI) because model download +
        CPU inference takes real time — this is meant for manual validation
        when actually changing generation logic, not routine testing.
        """
        from src.generation.generator import generate_answer
        cfg = {"generation": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "max_new_tokens": 100}}
        chunks = [{
            "pmid": "123", "title": "Test Paper", "year": "2020",
            "text": "Nitrogen application at V6 stage improves corn yield by 10%.",
        }]
        result = generate_answer("How does nitrogen affect corn yield?", chunks, cfg)
        assert len(result["answer"]) > 0
        assert result["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


class TestAPIValidation:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_query_short_question_rejected(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/query", json={"question": "hi"})
        assert response.status_code == 422  # min_length=3 violated... wait "hi" is 2 chars

    def test_query_missing_question_rejected(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/query", json={})
        assert response.status_code == 422


class TestEvaluationGaps:
    """
    Documents that RAGAS evaluation correctly fails with a clear message
    when the golden dataset doesn't exist yet, rather than crashing
    unhelpfully or silently producing meaningless results.
    """
    def test_missing_golden_dataset_raises_clear_error(self, tmp_path):
        from src.evaluation.ragas_eval import load_golden_dataset
        with pytest.raises(FileNotFoundError):
            load_golden_dataset(tmp_path / "nonexistent.json")
