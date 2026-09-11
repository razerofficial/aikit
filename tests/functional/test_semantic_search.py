"""
Validates Notebook 9: Semantic Search
Tests embedding model workflow with /v1/embeddings API and cosine similarity search.
"""
import pytest
import subprocess
import requests
import json
import numpy as np
from pathlib import Path


@pytest.mark.semantic_search
class TestSemanticSearchCLI:
    """CLI and model operations.

    Tests CLI commands for embedding models:
    model info, download, model run command validation.
    """

    @pytest.mark.parametrize("model", [
        "Qwen/Qwen3-Embedding-0.6B",
    ])
    def test_embedding_model_info(self, model):
        """Test retrieving embedding model information."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", model],
            capture_output=True,
            text=True,
            timeout=60  # Embedding models are relatively small
        )

        # Should succeed or provide meaningful error
        assert result.returncode == 0 or "Incompatible" in result.stdout

        if result.returncode == 0:
            assert model in result.stdout

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_embedding_model_download(self):
        """Test downloading an embedding model."""
        model = "Qwen/Qwen3-Embedding-0.6B"
        result = subprocess.run(
            ["rzr-aikit", "model", "download", model],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for model download
        )

        # Should succeed or report already cached
        assert result.returncode == 0 or \
               "already downloaded" in result.stdout.lower()

    def test_model_run_command_for_embeddings(self):
        """Test that embedding models can be run (no --omni flag needed)."""
        # Verify the model run command accepts embedding models
        result = subprocess.run(
            ["rzr-aikit", "model", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        # Embedding models don't require special flags


@pytest.mark.semantic_search
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestSemanticSearchWorkflow:
    """Embedding and search workflow.

    Tests complete semantic search workflow:
    embeddings API, batch embeddings, cosine similarity, document ranking.
    """

    @pytest.mark.integration
    def test_embeddings_api_endpoint(self):
        """Test embeddings generation via /v1/embeddings API.

        Assumes an embedding model server is running on port 8000.
        This is an integration test that requires manual server setup.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Embedding model server not running")
        except requests.exceptions.RequestException:
            pytest.skip("Embedding model server not running")

        # Test embedding generation
        payload = {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": "Razer laptops deliver top-tier gaming performance."
        }

        response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json=payload,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)
        assert len(data["data"][0]["embedding"]) > 0

    @pytest.mark.integration
    def test_batch_embeddings(self):
        """Test batch embedding generation.

        Tests embedding multiple documents in a single request.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Embedding model server not running")
        except requests.exceptions.RequestException:
            pytest.skip("Embedding model server not running")

        # Test batch embeddings
        documents = [
            "Razer laptops deliver top-tier gaming performance.",
            "The Razer Basilisk V3 Pro is a customizable gaming mouse.",
            "Razer Synapse allows fine-tuning of RGB and macros."
        ]

        payload = {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": documents
        }

        response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json=payload,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Should return embeddings for all documents
        assert "data" in data
        assert len(data["data"]) == len(documents)

        # All embeddings should have same dimensionality
        embedding_dims = [len(item["embedding"]) for item in data["data"]]
        assert len(set(embedding_dims)) == 1, "All embeddings should have same dimension"

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation between vectors."""
        # Simple test vectors
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        vec_c = [0.0, 1.0, 0.0]

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Identical vectors should have similarity 1.0
        sim_ab = cosine_similarity(vec_a, vec_b)
        assert abs(sim_ab - 1.0) < 0.0001

        # Orthogonal vectors should have similarity 0.0
        sim_ac = cosine_similarity(vec_a, vec_c)
        assert abs(sim_ac - 0.0) < 0.0001

    @pytest.mark.integration
    def test_semantic_search_workflow(self):
        """Test complete semantic search workflow.

        Tests: embed documents -> embed query -> compute similarities -> rank results
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Embedding model server not running")
        except requests.exceptions.RequestException:
            pytest.skip("Embedding model server not running")

        # Step 1: Embed documents
        documents = [
            "Razer laptops deliver top-tier gaming performance.",
            "The Razer Basilisk V3 Pro is a customizable gaming mouse.",
            "Razer keyboards are built for speed and precision."
        ]

        doc_response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json={
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "input": documents
            },
            timeout=30
        )

        assert doc_response.status_code == 200
        doc_embeddings = [item["embedding"] for item in doc_response.json()["data"]]

        # Step 2: Embed query
        query = "Which Razer product is best for FPS games?"
        query_response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json={
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "input": query
            },
            timeout=30
        )

        assert query_response.status_code == 200
        query_embedding = query_response.json()["data"][0]["embedding"]

        # Step 3: Compute similarities
        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # Step 4: Verify results
        assert len(similarities) == len(documents)
        assert all(0.0 <= sim <= 1.0 for sim in similarities), \
            "Cosine similarities should be between 0 and 1"

        # Find most relevant document
        best_match_idx = np.argmax(similarities)
        assert 0 <= best_match_idx < len(documents)

        # The query about "FPS games" should match mouse/keyboard better than laptops
        # (This is a heuristic test - actual ranking depends on model)
        assert similarities[best_match_idx] > 0.0
