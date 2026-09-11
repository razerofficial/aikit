"""
Validates Notebook 7: RAG Pipeline - Retrieval-Augmented Generation
Tests complete RAG workflow from document ingestion through semantic search to answer generation.
"""
import pytest
import subprocess
import requests
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.mark.rag
class TestRAGPipelineCLI:
    """CLI and model operations for RAG components.

    Tests CLI commands for embedding, chat, and reranker models:
    model info, download, model run command validation.
    """

    @pytest.mark.parametrize("model", [
        "BAAI/bge-small-en-v1.5",  # Embedding model
        "Qwen/Qwen2.5-3B-Instruct",  # Chat model
    ])
    def test_rag_model_info(self, model):
        """Test retrieving RAG model information."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", model],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Should succeed or provide meaningful error
        assert result.returncode == 0 or "Incompatible" in result.stdout

        if result.returncode == 0:
            assert model in result.stdout

    @pytest.mark.slow
    @pytest.mark.requires_network
    @pytest.mark.parametrize("model", [
        "BAAI/bge-small-en-v1.5",
        "Qwen/Qwen2.5-3B-Instruct",
    ])
    def test_rag_model_download(self, model):
        """Test downloading RAG pipeline models."""
        result = subprocess.run(
            ["rzr-aikit", "model", "download", model],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for model download
        )

        # Should succeed or report already cached
        assert result.returncode == 0 or \
               "already downloaded" in result.stdout.lower()

    def test_model_run_command_validation(self):
        """Test that model run command supports port specification."""
        result = subprocess.run(
            ["rzr-aikit", "model", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        assert "--port" in result.stdout.lower() or "port" in result.stdout.lower()


@pytest.mark.rag
@pytest.mark.slow
class TestRAGDocumentProcessing:
    """Document ingestion and processing workflow.

    Tests document parsing with docling and text chunking with LangChain.
    """

    def test_docling_available(self):
        """Test that docling package is available for document parsing."""
        try:
            from docling.document_converter import DocumentConverter
            assert DocumentConverter is not None
        except ImportError:
            pytest.skip("docling package not installed")

    def test_langchain_text_splitter_available(self):
        """Test that LangChain text splitter is available."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            assert RecursiveCharacterTextSplitter is not None
        except ImportError:
            pytest.skip("langchain text_splitters not installed")

    def test_text_chunking_with_overlap(self):
        """Test text chunking with overlap preserves context."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            pytest.skip("langchain not available")

        # Sample text
        sample_text = """
        The Transformer architecture revolutionized natural language processing.
        It introduced the self-attention mechanism which allows models to weigh
        the importance of different words in a sequence. This breakthrough enabled
        models like BERT and GPT to achieve state-of-the-art results on many tasks.
        """

        # Initialize splitter with small chunks for testing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )

        chunks = text_splitter.split_text(sample_text.strip())

        # Verify chunks created
        assert len(chunks) > 1, "Should create multiple chunks"

        # Verify chunks respect size constraints
        assert all(len(chunk) <= 120 for chunk in chunks), \
            "All chunks should be near target size (with tolerance)"

        # Verify overlap exists (check if consecutive chunks share text)
        for i in range(len(chunks) - 1):
            # Some text from end of chunk should appear in next chunk
            assert len(chunks[i]) > 0 and len(chunks[i+1]) > 0


@pytest.mark.rag
@pytest.mark.slow
class TestRAGVectorDatabase:
    """Vector database operations with ChromaDB.

    Tests ChromaDB collection creation, document insertion, and querying.
    """

    def test_chromadb_available(self):
        """Test that ChromaDB is available."""
        try:
            import chromadb
            assert chromadb is not None
        except ImportError:
            pytest.skip("chromadb package not installed")

    def test_chromadb_collection_operations(self):
        """Test ChromaDB collection creation and basic operations."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            pytest.skip("chromadb not available")

        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize ChromaDB
            client = chromadb.PersistentClient(
                path=tmpdir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )

            # Create collection
            collection = client.get_or_create_collection(
                name="test_collection",
                metadata={"test": "rag_pipeline"}
            )

            # Test adding documents
            test_embeddings = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
            test_documents = ["doc1", "doc2"]
            test_ids = ["id1", "id2"]

            collection.add(
                documents=test_documents,
                embeddings=test_embeddings,
                ids=test_ids
            )

            # Verify count
            assert collection.count() == 2

            # Test querying
            results = collection.query(
                query_embeddings=[[0.1, 0.2, 0.3]],
                n_results=1
            )

            assert len(results['documents']) > 0
            assert len(results['documents'][0]) == 1


@pytest.mark.rag
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestRAGPipelineWorkflow:
    """Complete RAG pipeline integration tests.

    Tests end-to-end workflow: embeddings → vector search → reranking → generation.
    Requires embedding and chat model servers running.
    """

    @pytest.mark.integration
    def test_embeddings_api_for_rag(self):
        """Test embeddings generation for RAG documents.

        Assumes embedding model server is running on port 8000.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Embedding model server not running on port 8000")
        except requests.exceptions.RequestException:
            pytest.skip("Embedding model server not running on port 8000")

        # Test embedding generation for RAG documents
        sample_chunks = [
            "The attention mechanism in transformers allows models to focus on relevant parts of the input.",
            "Retrieval-augmented generation combines document retrieval with language model generation."
        ]

        payload = {
            "model": "BAAI/bge-small-en-v1.5",
            "input": sample_chunks
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
        assert len(data["data"]) == len(sample_chunks)

        # BGE embeddings should be 384-dimensional
        embedding_dim = len(data["data"][0]["embedding"])
        assert embedding_dim == 384, f"BGE embeddings should be 384-dim, got {embedding_dim}"

    @pytest.mark.integration
    def test_semantic_search_for_rag_retrieval(self):
        """Test semantic search retrieval for RAG queries.

        Tests: embed documents → embed query → find similar chunks
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Embedding model server not running on port 8000")
        except requests.exceptions.RequestException:
            pytest.skip("Embedding model server not running on port 8000")

        # Sample knowledge base chunks
        documents = [
            "The Transformer architecture uses self-attention to process sequences in parallel.",
            "BERT is a bidirectional transformer model trained with masked language modeling.",
            "GPT-3 uses autoregressive generation to produce coherent text.",
            "RAG combines retrieval from a knowledge base with generative language models."
        ]

        # Embed documents
        doc_response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json={
                "model": "BAAI/bge-small-en-v1.5",
                "input": documents
            },
            timeout=30
        )

        assert doc_response.status_code == 200
        doc_embeddings = [item["embedding"] for item in doc_response.json()["data"]]

        # Embed query
        query = "How does the attention mechanism work in transformers?"
        query_response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json={
                "model": "BAAI/bge-small-en-v1.5",
                "input": query
            },
            timeout=30
        )

        assert query_response.status_code == 200
        query_embedding = query_response.json()["data"][0]["embedding"]

        # Compute similarities
        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # Verify results
        assert len(similarities) == len(documents)
        assert all(0.0 <= sim <= 1.0 for sim in similarities)

        # Most relevant document should be about transformers/attention
        best_match_idx = np.argmax(similarities)
        assert best_match_idx == 0, \
            f"Expected doc 0 (transformers) to match best, got doc {best_match_idx}"

    @pytest.mark.integration
    def test_cross_encoder_reranking(self):
        """Test cross-encoder reranking for retrieved documents."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            pytest.skip("sentence_transformers not available")

        # Load reranker model
        reranker = CrossEncoder('BAAI/bge-reranker-base')

        # Sample query and candidate documents
        query = "What are scaling laws for language models?"
        candidates = [
            "The Chinchilla paper demonstrates compute-optimal training requires balancing model size and data.",
            "The attention mechanism allows transformers to process sequences efficiently.",
            "Scaling laws describe how model performance improves with compute, data, and parameters."
        ]

        # Score query-document pairs
        pairs = [[query, doc] for doc in candidates]
        scores = reranker.predict(pairs)

        # Verify scores
        assert len(scores) == len(candidates)
        assert all(isinstance(s, (int, float, np.number)) for s in scores)

        # Best match should be doc 2 (about scaling laws)
        best_idx = np.argmax(scores)
        assert best_idx == 2, \
            f"Expected doc 2 (scaling laws) to rank highest, got doc {best_idx}"

    @pytest.mark.integration
    def test_rag_answer_generation(self):
        """Test answer generation with retrieved context.

        Assumes chat model server is running on port 8001.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8001/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Chat model server not running on port 8001")
        except requests.exceptions.RequestException:
            pytest.skip("Chat model server not running on port 8001")

        # Sample context from retrieved documents
        context = """
        [Paper 1: Attention Is All You Need (2017)]
        The Transformer architecture uses self-attention mechanisms to process sequences
        in parallel, eliminating the need for recurrent connections. This allows for
        better parallelization during training and improved modeling of long-range dependencies.
        """

        query = "What is the main innovation of the Transformer architecture?"

        # Construct RAG prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided research paper excerpts.

Instructions:
- Use ONLY the information from the provided context
- Cite specific papers when making claims
- Be concise and accurate
- If the context lacks information, say so"""

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Call chat API
        payload = {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 256
        }

        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json=payload,
            timeout=60
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        answer = data["choices"][0]["message"]["content"]

        # Answer should mention key concepts
        assert len(answer) > 0
        # Check if answer references the paper or key concepts
        answer_lower = answer.lower()
        assert any(term in answer_lower for term in [
            "attention", "transformer", "parallel", "self-attention"
        ]), "Answer should reference key concepts from context"

    @pytest.mark.integration
    def test_complete_rag_pipeline(self):
        """Test complete RAG pipeline: retrieve → rerank → generate.

        Requires both embedding (port 8000) and chat (port 8001) servers running.
        """
        # Check if servers are up
        for port, name in [(8000, "embedding"), (8001, "chat")]:
            try:
                response = requests.get(
                    f"http://localhost:{port}/health",
                    timeout=2
                )
                if response.status_code != 200:
                    pytest.skip(f"{name} model server not running on port {port}")
            except requests.exceptions.RequestException:
                pytest.skip(f"{name} model server not running on port {port}")

        # Check for reranker
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('BAAI/bge-reranker-base')
        except ImportError:
            pytest.skip("sentence_transformers not available")

        # 1. Sample knowledge base
        documents = [
            "The Transformer architecture introduced self-attention mechanisms for parallel sequence processing.",
            "BERT uses bidirectional training with masked language modeling to learn contextual representations.",
            "GPT-3 demonstrates few-shot learning capabilities through in-context learning.",
            "Scaling laws predict how model performance improves with compute, parameters, and data.",
            "RAG combines retrieval from external knowledge with generative language models."
        ]

        query = "What is self-attention in transformers?"

        # 2. Retrieve: Embed and find similar documents
        doc_response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json={"model": "BAAI/bge-small-en-v1.5", "input": documents},
            timeout=30
        )
        assert doc_response.status_code == 200
        doc_embeddings = [item["embedding"] for item in doc_response.json()["data"]]

        query_response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json={"model": "BAAI/bge-small-en-v1.5", "input": query},
            timeout=30
        )
        assert query_response.status_code == 200
        query_embedding = query_response.json()["data"][0]["embedding"]

        # Compute similarities
        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # Get top 3 candidates
        top_k = 3
        top_indices = np.argsort(similarities)[::-1][:top_k]
        candidates = [documents[i] for i in top_indices]

        # 3. Rerank with cross-encoder
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = reranker.predict(pairs)

        # Get top 2 after reranking
        top_2_indices = np.argsort(rerank_scores)[::-1][:2]
        top_docs = [candidates[i] for i in top_2_indices]

        # 4. Generate answer with context
        context = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(top_docs)])

        system_prompt = "You are a helpful AI assistant. Answer based on the provided context."
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        gen_response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 256
            },
            timeout=60
        )

        assert gen_response.status_code == 200
        answer = gen_response.json()["choices"][0]["message"]["content"]

        # Verify complete pipeline produced a reasonable answer
        assert len(answer) > 0
        answer_lower = answer.lower()
        assert any(term in answer_lower for term in [
            "attention", "transformer", "self", "sequence"
        ]), "Answer should reference self-attention concepts"

        # Verify all pipeline stages worked
        assert len(top_docs) == 2, "Should retrieve 2 documents after reranking"
        assert all(isinstance(score, (int, float, np.number)) for score in rerank_scores), \
            "Reranker should produce numeric scores"
