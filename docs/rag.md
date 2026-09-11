# Retrieval-Augmented Generation (RAG)

Guide to grounding local models in your own documents with Razer AIKit.

## Overview

A language model can only answer from what is baked into its weights. Ask it about your internal
design docs, a paper published last week, or a contract it has never seen, and it will either refuse
or invent something plausible.

Retrieval-Augmented Generation fixes this without retraining anything. At query time the system
searches a collection of your documents, pulls out the passages most relevant to the question, and
places them in the model's prompt as context. The model then answers from that supplied text rather
than from memory.

This buys three things:

- **Grounded answers** - responses trace back to specific passages you can cite and verify.
- **No training required** - adding, updating, or removing knowledge means re-indexing documents, not
  fine-tuning a model. Compare this with [fine-tuning](fine-tuning.md), which teaches a model new
  *behaviour* rather than new *facts*.
- **Full privacy** - with AIKit every stage runs on your own hardware. Embeddings, the vector store,
  reranking, and generation are all local, so document contents never leave the machine.

This guide covers:

- **How the pipeline works** - the retrieval stages and the models that serve them
- **Quick setup with Open WebUI** - a no-code RAG pipeline in a few commands
- **Troubleshooting** - the configuration pitfalls worth knowing about
- **Building a custom pipeline** - when to drop down to the notebook instead

---

## How the pipeline works

Every RAG system, including the one Open WebUI provides, runs the same five stages:

1. **Chunk** - split each document into passages small enough to embed. Chunks overlap slightly so a
   sentence spanning a boundary is not lost to both sides.
2. **Embed** - convert each chunk into a vector that encodes its meaning, using an embedding model.
3. **Store** - write those vectors to a vector database, alongside the chunk text and its metadata.
4. **Retrieve** - embed the user's question the same way, find the nearest chunk vectors, then
   **rerank** the shortlist with a cross-encoder. Reranking matters: embedding search compares the
   question and chunk independently and is fast but approximate, while a cross-encoder reads both
   together and scores them directly. Retrieving broadly and then reranking down gives noticeably
   better context than embedding search alone.
5. **Generate** - hand the top-ranked chunks to the chat model as context and let it answer.

Stages 2, 4, and 5 each need a model. AIKit serves all three simultaneously as
OpenAI-compatible endpoints:

| Port | Model | Role |
| --- | --- | --- |
| 8000 | `BAAI/bge-small-en-v1.5` | Embeddings - 384 dimensions, hard-capped at 512 input tokens |
| 8001 | `Qwen/Qwen3.5-4B` | Chat - generates the final answer from retrieved context |
| 8002 | `BAAI/bge-reranker-base` | Cross-encoder - rescores retrieved chunks for relevance |

This layout is the same whether you drive the pipeline through Open WebUI or through
[`notebooks/11_RAG_Pipeline.ipynb`](../notebooks/11_RAG_Pipeline.ipynb), so servers started for one
are reusable by the other.

---

## Quick setup with Open WebUI

Open WebUI ships a complete RAG implementation - upload documents, attach them to a chat, done. The
AIKit stack pre-configures it to use your local models rather than any cloud service.

### 1. Start the three models

Each command runs in its own terminal, or use `-d` style backgrounding if you prefer. The
`--gpu-memory-utilization` values are chosen to let all three coexist on one GPU: the two small BGE
models need very little, leaving the bulk for the chat model.

**Embedding model** - port 8000 (the default):

```bash
rzr-aikit model run BAAI/bge-small-en-v1.5 --gpu-memory-utilization 0.1
```

**Chat model** - port 8001:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --port 8001 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 180000 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

The large `--max-model-len` leaves room for many retrieved chunks in the prompt. Thinking mode is
disabled because RAG answers should come from the supplied context, not from extended reasoning.

**Reranker model** - port 8002:

```bash
rzr-aikit model run BAAI/bge-reranker-base --port 8002 --gpu-memory-utilization 0.1
```

Confirm all three servers are actually responding before continuing. Note that
`rzr-aikit model list` shows *cached* models rather than running ones, so query the endpoints
directly:

```bash
for port in 8000 8001 8002; do
  echo "port $port:"
  curl -s "http://localhost:$port/v1/models" | head -c 200
  echo
done
```

If your GPU cannot host all three at once, see [GPU Compatibility](gpu-compatibility.md). You can
also run without reranking by removing the `RAG_RERANKING_ENGINE` and `RAG_EXTERNAL_RERANKER_*`
variables from the configuration below, at some cost to retrieval quality.

### 2. Review the Open WebUI configuration

The `openwebui` service in `docker_compose/docker-compose.yaml` already ships wired for this setup.
You only need to change it if your ports or models differ from the table above.

The RAG-relevant environment variables:

```yaml
# Chat → port 8001
- OPENAI_API_BASE_URL=http://localhost:8001/v1

# Embeddings → port 8000
- RAG_EMBEDDING_ENGINE=openai
- RAG_OPENAI_API_BASE_URL=http://localhost:8000/v1
- RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
- RAG_EMBEDDING_BATCH_SIZE=32            # default 1 = one round trip per chunk

# Reranking → port 8002 (full path required)
- RAG_RERANKING_ENGINE=external
- RAG_EXTERNAL_RERANKER_URL=http://localhost:8002/v1/rerank
- RAG_RERANKING_MODEL=BAAI/bge-reranker-base

# Retrieval and storage
- VECTOR_DB=chroma
- RAG_TOP_K=10                           # chunks retrieved by embedding search
- RAG_TOP_K_RERANKER=5                   # chunks kept after reranking
- ENABLE_RAG_HYBRID_SEARCH=true          # semantic + BM25 keyword search

# Chunking
- RAG_TEXT_SPLITTER=token_transformers
- RAG_TOKENIZER_MODEL=BAAI/bge-small-en-v1.5
- CHUNK_SIZE=450                         # tokens, leaves margin under 512
- CHUNK_OVERLAP=50                       # tokens (~11%)
```

Two of these are worth understanding before you change anything:

**`RAG_TEXT_SPLITTER=token_transformers`** - chunks are measured in the *embedding model's own
tokens*, not characters. `bge-small-en-v1.5` hard-caps at 512 tokens and vLLM rejects longer input
with HTTP 400 rather than truncating it, so a single oversized chunk fails its entire embedding
batch. Character-based sizing is unsafe here: 1000 characters of prose is roughly 250 tokens, but
1000 characters of tables or formulas can blow past 512. `CHUNK_SIZE=450` tokens keeps a deliberate
margin under the cap.

**`ENABLE_PERSISTENT_CONFIG=False`** - this makes the compose file the single source of truth.
Without it, Open WebUI seeds its internal `config` database table on first boot, and those rows then
shadow every environment variable above for the entire life of the `openwebui-data` volume - editing
the compose file would appear to do nothing.

### 3. Start the container

From the repository root:

```bash
docker compose -f docker_compose/docker-compose.yaml up openwebui -d
```

The service uses `network_mode: host`, which is why it reaches your model servers on `localhost`.

### 4. Build a knowledge base

Open `http://localhost:1919`. Local development sets `WEBUI_AUTH=false`, so there is no login step.

1. Go to `Workspace → Knowledge → Create` and fill in a name and description.
2. Upload your documents with `+ → Upload files`. Open WebUI chunks and embeds them as they arrive -
   watch the model server logs on port 8000 to see the embedding requests.
3. Start a new chat, then `+ → Attach knowledge` and select the collection.

Chat queries now retrieve from your documents automatically, and responses cite the passages they
drew on.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Config edits have no effect | Open WebUI's persisted config is shadowing the compose file. Confirm `ENABLE_PERSISTENT_CONFIG=False`, then reset state with `docker compose -f docker_compose/docker-compose.yaml down && docker volume rm openwebui-data`. |
| HTTP 400 from port 8000 during upload | A chunk exceeds the embedder's 512-token limit. Keep `RAG_TEXT_SPLITTER=token_transformers` and lower `CHUNK_SIZE`. |
| 404 from the reranker | `RAG_EXTERNAL_RERANKER_URL` needs the full path, including `/v1/rerank` - a base URL alone will not resolve. |
| Answers ignore your documents | The knowledge collection is not attached to that chat. Re-attach with `+ → Attach knowledge`; attachment is per-conversation. |
| Retrieval returns nothing relevant | Raise `RAG_TOP_K` to retrieve more candidates before reranking, and confirm the reranker on port 8002 is actually running. |
| Out-of-memory starting a model | The three `--gpu-memory-utilization` fractions must sum below 1.0. See [GPU Compatibility](gpu-compatibility.md). |

---

## Building a custom pipeline

Open WebUI is the fast path, and for question-answering over a document set it is usually enough. But
its stages are fixed: you configure them through environment variables, you do not replace them.

Reach for [`notebooks/11_RAG_Pipeline.ipynb`](../notebooks/11_RAG_Pipeline.ipynb) when you need
control over the pipeline itself. The notebook builds the same five stages in Python, one at a time,
and lets you change any of them:

- **Document parsing** - uses `docling` to preserve document structure such as tables and headings,
  instead of flattening everything to plain text
- **Chunking strategy** - swap in any LangChain text splitter, or write your own boundary rules
- **Metadata filtering** - restrict retrieval by source, date, section, or any field you attach
- **Retrieval and reranking depth** - tune the shortlist size and compare results with and against
  reranking directly
- **Prompt construction** - decide exactly how retrieved context is formatted and instructed
- **Direct vector store access** - query ChromaDB yourself, inspect embeddings, debug what was
  actually retrieved

It runs against the same three model servers described above, so anything already running from the
Open WebUI walkthrough can be reused as-is.

---

## Additional Resources

- **RAG Notebook** - code-first custom pipeline: [`notebooks/11_RAG_Pipeline.ipynb`](../notebooks/11_RAG_Pipeline.ipynb)
- **Semantic Search Notebook** - embeddings and vector search on their own: [`notebooks/9_Semantic_Search.ipynb`](../notebooks/9_Semantic_Search.ipynb)
- **Inference Guide** - running and tuning the model servers: [inferencing.md](inferencing.md)
- **Fine-Tuning Guide** - teaching a model new behaviour rather than new facts: [fine-tuning.md](fine-tuning.md)
- **CLI Reference** - complete command documentation: [cli-reference.md](cli-reference.md)
- **Setup Guide** - installation and environment setup: [setup.md](setup.md)
- **Known Issues** - common problems and solutions: [known-issues.md](known-issues.md)
- **Open WebUI Documentation** - upstream project docs: https://docs.openwebui.com
