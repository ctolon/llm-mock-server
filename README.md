# OpenAI & vLLM Compatible FastAPI Mock Server (2026 SOTA Edition)

A lightweight, flat, memory-safe **OpenAI & vLLM compatible mock FastAPI server** managed with [`uv`](https://docs.astral.sh/uv/). All responses are dynamically generated from curated 2026 State-of-the-Art English `list[dict]` datasets in [`mock_data.py`](mock_data.py).

Designed for local development, CI/CD integration testing, high-concurrency load testing, SDK verification, and Kubernetes/Docker staging environments without requiring GPUs or external API credits.

---

## Key Features

- **100% English 2026 SOTA `list[dict]` Data Generation**: Every response (chat completions, chain-of-thought reasoning, text completions, structured JSON, tool call arguments, audio transcriptions, image prompts, classification labels, fine-tuning events) is selected from structured English `list[dict]` pools in [`mock_data.py`](mock_data.py) covering 2026 frontier topics (disaggregated prefill/decode over RDMA/NIXL, MXFP4/NVFP4 quantization, EAGLE-3/MTP speculative decoding, Multi-head Latent Attention, RLVR, MCP/A2A agent orchestration, and Confidential GPU TEEs).
- **Zero Memory Leaks (`BoundedListStore` & `BoundedDictStore`)**:
  - All in-memory stateful stores (`MOCK_CHAT_COMPLETIONS_STORE`, `MOCK_RESPONSES_STORE`, `MOCK_FILES_STORE`, `MOCK_BATCHES_STORE`, `MOCK_FINE_TUNING_STORE`, `MOCK_UPLOADS_STORE`, `MOCK_VECTOR_STORES`, `MOCK_ASSISTANTS_STORE`, `MOCK_THREADS_STORE`, `MOCK_THREAD_MESSAGES_STORE`, `MOCK_THREAD_RUNS_STORE`, `MOCK_MODELS`) use FIFO/LRU bounded containers (`MOCK_MAX_STORE_SIZE`, default `500`) that automatically evict the oldest entries.
  - File upload payload reads are strictly capped (`MOCK_MAX_FILE_BYTES`, default `64 KB`) and `UploadFile` handles are immediately closed.
  - Generation `n` and embedding `dimensions` are bounded per request, and WAV audio bytes are pre-cached in memory (`CACHED_WAV_BYTES`).
- **Pre-Populated 2026 Frontier Model Lineup**:
  - `gpt-sol-6.1`, `gpt-astra-6`
  - `claude-opus-5.5`, `claude-sonnet-5.5`, `claude-fable-5.1`
  - `gemini-argon-4`, `gemini-flash-3.8`
  - `glm-5.3-flash`, `deepseek-ai/DeepSeek-R2`, `Qwen/Qwen3.5-144B-Instruct`, `meta-llama/Llama-4-405B-Instruct`
  - `text-embedding-4-large`, `BAAI/bge-reranker-v3-m3`, `whisper-v4-turbo`, `tts-2-hd`, `imagen-4-ultra`
  - Plus any arbitrary model ID passed in a request is accepted and echoed dynamically.
- **Full OpenAI & vLLM API Surface (`/v1`, `/v2`, and Unversioned)**:
  - **Generation**: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/invocations`, and `WS /v1/realtime`.
  - **Embeddings & Pooling**: `/v1/embeddings`, `/v2/embeddings`, `/embeddings`, `/pooling`, `/v1/pooling`, `/v2/pooling` (`float` and `base64` formats).
  - **Reranking, Scoring & Classification**: `/v1/rerank`, `/v2/rerank`, `/rerank`, `/v1/score`, `/v2/score`, `/score`, `/classify`, `/v1/classify`, `/v2/classify`.
  - **vLLM Tokenizer & LoRA Management**: `/tokenize`, `/v1/tokenize`, `/v2/tokenize`, `/detokenize`, `/v1/detokenize`, `/v2/detokenize`, `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`.
  - **vLLM Engine, Health, Metrics & Profiler Control**: `/health`, `/ping`, `/version`, `/load`, `/metrics` (Prometheus format), `/server_info`, `/v1/server_info`, `/sleep`, `/wake_up`, `/is_sleeping`, `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`, `/start_profile`, `/stop_profile`, `/scale_elastic_ep`, `/collective_rpc`.
  - **Full OpenAI Platform Suite**: Audio (`/v1/audio/speech`, `/v1/audio/voices`, `/v1/audio/transcriptions`, `/v1/audio/translations`), Images (`/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`), Moderations (`/v1/moderations`), Files (`/v1/files`), Multipart Uploads (`/v1/uploads`), Batches (`/v1/batches`), Fine-Tuning (`/v1/fine_tuning/jobs`), Vector Stores (`/v1/vector_stores`), and Assistants/Threads/Runs (`/v1/assistants`, `/v1/threads`).

---

## Project Structure

```text
.
├── constants.py          # Centralized Constants class (vLLM 0.32.0, model IDs, store bounds, defaults)
├── main.py               # FastAPI application, middleware, and all OpenAI/vLLM routes
├── mock_data.py          # 2026 SOTA English list[dict] datasets, bounded stores, and generators
├── schemas.py            # Pydantic v2 request/response schemas (extra="allow" for vLLM params)
├── test_main.py          # Automated pytest suite (TestClient + official OpenAI SDK + WebSocket + Memory tests)
├── pyproject.toml        # Project metadata and uv dependencies
├── uv.lock               # Deterministic dependency lockfile
├── Dockerfile            # Multi-layer container build using ghcr.io/astral-sh/uv
├── docker-compose.yml    # Docker Compose service configuration
├── kubernetes.yaml       # Kubernetes ConfigMap, Deployment, Service, and HPA manifests
├── Makefile              # Developer convenience commands
├── .env.example          # Sample environment variable configuration
├── .gitignore            # Git ignore rules
└── .dockerignore         # Docker build context ignore rules
```

---

## Quick Start

### Prerequisites
- Python `3.11+` (tested up to Python `3.14`)
- [`uv`](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 1. Install & Run Locally
```bash
# Install dependencies
uv sync

# Start the development server with hot-reload on port 8000
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Or using `make`:
```bash
make install
make dev
```

Interactive OpenAPI documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### 2. Run the Test Suite
```bash
uv run pytest -v
# or
make test
```

---

## Configuration (`Constants`, Environment Variables & HTTP Headers)

All model identifiers, vLLM engine version (`0.32.0`), default endpoint models, embedding dimensions, and store capacity bounds are centralized in the `Constants` class in [`constants.py`](constants.py).

| Environment Variable | Request Header Override | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | — | `8000` | HTTP listening port |
| `MOCK_VLLM_VERSION` | — | `"0.32.0"` | vLLM engine version reported by `/version` and `/server_info` |
| `MOCK_API_KEY` | `Authorization: Bearer <key>` | `""` (disabled) | If set, enforces Bearer token authentication on non-public routes |
| `MOCK_LATENCY_MS` | `X-Mock-Latency-Ms` | `0` | Artificial delay (ms) applied before returning responses |
| `MOCK_STREAM_DELAY_MS` | `X-Mock-Stream-Delay-Ms` | `0` | Artificial delay (ms) between streamed SSE token chunks |
| `MOCK_INCLUDE_REASONING` | — | `"true"` | Controls whether `reasoning_content` is included in chat responses |
| `MOCK_ENABLE_LOGGING` | — | `"true"` | Enables non-blocking `stdout` streaming request logging (no file I/O) |
| `MOCK_LOG_FORMAT` | — | `"json"` | Stdout log format for Docker/Kubernetes (`"json"` or `"text"`) |
| `MOCK_LOG_LEVEL` | — | `"INFO"` | Logging verbosity (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`) |
| `MOCK_ENABLE_OBSERVABILITY` | `X-Request-ID`, `X-Forwarded-For`, `X-Real-IP`, `CF-Connecting-IP` | `"true"` | Injects `X-Request-ID`, `X-Client-IP`, `X-Process-Time-Ms`, and `X-Response-Time` headers |
| `MOCK_MAX_STORE_SIZE` | — | `500` | Maximum items retained in each bounded in-memory store before FIFO/LRU eviction |
| `MOCK_MAX_FILE_BYTES` | — | `65536` | Maximum bytes read/stored per uploaded file to prevent memory bloat |

---

## Complete Endpoint Reference

### 1. Generation & Realtime Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | Chat Completions (sync, SSE `stream=True`, tools, `json_object`/`json_schema`, `guided_json`, `reasoning_content`) |
| `GET` | `/v1/chat/completions` | List stored chat completions |
| `GET` | `/v1/chat/completions/{completion_id}` | Retrieve a stored chat completion |
| `GET` | `/v1/chat/completions/{completion_id}/messages` | List input messages for a stored chat completion |
| `POST` | `/v1/chat/completions/{completion_id}` | Update metadata on a stored chat completion |
| `DELETE` | `/v1/chat/completions/{completion_id}` | Delete a stored chat completion |
| `POST` | `/v1/completions` | Legacy text completions (sync & SSE `stream=True`, `echo`, `suffix`, `logprobs`) |
| `POST` | `/v1/responses` | OpenAI / vLLM Responses API (sync & SSE `stream=True`) |
| `GET` | `/v1/responses/{response_id}` | Retrieve a stored response object |
| `DELETE` | `/v1/responses/{response_id}` | Delete a stored response object |
| `POST` | `/v1/responses/{response_id}/cancel` | Cancel an in-progress response |
| `GET` | `/v1/responses/{response_id}/input_items` | List input items for a response |
| `POST` | `/invocations` | AWS SageMaker / BentoML compatible vLLM inference route |
| `WS` | `/v1/realtime` | OpenAI / vLLM Realtime WebSocket API (`session.created`, `response.text.delta`, `response.done`) |

### 2. Embeddings, Pooling, Reranking, Scoring & Classification
| Method | Endpoint (Unversioned / `v1` / `v2`) | Description |
| :--- | :--- | :--- |
| `POST` | `/embeddings`, `/v1/embeddings`, `/v2/embeddings` | Vector embeddings (`float` list or `base64` float32 bytes, custom `dimensions`) |
| `POST` | `/pooling`, `/v1/pooling`, `/v2/pooling` | vLLM pooling model outputs |
| `POST` | `/rerank`, `/v1/rerank`, `/v2/rerank` | Cross-encoder document reranking sorted by `relevance_score` |
| `POST` | `/score`, `/v1/score`, `/v2/score` | Pairwise text similarity scoring |
| `POST` | `/classify`, `/v1/classify`, `/v2/classify` | Sequence classification label & probability distribution |

### 3. vLLM Tokenizer, Models, LoRA & Engine Control
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/tokenize`, `/v1/tokenize`, `/v2/tokenize` | Tokenize prompt string or chat messages into token IDs |
| `POST` | `/detokenize`, `/v1/detokenize`, `/v2/detokenize` | Detokenize token IDs into an English string |
| `GET` | `/v1/models`, `/v1/models/{model_id}` | List or retrieve available 2026 SOTA OpenAI & vLLM models |
| `DELETE` | `/v1/models/{model_id}` | Delete a fine-tuned or dynamic LoRA model |
| `POST` | `/v1/load_lora_adapter` | Dynamically register a LoRA adapter in `/v1/models` |
| `POST` | `/v1/unload_lora_adapter` | Unload and remove a LoRA adapter from `/v1/models` |
| `GET`, `POST` | `/health`, `/ping` | Health and readiness check (`200 OK`) |
| `GET` | `/version`, `/load`, `/server_info`, `/v1/server_info` | vLLM version (`0.32.0`), active request load, and server metadata |
| `GET` | `/metrics` | Prometheus metrics (`vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`, etc.) |
| `POST`, `GET` | `/sleep`, `/wake_up`, `/is_sleeping` | vLLM engine sleep mode state management |
| `POST` | `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache` | Reset KV prefix and multimodal encoder caches |
| `POST` | `/start_profile`, `/stop_profile` | Start or stop vLLM profiler |
| `POST` | `/scale_elastic_ep`, `/collective_rpc` | vLLM elastic expert parallelism scaling and distributed worker RPC |

### 4. Audio, Images, Moderations, Files, Uploads, Batches, Fine-Tuning, Vector Stores & Assistants
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/audio/speech` | Text-to-speech returning valid RIFF/WAVE PCM audio bytes |
| `GET` | `/v1/audio/voices` | List available TTS voices (`sol`, `astra`, `argon`, `fable`, `alloy`, `nova`) |
| `POST` | `/v1/audio/transcriptions`, `/v1/audio/translations` | Speech-to-text transcription and translation (`json`, `verbose_json`, `text`, `srt`, `vtt`) |
| `POST` | `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations` | Image generation returning URLs or base64 PNGs with English `revised_prompt` |
| `POST` | `/v1/moderations` | Content moderation safety scores |
| `GET`, `POST`, `DELETE` | `/v1/files`, `/v1/files/{id}`, `/v1/files/{id}/content` | File upload, listing, metadata retrieval, raw content download, and deletion |
| `POST` | `/v1/uploads`, `/v1/uploads/{id}/parts`, `/complete`, `/cancel` | Multipart chunked upload lifecycle |
| `GET`, `POST` | `/v1/batches`, `/v1/batches/{id}`, `/v1/batches/{id}/cancel` | Batch processing job creation, inspection, and cancellation |
| `GET`, `POST` | `/v1/fine_tuning/jobs`, `/{id}`, `/{id}/events`, `/{id}/checkpoints`, `/{id}/cancel` | Fine-tuning job management |
| `GET`, `POST`, `DELETE` | `/v1/vector_stores`, `/{id}`, `/{id}/search` | Vector store management and semantic search |
| `GET`, `POST`, `DELETE` | `/v1/assistants`, `/v1/threads`, `/messages`, `/runs` | Assistants API v2 threads, messages, and runs |

---

## Usage Examples

### Official OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    api_key="mock-key",
    base_url="http://localhost:8000/v1",
)

# 1. Standard Chat Completion with Claude Opus 5.5
response = client.chat.completions.create(
    model="claude-opus-5.5",
    messages=[{"role": "user", "content": "How does disaggregated prefill-and-decode serving work?"}],
)
print(response.choices[0].message.content)

# 2. Streaming Chat Completion with Gemini Argon 4
stream = client.chat.completions.create(
    model="gemini-argon-4",
    messages=[{"role": "user", "content": "Explain MXFP4 quantization and EAGLE-3 speculative decoding."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### cURL Examples
```bash
# Chat Completions with GPT-Sol-6.1 and simulated 50ms latency
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Mock-Latency-Ms: 50" \
  -d '{
    "model": "gpt-sol-6.1",
    "messages": [{"role": "user", "content": "Explain confidential GPU TEE attestation."}]
  }'

# vLLM v2 Rerank API with BAAI/bge-reranker-v3-m3
curl -X POST http://localhost:8000/v2/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v3-m3",
    "query": "KV cache RDMA transport optimization",
    "documents": [
      "RLVR trains reasoning models against deterministic compilers.",
      "Disaggregated prefill-and-decode transfers KV blocks over RDMA using NIXL."
    ],
    "top_n": 1
  }'
```

---

## Docker & Kubernetes Deployment

### Docker Compose
```bash
# Build and start in background
make docker-up
# or
docker compose up --build -d

# Stop containers
make docker-down
```

### Kubernetes
```bash
# Build image locally (or push to your registry)
make docker-build

# Deploy ConfigMap, Deployment, Service, and HPA
make k8s-apply
# or
kubectl apply -f kubernetes.yaml

# Remove resources
make k8s-delete
```
