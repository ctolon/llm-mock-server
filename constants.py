"""
Centralized Constants configuration class for the OpenAI & vLLM Compatible Mock Server.
Modify model identifiers, vLLM engine version, default dimensions, logging format,
or memory bounds here to propagate changes across the entire server, schemas, datasets, and test suite.
"""

from __future__ import annotations

import os


class Constants:
    """Single source of truth for versions, model names, defaults, logging, security, and resource limits."""

    # -----------------------------------------------------------------------
    # 1. Server & vLLM Engine Version Metadata
    # -----------------------------------------------------------------------
    VLLM_VERSION: str = os.environ.get("MOCK_VLLM_VERSION", "0.32.0")
    APP_VERSION: str = f"{VLLM_VERSION}.mock"
    APP_TITLE: str = "OpenAI & vLLM Compatible Mock Server"
    APP_DESCRIPTION: str = (
        "Full-featured 2026 SOTA OpenAI & vLLM compatible mock FastAPI server "
        "powered by English list[dict] datasets."
    )
    DEFAULT_HOST: str = os.environ.get("HOST", "0.0.0.0")
    DEFAULT_PORT: int = int(os.environ.get("PORT", "8000"))

    ORGANIZATION_ID: str = "org-mock-frontier-2026"
    SYSTEM_FINGERPRINT_CHAT: str = "fp_mock_vllm_openai_2026"
    SYSTEM_FINGERPRINT_COMPLETION: str = "fp_mock_vllm_cmpl_2026"

    # -----------------------------------------------------------------------
    # 2. 2026 State-of-the-Art Model Identifiers
    # -----------------------------------------------------------------------
    # OpenAI Frontier Series
    MODEL_GPT_SOL: str = "gpt-sol-6.1"
    MODEL_GPT_SOL_REALTIME: str = "gpt-sol-6.1-realtime"
    MODEL_GPT_ASTRA: str = "gpt-astra-6"

    # Anthropic Claude 5.x Series
    MODEL_CLAUDE_OPUS: str = "claude-opus-5.5"
    MODEL_CLAUDE_SONNET: str = "claude-sonnet-5.5"
    MODEL_CLAUDE_FABLE: str = "claude-fable-5.1"

    # Zhipu GLM 5.x Series
    MODEL_GLM_FLASH: str = "glm-5.3-flash"

    # Google DeepMind Gemini Series
    MODEL_GEMINI_FLASH: str = "gemini-flash-3.8"
    MODEL_GEMINI_ARGON: str = "gemini-argon-4"

    # Open-Weights vLLM Frontier Models
    MODEL_DEEPSEEK_R2: str = "deepseek-ai/DeepSeek-R2"
    MODEL_QWEN_INSTRUCT: str = "Qwen/Qwen3.5-144B-Instruct"
    MODEL_LLAMA_INSTRUCT: str = "meta-llama/Llama-4-405B-Instruct"

    # Embeddings, Reranking, Audio, Vision & Moderation Models
    MODEL_EMBEDDING: str = "text-embedding-4-large"
    MODEL_RERANKER: str = "BAAI/bge-reranker-v3-m3"
    MODEL_WHISPER: str = "whisper-v4-turbo"
    MODEL_TTS: str = "tts-2-hd"
    MODEL_IMAGE: str = "imagen-4-ultra"
    MODEL_MODERATION: str = "omni-moderation-2026"
    MODEL_SPECULATIVE_DRAFT: str = "eagle3-draft-v2"

    # -----------------------------------------------------------------------
    # 3. Default Endpoint Model & Voice Bindings
    # -----------------------------------------------------------------------
    DEFAULT_CHAT_MODEL: str = MODEL_GPT_SOL
    DEFAULT_COMPLETION_MODEL: str = MODEL_QWEN_INSTRUCT
    DEFAULT_RESPONSES_MODEL: str = MODEL_GPT_SOL
    DEFAULT_EMBEDDING_MODEL: str = MODEL_EMBEDDING
    DEFAULT_POOLING_MODEL: str = MODEL_RERANKER
    DEFAULT_SCORE_MODEL: str = MODEL_RERANKER
    DEFAULT_RERANK_MODEL: str = MODEL_RERANKER
    DEFAULT_CLASSIFY_MODEL: str = MODEL_RERANKER
    DEFAULT_TOKENIZE_MODEL: str = MODEL_GPT_SOL
    DEFAULT_SPEECH_MODEL: str = MODEL_TTS
    DEFAULT_TRANSCRIPTION_MODEL: str = MODEL_WHISPER
    DEFAULT_IMAGE_MODEL: str = MODEL_IMAGE
    DEFAULT_MODERATION_MODEL: str = MODEL_MODERATION
    DEFAULT_FINE_TUNE_MODEL: str = MODEL_GLM_FLASH
    DEFAULT_ASSISTANT_MODEL: str = MODEL_CLAUDE_OPUS
    DEFAULT_REALTIME_MODEL: str = MODEL_GPT_SOL_REALTIME
    DEFAULT_VOICE: str = "sol"

    # Tuple of all core SOTA model IDs registered in /v1/models
    ALL_SOTA_MODEL_IDS: tuple[str, ...] = (
        MODEL_GPT_SOL,
        MODEL_GPT_ASTRA,
        MODEL_CLAUDE_OPUS,
        MODEL_CLAUDE_SONNET,
        MODEL_CLAUDE_FABLE,
        MODEL_GLM_FLASH,
        MODEL_GEMINI_FLASH,
        MODEL_GEMINI_ARGON,
        MODEL_DEEPSEEK_R2,
        MODEL_QWEN_INSTRUCT,
        MODEL_LLAMA_INSTRUCT,
        MODEL_EMBEDDING,
        MODEL_RERANKER,
        MODEL_WHISPER,
        MODEL_TTS,
        MODEL_IMAGE,
    )

    # -----------------------------------------------------------------------
    # 4. Memory-Safe Store Bounds, Optional SQLite State & Resource Caps
    # -----------------------------------------------------------------------
    MAX_STORE_SIZE: int = int(os.environ.get("MOCK_MAX_STORE_SIZE", "500"))
    MAX_STORE_BYTES: int = int(os.environ.get("MOCK_MAX_STORE_BYTES", str(16 * 1024 * 1024)))
    MAX_THREAD_SUBSTORE_SIZE: int = 100
    MAX_STORED_FILE_BYTES: int = int(os.environ.get("MOCK_MAX_FILE_BYTES", "65536"))
    MAX_CHOICES_N: int = 16
    MAX_BATCH_INPUT_ITEMS: int = 32
    MAX_STORED_INPUT_MESSAGES: int = 64
    MAX_PARALLEL_TOOL_CALLS: int = 4

    # Optional SQLite file path for multi-replica / multi-worker state consistency
    STATE_DB_PATH: str = os.environ.get("MOCK_STATE_DB_PATH", "").strip()

    # Deterministic mode: when true, identical inputs always yield identical outputs even without `seed`
    DETERMINISTIC_MODE: bool = os.environ.get("MOCK_DETERMINISTIC", "false").lower() in ("1", "true", "yes", "on")

    DEFAULT_EMBEDDING_DIMENSIONS: int = 1536
    DEFAULT_POOLING_DIMENSIONS: int = 128
    MAX_EMBEDDING_DIMENSIONS: int = 8192

    DEFAULT_MAX_MODEL_LEN: int = 128000
    DEFAULT_LORA_MAX_MODEL_LEN: int = 32768
    BOS_TOKEN_ID: int = 128000

    DEFAULT_WAV_DURATION_MS: int = 100
    DEFAULT_WAV_SAMPLE_RATE: int = 16000

    # -----------------------------------------------------------------------
    # 5. Seed Store Identifiers
    # -----------------------------------------------------------------------
    SEED_FILE_EVAL_ID: str = "file-mock2026evalprompts"
    SEED_FILE_EVAL_NAME: str = "sota_2026_eval_prompts.jsonl"
    SEED_FILE_RLVR_ID: str = "file-mock2026rlvrdata"
    SEED_FILE_RLVR_NAME: str = "rlvr_verifiable_traces.jsonl"
    SEED_BATCH_ID: str = "batch_mock2026completed"
    SEED_FT_JOB_ID: str = "ftjob-mock2026rlvr01"
    SEED_VECTOR_STORE_ID: str = "vs_mock2026frontierdocs"
    SEED_ASSISTANT_ID: str = "asst_mock2026architect"

    # -----------------------------------------------------------------------
    # 6. Stdout Streaming Logging, Real-IP, Response Time, Chaos & Security
    # -----------------------------------------------------------------------
    LOGGER_NAME: str = "openai_vllm_mock"
    ENABLE_LOGGING: bool = os.environ.get("MOCK_ENABLE_LOGGING", "true").lower() in ("1", "true", "yes", "on")
    LOG_FORMAT: str = os.environ.get("MOCK_LOG_FORMAT", "json").lower()
    LOG_LEVEL: str = os.environ.get("MOCK_LOG_LEVEL", "INFO").upper()

    ENABLE_OBSERVABILITY_MIDDLEWARE: bool = os.environ.get("MOCK_ENABLE_OBSERVABILITY", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    CORS_ALLOW_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.environ.get("MOCK_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    HEADER_REQUEST_ID: str = "X-Request-ID"
    HEADER_CORRELATION_ID: str = "X-Correlation-ID"
    HEADER_PROCESS_TIME_MS: str = "X-Process-Time-Ms"
    HEADER_RESPONSE_TIME: str = "X-Response-Time"
    HEADER_CLIENT_IP: str = "X-Client-IP"

    # Chaos / Fault Injection Headers
    HEADER_MOCK_ERROR_STATUS: str = "X-Mock-Error-Status"
    HEADER_MOCK_ERROR_RATE: str = "X-Mock-Error-Rate"
    HEADER_MOCK_ERROR_MESSAGE: str = "X-Mock-Error-Message"
    HEADER_MOCK_MALFORMED_SSE: str = "X-Mock-Malformed-SSE"
    DEFAULT_ERROR_RATE: float = float(os.environ.get("MOCK_ERROR_RATE", "0.0"))

    # Priority order for real client IP extraction behind reverse proxies / K8s Ingress / Cloudflare
    REAL_IP_HEADERS: tuple[str, ...] = (
        "CF-Connecting-IP",
        "True-Client-IP",
        "X-Real-IP",
        "X-Forwarded-For",
        "X-Client-IP",
    )
