"""
Curated 2026 State-of-the-Art English list[dict] datasets, byte-bounded + optional SQLite
stateful stores, deterministic PRNG helpers, recursive JSON Schema synthesizer, reversible
tokenizer, and lexical reranker/scorer for the OpenAI & vLLM compatible FastAPI mock server.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import sqlite3
import struct
from typing import Any
import uuid

from constants import Constants

# ---------------------------------------------------------------------------
# 0. Byte-Bounded & Optional SQLite-Backed Store Containers
# ---------------------------------------------------------------------------
DEFAULT_MAX_STORE_SIZE = Constants.MAX_STORE_SIZE
DEFAULT_MAX_STORE_BYTES = Constants.MAX_STORE_BYTES
MAX_STORED_FILE_CONTENT_BYTES = Constants.MAX_STORED_FILE_BYTES


def _estimate_bytes(obj: Any) -> int:
    """Estimates the serialized UTF-8 byte footprint of a store entry."""
    try:
        return len(json.dumps(obj, default=str).encode("utf-8"))
    except Exception:
        return 256


def get_active_state_db_path() -> str:
    """Returns the configured SQLite state DB path if enabled via env or Constants."""
    return os.environ.get("MOCK_STATE_DB_PATH", Constants.STATE_DB_PATH).strip()


class _SQLiteStateSync:
    """
    Lightweight process-safe SQLite state synchronizer using stdlib `sqlite3` (WAL mode).
    Used when `MOCK_STATE_DB_PATH` is configured so multiple Kubernetes replicas or
    Gunicorn/Uvicorn workers share stateful resources (files, responses, threads, batches).
    """

    _initialized_paths: set[str] = set()

    @classmethod
    def _ensure_schema(cls, db_path: str) -> None:
        if db_path in cls._initialized_paths and os.path.exists(db_path):
            return
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with contextlib.closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mock_store_state (
                    store_name TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        cls._initialized_paths.add(db_path)

    @classmethod
    def load(cls, store_name: str | None) -> Any | None:
        db_path = get_active_state_db_path()
        if not db_path or not store_name:
            return None
        try:
            cls._ensure_schema(db_path)
            with contextlib.closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
                row = conn.execute(
                    "SELECT payload_json FROM mock_store_state WHERE store_name = ?",
                    (store_name,),
                ).fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        except Exception:
            return None
        return None

    @classmethod
    def save(cls, store_name: str | None, payload: Any) -> None:
        db_path = get_active_state_db_path()
        if not db_path or not store_name:
            return
        try:
            cls._ensure_schema(db_path)
            serialized = json.dumps(payload, default=str)
            with contextlib.closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
                conn.execute(
                    """
                    INSERT INTO mock_store_state (store_name, payload_json)
                    VALUES (?, ?)
                    ON CONFLICT(store_name) DO UPDATE SET payload_json = excluded.payload_json
                    """,
                    (store_name, serialized),
                )
                conn.commit()
        except Exception:
            return


class BoundedListStore(list):
    """
    A list[dict] subclass that enforces strict upper bounds on both item count (`max_size`)
    and total serialized byte footprint (`max_bytes`). Optionally syncs with SQLite when
    `MOCK_STATE_DB_PATH` is set.
    """

    def __init__(
        self,
        initial: list[dict[str, Any]] | None = None,
        max_size: int = Constants.MAX_STORE_SIZE,
        max_bytes: int = Constants.MAX_STORE_BYTES,
        store_name: str | None = None,
    ) -> None:
        super().__init__(initial or [])
        self.max_size = max(1, max_size)
        self.max_bytes = max(1, max_bytes)
        self.store_name = store_name
        self._item_bytes: list[int] = [_estimate_bytes(x) for x in self]
        self.current_bytes: int = sum(self._item_bytes)
        self._trim()

    @property
    def total_bytes(self) -> int:
        return self.current_bytes

    def _sync_from_db(self) -> None:
        loaded = _SQLiteStateSync.load(self.store_name)
        if isinstance(loaded, list):
            super().__init__(loaded)
            self._item_bytes = [_estimate_bytes(x) for x in self]
            self.current_bytes = sum(self._item_bytes)

    def _sync_to_db(self) -> None:
        if self.store_name and get_active_state_db_path():
            _SQLiteStateSync.save(self.store_name, list(self))

    def _trim(self) -> None:
        while len(self) > self.max_size or (len(self) > 1 and self.current_bytes > self.max_bytes):
            super().pop(0)
            if self._item_bytes:
                removed = self._item_bytes.pop(0)
                self.current_bytes = max(0, self.current_bytes - removed)

    def append(self, item: dict[str, Any]) -> None:
        self._sync_from_db()
        super().append(item)
        sz = _estimate_bytes(item)
        self._item_bytes.append(sz)
        self.current_bytes += sz
        self._trim()
        self._sync_to_db()

    def extend(self, iterable: Any) -> None:
        self._sync_from_db()
        items = list(iterable)
        super().extend(items)
        for item in items:
            sz = _estimate_bytes(item)
            self._item_bytes.append(sz)
            self.current_bytes += sz
        self._trim()
        self._sync_to_db()

    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)
        self._item_bytes = [_estimate_bytes(x) for x in self]
        self.current_bytes = sum(self._item_bytes)
        self._trim()
        self._sync_to_db()

    def __iter__(self) -> Any:
        self._sync_from_db()
        return super().__iter__()


class BoundedDictStore(OrderedDict):
    """
    An OrderedDict subclass with strict LRU/FIFO capacity eviction by both entry count
    (`max_size`) and total serialized byte footprint (`max_bytes`). Optionally syncs
    with SQLite when `MOCK_STATE_DB_PATH` is configured.
    """

    def __init__(
        self,
        initial: dict[str, Any] | None = None,
        max_size: int = Constants.MAX_STORE_SIZE,
        max_bytes: int = Constants.MAX_STORE_BYTES,
        store_name: str | None = None,
    ) -> None:
        super().__init__()
        self.max_size = max(1, max_size)
        self.max_bytes = max(1, max_bytes)
        self.store_name = store_name
        self._entry_bytes: dict[str, int] = {}
        self.current_bytes: int = 0
        if initial:
            for k, v in initial.items():
                self[k] = v

    @property
    def total_bytes(self) -> int:
        return self.current_bytes

    def _sync_from_db(self) -> None:
        loaded = _SQLiteStateSync.load(self.store_name)
        if isinstance(loaded, dict):
            super().clear()
            self._entry_bytes.clear()
            self.current_bytes = 0
            for k, v in loaded.items():
                super().__setitem__(k, v)
                sz = _estimate_bytes(v)
                self._entry_bytes[k] = sz
                self.current_bytes += sz

    def _sync_to_db(self) -> None:
        if self.store_name and get_active_state_db_path():
            _SQLiteStateSync.save(self.store_name, dict(self))

    def _trim(self) -> None:
        while len(self) > self.max_size or (len(self) > 1 and self.current_bytes > self.max_bytes):
            oldest_key, _ = self.popitem(last=False)
            removed_sz = self._entry_bytes.pop(oldest_key, 0)
            self.current_bytes = max(0, self.current_bytes - removed_sz)

    def __setitem__(self, key: str, value: Any) -> None:
        self._sync_from_db()
        if key in self:
            self.move_to_end(key)
            self.current_bytes = max(0, self.current_bytes - self._entry_bytes.get(key, 0))
        super().__setitem__(key, value)
        sz = _estimate_bytes(value)
        self._entry_bytes[key] = sz
        self.current_bytes += sz
        self._trim()
        self._sync_to_db()

    def __getitem__(self, key: str) -> Any:
        self._sync_from_db()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._sync_from_db()
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self._sync_from_db()
        return super().get(key, default)

    def pop(self, key: str, *args: Any) -> Any:
        self._sync_from_db()
        val = super().pop(key, *args)
        removed_sz = self._entry_bytes.pop(key, 0)
        self.current_bytes = max(0, self.current_bytes - removed_sz)
        self._sync_to_db()
        return val

    def values(self) -> Any:
        self._sync_from_db()
        return super().values()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._sync_from_db()
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        self[key] = default
        return default


# ---------------------------------------------------------------------------
# 1. Core 2026 SOTA English list[dict] Response Dataset
# ---------------------------------------------------------------------------
MOCK_RESPONSES: list[dict[str, Any]] = [
    {
        "id": "resp_disagg_01",
        "category": "disaggregated_serving",
        "content": (
            "Disaggregated prefill-and-decode serving separates compute-bound prompt processing from "
            "memory-bandwidth-bound token generation across dedicated GPU pools. By transferring KV cache "
            f"blocks over RDMA and NVLink fabrics using NIXL and Mooncake engines, vLLM {Constants.VLLM_VERSION} "
            "eliminates prefill-induced inter-token latency spikes and satisfies strict P99 TPOT service-level objectives."
        ),
        "reasoning_content": (
            "The user is asking about modern high-throughput LLM serving architecture. Let's analyze why "
            "colocating prefill and decode on the same GPU replica causes interference: long-context prefills "
            "monopolize tensor cores and stall ongoing decode batches. Separating prefill nodes from decode "
            "nodes and streaming KV blocks asynchronously over zero-copy RDMA solves this bottleneck."
        ),
        "completion": (
            " while dynamic KV cache routing across disaggregated prefill and decode workers maintains optimal "
            "time-to-first-token (TTFT) and time-per-output-token (TPOT) under bursty multi-tenant traffic."
        ),
        "structured_json": {
            "status": "optimal",
            "category": "disaggregated_serving",
            "architecture": "prefill_decode_disaggregation",
            "vllm_version": Constants.VLLM_VERSION,
            "kv_transport": "NIXL_RDMA_ZeroCopy",
            "recommendations": [
                "Isolate compute-heavy prefill onto high-FLOPs nodes",
                "Route autoregressive decode to high-HBM3e/HBM4 bandwidth pools",
                "Enable prefix-aware KV cache routing",
            ],
            "confidence_score": 0.99,
        },
        "tool_call_arguments": {
            "cluster_topology": "disaggregated-pd-v2",
            "kv_connector": "nixl_rdma",
            "max_prefill_tokens": 131072,
        },
    },
    {
        "id": "resp_quant_02",
        "category": "low_precision_inference",
        "content": (
            "Microscaling FP4 (MXFP4) and NVFP4 quantization formats leverage fine-grained block-wise scaling "
            "factors to preserve frontier model accuracy while halving memory footprint compared to FP8. "
            "Combined with Marlin and CUTLASS fused kernels, sub-4-bit weight-and-activation execution "
            "doubles effective token throughput on next-generation tensor core architectures."
        ),
        "reasoning_content": (
            f"Let's evaluate next-generation quantization techniques for frontier models like {Constants.MODEL_GPT_SOL} and "
            f"{Constants.MODEL_CLAUDE_OPUS}. Standard per-tensor INT4 suffers from outlier degradation. Microscaling formats "
            "(MXFP4 / NVFP4) attach shared E8M0/FP8 scale factors to 16- or 32-element micro-blocks, clamping "
            "activation outliers and achieving near-lossless perplexity at 2x the speed of FP8."
        ),
        "completion": (
            " enabling trillion-parameter mixture-of-experts models to serve 256k context windows within a "
            "single multi-GPU NVLink domain."
        ),
        "structured_json": {
            "status": "verified",
            "category": "low_precision_inference",
            "precision": "MXFP4",
            "block_size": 32,
            "metrics": {"perplexity_delta": 0.012, "throughput_speedup_vs_fp16": 3.85},
            "confidence_score": 0.98,
        },
        "tool_call_arguments": {
            "model": Constants.MODEL_GPT_SOL,
            "quantization": "mxfp4",
            "kv_cache_dtype": "fp8_e4m3",
        },
    },
    {
        "id": "resp_spec_03",
        "category": "speculative_decoding",
        "content": (
            "EAGLE-3 and Multi-Token Prediction (MTP) heads accelerate autoregressive decoding by drafting "
            "multiple candidate continuation tokens in a single forward pass and verifying them in parallel "
            "against the target model distribution. Because verification is mathematically lossless, speculative "
            "decoding achieves 2.8x to 3.5x lower end-to-end latency on code generation and reasoning tasks."
        ),
        "reasoning_content": (
            "Why is standard autoregressive decoding slow? Because each generated token requires reading all "
            "model weights from HBM into SRAM. By using native Multi-Token Prediction (MTP) heads or lightweight "
            "EAGLE-3 feature-level drafters, the target model verifies K draft tokens in one matrix-matrix multiply "
            "instead of K sequential matrix-vector multiplies."
        ),
        "completion": (
            " without altering the output probability distribution of the underlying frontier model."
        ),
        "structured_json": {
            "status": "accelerated",
            "category": "speculative_decoding",
            "method": "MTP_EAGLE3_Hybrid",
            "num_speculative_tokens": 5,
            "mean_acceptance_rate": 0.86,
            "confidence_score": 0.99,
        },
        "tool_call_arguments": {
            "speculative_model": Constants.MODEL_SPECULATIVE_DRAFT,
            "num_speculative_tokens": 5,
            "enable_mtp": True,
        },
    },
    {
        "id": "resp_agent_04",
        "category": "agentic_orchestration",
        "content": (
            "Multi-agent orchestration systems combine the Model Context Protocol (MCP) for standardized tool "
            "discovery with Agent-to-Agent (A2A) state handoffs and deterministic sandboxed execution. "
            "By isolating sub-agent memory scopes and enforcing structured JSON schema contracts at every boundary, "
            "complex autonomous workflows remain auditable, fault-tolerant, and reproducible."
        ),
        "reasoning_content": (
            f"Let's consider how state-of-the-art models like {Constants.MODEL_GEMINI_ARGON} and {Constants.MODEL_CLAUDE_SONNET} coordinate "
            "in multi-agent topologies. Unbounded shared context windows quickly degrade signal-to-noise ratio. "
            "Scoped sub-agent delegation with typed MCP tool schemas and explicit verification gates prevents "
            "compounding hallucinations."
        ),
        "completion": (
            " while stateful checkpointing allows long-horizon reasoning agents to resume seamlessly from "
            "transient tool execution failures."
        ),
        "structured_json": {
            "status": "orchestrated",
            "category": "agentic_orchestration",
            "protocols": ["MCP_2026", "A2A_Stream"],
            "models": [
                Constants.MODEL_GEMINI_ARGON,
                Constants.MODEL_CLAUDE_OPUS,
                Constants.MODEL_GPT_ASTRA,
            ],
            "capabilities": ["parallel_tool_execution", "sandboxed_verification", "context_compaction"],
            "confidence_score": 0.97,
        },
        "tool_call_arguments": {
            "agent_role": "principal_verifier",
            "protocol": "mcp",
            "max_parallel_tools": 8,
        },
    },
    {
        "id": "resp_attn_05",
        "category": "sparse_linear_attention",
        "content": (
            "Multi-head Latent Attention (MLA) and hybrid Lightning/Gated-DeltaNet layers compress key-value "
            "representations into low-rank latent vectors, slashing KV cache memory consumption by over 75%. "
            "Interleaving full softmax attention layers with linear-complexity state-space blocks enables "
            "million-token context reasoning with near-constant per-token decoding cost."
        ),
        "reasoning_content": (
            "Standard Grouped-Query Attention (GQA) still scales linearly with sequence length. Multi-head Latent "
            "Attention (MLA) projects keys and values into a shared compressed latent vector that absorbs RoPE "
            "cleanly, while hybrid Gated-DeltaNet layers maintain fixed-size recurrent state across ultra-long contexts."
        ),
        "completion": (
            " making repository-scale code synthesis and multi-document legal analysis practical at interactive speeds."
        ),
        "structured_json": {
            "status": "compressed",
            "category": "sparse_linear_attention",
            "attention_mechanism": "Hybrid_MLA_GatedDeltaNet",
            "kv_cache_reduction_pct": 78.4,
            "max_context_length": 2097152,
            "confidence_score": 0.98,
        },
        "tool_call_arguments": {
            "model": Constants.MODEL_GLM_FLASH,
            "context_window": 1048576,
            "enable_chunked_prefill": True,
        },
    },
    {
        "id": "resp_rl_06",
        "category": "verifiable_reasoning",
        "content": (
            "Reinforcement Learning with Verifiable Rewards (RLVR) trains frontier reasoning models against "
            "deterministic compilers, formal theorem provers, and unit-test sandboxes rather than heuristic "
            "preference models alone. This grounds test-time compute scaling in objective execution feedback, "
            "dramatically improving accuracy on algorithmic synthesis and formal verification."
        ),
        "reasoning_content": (
            f"Why did RLVR supersede pure RLHF for reasoning models like {Constants.MODEL_GPT_ASTRA} and {Constants.MODEL_CLAUDE_OPUS}? "
            "Because neural reward models can be reward-hacked on complex math and code tasks. Verifiable "
            "environments provide exact binary or graded signals from actual test execution and Lean4 proofs."
        ),
        "completion": (
            " while adaptive test-time compute dynamically allocates thinking tokens based on problem complexity."
        ),
        "structured_json": {
            "status": "verified",
            "category": "verifiable_reasoning",
            "training_paradigm": "RLVR_GRPO",
            "verifiers": ["pytest_sandbox", "lean4_kernel", "ast_static_analyzer"],
            "pass_at_1_gain_pct": 34.2,
            "confidence_score": 0.99,
        },
        "tool_call_arguments": {
            "verifier": "lean4_and_pytest",
            "reasoning_effort": "high",
            "budget_tokens": 16384,
        },
    },
    {
        "id": "resp_sec_07",
        "category": "confidential_ai_security",
        "content": (
            "Confidential GPU computing with hardware-rooted Trusted Execution Environments (TEEs) and "
            "end-to-end encrypted NVLink fabrics ensures that sensitive prompts, model weights, and KV caches "
            "remain cryptographically isolated in memory. Remote attestation tokens verify enclave integrity "
            "before any client payload is decrypted inside the inference worker."
        ),
        "reasoning_content": (
            "Enterprise deployments of frontier models require strict zero-trust guarantees even against cloud "
            "host operators. Hardware memory encryption on H100/B200/R100 TEEs combined with SPDM attestation "
            "guarantees that plaintext weights and user prompts never leave the secure GPU boundary."
        ),
        "completion": (
            " satisfying zero-data-retention and sovereign cloud compliance mandates for regulated industries."
        ),
        "structured_json": {
            "status": "attested",
            "category": "confidential_ai_security",
            "enclave_mode": "GPU_CC_TEE_Strict",
            "attestation_protocol": "IETF_RATS_EAT",
            "controls": ["encrypted_hbm", "encrypted_nvlink", "ephemeral_session_keys"],
            "confidence_score": 0.97,
        },
        "tool_call_arguments": {
            "security_profile": "confidential-gpu-tee",
            "verify_attestation": True,
            "region": "eu-central-1",
        },
    },
    {
        "id": "resp_data_08",
        "category": "multimodal_rag_2026",
        "content": (
            "Late-interaction ColPali/ColQwen visual-document embeddings and hybrid sparse-dense vector indexes "
            "allow retrieval pipelines to index complex PDFs, schematics, and tables directly from page patches "
            f"without brittle OCR preprocessing. A cross-encoder reranker (`{Constants.MODEL_RERANKER}`) then scores "
            "the top candidates before feeding grounded citations to the generator."
        ),
        "reasoning_content": (
            "Traditional OCR-plus-chunking pipelines lose spatial layout, charts, and multi-column table relationships. "
            "By encoding page images into multi-vector MaxSim representations and combining them with lexical BM25 "
            "and dense embeddings, retrieval accuracy on technical manuals improves substantially."
        ),
        "completion": (
            " eliminating layout-induced parsing errors and preserving exact visual coordinates for inline citations."
        ),
        "structured_json": {
            "status": "indexed",
            "category": "multimodal_rag_2026",
            "embedding_model": Constants.MODEL_EMBEDDING,
            "reranker_model": Constants.MODEL_RERANKER,
            "scoring_operator": "MaxSim_Hybrid_RRF",
            "top_k": 10,
            "confidence_score": 0.98,
        },
        "tool_call_arguments": {
            "query": "disaggregated KV cache RDMA latency benchmarks",
            "top_k": 10,
            "reranker": Constants.MODEL_RERANKER,
        },
    },
]

# ---------------------------------------------------------------------------
# 2. State-of-the-Art 2026 Mock Models Dataset (BoundedListStore of dicts)
# ---------------------------------------------------------------------------
MOCK_MODELS: BoundedListStore = BoundedListStore(
    [
        {
            "id": Constants.MODEL_GPT_SOL,
            "object": "model",
            "created": 1775000000,
            "owned_by": "openai",
            "root": Constants.MODEL_GPT_SOL,
            "parent": None,
            "max_model_len": 1048576,
        },
        {
            "id": Constants.MODEL_GPT_ASTRA,
            "object": "model",
            "created": 1774500000,
            "owned_by": "openai",
            "root": Constants.MODEL_GPT_ASTRA,
            "parent": None,
            "max_model_len": 1048576,
        },
        {
            "id": Constants.MODEL_CLAUDE_OPUS,
            "object": "model",
            "created": 1774000000,
            "owned_by": "anthropic",
            "root": Constants.MODEL_CLAUDE_OPUS,
            "parent": None,
            "max_model_len": 1000000,
        },
        {
            "id": Constants.MODEL_CLAUDE_SONNET,
            "object": "model",
            "created": 1774100000,
            "owned_by": "anthropic",
            "root": Constants.MODEL_CLAUDE_SONNET,
            "parent": None,
            "max_model_len": 1000000,
        },
        {
            "id": Constants.MODEL_CLAUDE_FABLE,
            "object": "model",
            "created": 1773500000,
            "owned_by": "anthropic",
            "root": Constants.MODEL_CLAUDE_FABLE,
            "parent": None,
            "max_model_len": 500000,
        },
        {
            "id": Constants.MODEL_GLM_FLASH,
            "object": "model",
            "created": 1773800000,
            "owned_by": "zhipu-vllm",
            "root": Constants.MODEL_GLM_FLASH,
            "parent": None,
            "max_model_len": 1048576,
        },
        {
            "id": Constants.MODEL_GEMINI_FLASH,
            "object": "model",
            "created": 1774200000,
            "owned_by": "google-deepmind",
            "root": Constants.MODEL_GEMINI_FLASH,
            "parent": None,
            "max_model_len": 2097152,
        },
        {
            "id": Constants.MODEL_GEMINI_ARGON,
            "object": "model",
            "created": 1775500000,
            "owned_by": "google-deepmind",
            "root": Constants.MODEL_GEMINI_ARGON,
            "parent": None,
            "max_model_len": 2097152,
        },
        {
            "id": Constants.MODEL_DEEPSEEK_R2,
            "object": "model",
            "created": 1773000000,
            "owned_by": "vllm",
            "root": Constants.MODEL_DEEPSEEK_R2,
            "parent": None,
            "max_model_len": 262144,
        },
        {
            "id": Constants.MODEL_QWEN_INSTRUCT,
            "object": "model",
            "created": 1773200000,
            "owned_by": "vllm",
            "root": Constants.MODEL_QWEN_INSTRUCT,
            "parent": None,
            "max_model_len": 262144,
        },
        {
            "id": Constants.MODEL_LLAMA_INSTRUCT,
            "object": "model",
            "created": 1772800000,
            "owned_by": "vllm",
            "root": Constants.MODEL_LLAMA_INSTRUCT,
            "parent": None,
            "max_model_len": 524288,
        },
        {
            "id": Constants.MODEL_EMBEDDING,
            "object": "model",
            "created": 1772000000,
            "owned_by": "openai",
            "root": Constants.MODEL_EMBEDDING,
            "parent": None,
            "max_model_len": 32768,
        },
        {
            "id": Constants.MODEL_RERANKER,
            "object": "model",
            "created": 1772100000,
            "owned_by": "vllm",
            "root": Constants.MODEL_RERANKER,
            "parent": None,
            "max_model_len": 32768,
        },
        {
            "id": Constants.MODEL_WHISPER,
            "object": "model",
            "created": 1771500000,
            "owned_by": "openai",
            "root": Constants.MODEL_WHISPER,
            "parent": None,
            "max_model_len": 1024,
        },
        {
            "id": Constants.MODEL_TTS,
            "object": "model",
            "created": 1771600000,
            "owned_by": "openai",
            "root": Constants.MODEL_TTS,
            "parent": None,
            "max_model_len": 8192,
        },
        {
            "id": Constants.MODEL_IMAGE,
            "object": "model",
            "created": 1771700000,
            "owned_by": "system",
            "root": Constants.MODEL_IMAGE,
            "parent": None,
            "max_model_len": 8192,
        },
    ],
    store_name="models",
)

# ---------------------------------------------------------------------------
# 3. English Audio Transcriptions & Translations Dataset (list[dict])
# ---------------------------------------------------------------------------
MOCK_TRANSCRIPTIONS: list[dict[str, Any]] = [
    {
        "text": (
            f"Disaggregated prefill and decode clusters running {Constants.MODEL_GPT_SOL} and {Constants.MODEL_CLAUDE_OPUS} "
            "achieved sub-ten-millisecond inter-token latency across all regional availability zones."
        ),
        "language": "english",
        "duration": 5.84,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 5.84,
                "text": (
                    f" Disaggregated prefill and decode clusters running {Constants.MODEL_GPT_SOL} and {Constants.MODEL_CLAUDE_OPUS} "
                    "achieved sub-ten-millisecond inter-token latency across all regional availability zones."
                ),
                "tokens": [50364, 440, 14883, 11523, 7365, 439, 10862, 11342, 293, 18942, 264, 50635],
                "temperature": 0.0,
                "avg_logprob": -0.09,
                "compression_ratio": 1.24,
                "no_speech_prob": 0.004,
            }
        ],
    },
    {
        "text": (
            "Microscaling FP4 quantization with EAGLE-3 speculative decoding doubled effective token "
            f"throughput on {Constants.MODEL_GEMINI_ARGON} and {Constants.MODEL_GLM_FLASH} serving pools."
        ),
        "language": "english",
        "duration": 6.12,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 6.12,
                "text": (
                    " Microscaling FP4 quantization with EAGLE-3 speculative decoding doubled effective token "
                    f"throughput on {Constants.MODEL_GEMINI_ARGON} and {Constants.MODEL_GLM_FLASH} serving pools."
                ),
                "tokens": [50364, 28940, 6032, 278, 293, 3028, 292, 3202, 10622, 5407, 50673],
                "temperature": 0.0,
                "avg_logprob": -0.08,
                "compression_ratio": 1.19,
                "no_speech_prob": 0.003,
            }
        ],
    },
    {
        "text": (
            "All confidential GPU enclave attestation checks passed prior to initializing the "
            f"{Constants.MODEL_CLAUDE_SONNET} and {Constants.MODEL_GPT_ASTRA} inference workers."
        ),
        "language": "english",
        "duration": 4.95,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 4.95,
                "text": (
                    " All confidential GPU enclave attestation checks passed prior to initializing the "
                    f"{Constants.MODEL_CLAUDE_SONNET} and {Constants.MODEL_GPT_ASTRA} inference workers."
                ),
                "tokens": [50364, 1057, 4562, 26541, 1585, 29540, 6902, 2726, 25012, 50606],
                "temperature": 0.0,
                "avg_logprob": -0.10,
                "compression_ratio": 1.16,
                "no_speech_prob": 0.002,
            }
        ],
    },
]

# ---------------------------------------------------------------------------
# 4. English Image Generation Dataset (list[dict])
# ---------------------------------------------------------------------------
MOCK_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

MOCK_IMAGE_PROMPTS: list[dict[str, Any]] = [
    {
        "url": "https://mock-cdn.local/images/exascale-nvlink-cluster-2026.png",
        "b64_json": MOCK_TINY_PNG_B64,
        "revised_prompt": (
            "An ultra-detailed wide-angle architectural visualization of a 2026 exascale liquid-cooled "
            "AI supercluster interconnected by photonic optical waveguides and cyan NVLink backplanes."
        ),
    },
    {
        "url": "https://mock-cdn.local/images/disaggregated-kv-mesh-2026.png",
        "b64_json": MOCK_TINY_PNG_B64,
        "revised_prompt": (
            "A precision 3D isometric diagram illustrating disaggregated prefill and decode GPU nodes "
            "exchanging luminous low-rank latent attention tensors over a zero-copy RDMA fabric."
        ),
    },
    {
        "url": "https://mock-cdn.local/images/autonomous-agent-topology-2026.png",
        "b64_json": MOCK_TINY_PNG_B64,
        "revised_prompt": (
            "A futuristic holographic interface displaying a hierarchical multi-agent reasoning graph "
            f"coordinating {Constants.MODEL_GEMINI_ARGON}, {Constants.MODEL_CLAUDE_OPUS}, and {Constants.MODEL_GPT_SOL} nodes in real time."
        ),
    },
]

# ---------------------------------------------------------------------------
# 5. English Classification Labels Dataset (list[dict]) for vLLM /classify
# ---------------------------------------------------------------------------
MOCK_CLASSIFICATIONS: list[dict[str, Any]] = [
    {"label": "deep_reasoning_routing", "probs": [0.88, 0.08, 0.04]},
    {"label": "fast_flash_completion", "probs": [0.09, 0.85, 0.06]},
    {"label": "agentic_tool_invocation", "probs": [0.05, 0.11, 0.84]},
]

# ---------------------------------------------------------------------------
# 6. Memory-Bounded Stateful Stores (with optional SQLite store_name sync)
# ---------------------------------------------------------------------------
MOCK_FILES_STORE: BoundedListStore = BoundedListStore(
    [
        {
            "id": Constants.SEED_FILE_EVAL_ID,
            "object": "file",
            "bytes": 2048,
            "created_at": 1775000000,
            "filename": Constants.SEED_FILE_EVAL_NAME,
            "purpose": "batch",
            "status": "processed",
            "content": (
                '{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", '
                f'"body": {{"model": "{Constants.MODEL_GPT_SOL}", "messages": [{{"role": "user", "content": "Explain MXFP4 quantization."}}]}}}}\n'
            ),
        },
        {
            "id": Constants.SEED_FILE_RLVR_ID,
            "object": "file",
            "bytes": 4096,
            "created_at": 1775003600,
            "filename": Constants.SEED_FILE_RLVR_NAME,
            "purpose": "fine-tune",
            "status": "processed",
            "content": (
                '{"messages": [{"role": "system", "content": "You are a principal distributed systems architect."}, '
                '{"role": "user", "content": "How does disaggregated prefill-decode serving work?"}, '
                '{"role": "assistant", "content": "It separates compute-bound prefill from bandwidth-bound decode over RDMA."}]}\n'
            ),
        },
    ],
    store_name="files",
)

MOCK_BATCHES_STORE: BoundedListStore = BoundedListStore(
    [
        {
            "id": Constants.SEED_BATCH_ID,
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "errors": None,
            "input_file_id": Constants.SEED_FILE_EVAL_ID,
            "completion_window": "24h",
            "status": "completed",
            "output_file_id": Constants.SEED_FILE_RLVR_ID,
            "error_file_id": None,
            "created_at": 1775000100,
            "in_progress_at": 1775000120,
            "expires_at": 1775086500,
            "finalizing_at": 1775000500,
            "completed_at": 1775000600,
            "failed_at": None,
            "expired_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "request_counts": {"total": 50, "completed": 50, "failed": 0},
            "metadata": {"environment": "production-2026", "model": Constants.MODEL_CLAUDE_OPUS},
        }
    ],
    store_name="batches",
)

MOCK_RESPONSES_STORE: BoundedDictStore = BoundedDictStore(store_name="responses")

MOCK_CHAT_COMPLETIONS_STORE: BoundedDictStore = BoundedDictStore(store_name="chat_completions")

MOCK_VOICES: list[dict[str, Any]] = [
    {"voice_id": "sol", "name": "Sol", "language": "en-US", "gender": "neutral"},
    {"voice_id": "astra", "name": "Astra", "language": "en-US", "gender": "female"},
    {"voice_id": "argon", "name": "Argon", "language": "en-US", "gender": "male"},
    {"voice_id": "fable", "name": "Fable", "language": "en-GB", "gender": "neutral"},
    {"voice_id": "alloy", "name": "Alloy", "language": "en-US", "gender": "neutral"},
    {"voice_id": "nova", "name": "Nova", "language": "en-US", "gender": "female"},
]

MOCK_FINE_TUNING_STORE: BoundedListStore = BoundedListStore(
    [
        {
            "id": Constants.SEED_FT_JOB_ID,
            "object": "fine_tuning.job",
            "created_at": 1775001000,
            "finished_at": 1775004600,
            "model": Constants.MODEL_GLM_FLASH,
            "fine_tuned_model": f"ft:{Constants.MODEL_GLM_FLASH}:mock-org::rlvr2026",
            "organization_id": Constants.ORGANIZATION_ID,
            "result_files": [Constants.SEED_FILE_RLVR_ID],
            "status": "succeeded",
            "validation_file": None,
            "training_file": Constants.SEED_FILE_RLVR_ID,
            "hyperparameters": {"n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": "auto"},
            "trained_tokens": 256800,
            "error": {},
            "seed": 42,
        }
    ],
    store_name="fine_tuning",
)

MOCK_UPLOADS_STORE: BoundedDictStore = BoundedDictStore(store_name="uploads")

MOCK_VECTOR_STORES: BoundedListStore = BoundedListStore(
    [
        {
            "id": Constants.SEED_VECTOR_STORE_ID,
            "object": "vector_store",
            "created_at": 1775002000,
            "name": "2026 Frontier AI & Systems Architecture Corpus",
            "usage_bytes": 131072,
            "file_counts": {"in_progress": 0, "completed": 2, "failed": 0, "cancelled": 0, "total": 2},
            "status": "completed",
            "last_active_at": 1775005000,
            "metadata": {"domain": "disaggregated_serving_and_rlvr", "language": "en"},
        }
    ],
    store_name="vector_stores",
)

MOCK_ASSISTANTS_STORE: BoundedListStore = BoundedListStore(
    [
        {
            "id": Constants.SEED_ASSISTANT_ID,
            "object": "assistant",
            "created_at": 1775003000,
            "name": "2026 Frontier Principal Systems Architect",
            "description": "Provides guidance on disaggregated vLLM serving, MXFP4 quantization, and multi-agent MCP workflows.",
            "model": Constants.MODEL_CLAUDE_OPUS,
            "instructions": "Always respond in concise, authoritative technical English.",
            "tools": [{"type": "file_search"}, {"type": "code_interpreter"}],
            "metadata": {"team": "frontier_inference"},
            "temperature": 1.0,
            "top_p": 1.0,
        }
    ],
    store_name="assistants",
)

MOCK_THREADS_STORE: BoundedDictStore = BoundedDictStore(store_name="threads")
MOCK_THREAD_MESSAGES_STORE: BoundedDictStore = BoundedDictStore(store_name="thread_messages")
MOCK_THREAD_RUNS_STORE: BoundedDictStore = BoundedDictStore(store_name="thread_runs")


# ---------------------------------------------------------------------------
# 7. Deterministic PRNG, Reversible Tokenizer, JSON Schema Synthesizer & Scorer
# ---------------------------------------------------------------------------
def get_rng(seed: int | None = None, payload_key: str | None = None) -> random.Random:
    """
    Returns a deterministic `random.Random` instance when `seed` is provided or
    `MOCK_DETERMINISTIC=true` is active; otherwise returns an unseeded `random.Random`.
    """
    deterministic_env = os.environ.get("MOCK_DETERMINISTIC", str(Constants.DETERMINISTIC_MODE)).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if seed is not None or deterministic_env:
        digest_input = f"seed={seed if seed is not None else 0}|key={payload_key or ''}"
        seed_int = int.from_bytes(hashlib.sha256(digest_input.encode("utf-8")).digest()[:8], "big")
        return random.Random(seed_int)
    return random.Random()


def get_random_response_dict(rng: random.Random | None = None) -> dict[str, Any]:
    """Selects an English response dictionary from MOCK_RESPONSES using optional deterministic RNG."""
    active_rng = rng or random
    return active_rng.choice(MOCK_RESPONSES)


def get_random_transcription_dict(rng: random.Random | None = None) -> dict[str, Any]:
    """Selects an English transcription dictionary from MOCK_TRANSCRIPTIONS."""
    active_rng = rng or random
    return active_rng.choice(MOCK_TRANSCRIPTIONS)


def get_random_image_dict(rng: random.Random | None = None) -> dict[str, Any]:
    """Selects an English image prompt dictionary from MOCK_IMAGE_PROMPTS."""
    active_rng = rng or random
    return active_rng.choice(MOCK_IMAGE_PROMPTS)


def get_random_classification_dict(rng: random.Random | None = None) -> dict[str, Any]:
    """Selects a classification dictionary from MOCK_CLASSIFICATIONS."""
    active_rng = rng or random
    return active_rng.choice(MOCK_CLASSIFICATIONS)


def estimate_tokens(text: str) -> int:
    """Roughly estimates token count for English text."""
    if not text:
        return 1
    words = text.split()
    return max(1, int(len(words) * 1.3))


# Reversible bidirectional vocabulary table for /tokenize and /detokenize
_TOKEN_ID_TO_STR: BoundedDictStore = BoundedDictStore(max_size=20000)
_STR_TO_TOKEN_ID: BoundedDictStore = BoundedDictStore(max_size=20000)


def _seed_reversible_vocab() -> None:
    """Pre-populates common vocabulary tokens from MOCK_RESPONSES so arbitrary IDs also decode."""
    for entry in MOCK_RESPONSES:
        pieces = re.findall(r"\s*\S+|\s+", entry["content"])
        for piece in pieces:
            digest = int.from_bytes(hashlib.sha256(piece.encode("utf-8")).digest()[:4], "big")
            tok_id = (digest % 90000) + 1000
            _TOKEN_ID_TO_STR[str(tok_id)] = piece
            _STR_TO_TOKEN_ID[piece] = tok_id


_seed_reversible_vocab()


def simple_tokenize(text: str) -> list[int]:
    """
    Reversible deterministic tokenizer converting English text to integer token IDs.
    Preserves exact whitespace and punctuation pieces in a bounded lookup table so
    `simple_detokenize(simple_tokenize(text)) == text` round-trips losslessly.
    """
    if not text:
        return []
    pieces = re.findall(r"\s*\S+|\s+", text)
    tokens: list[int] = []
    for piece in pieces:
        existing = _STR_TO_TOKEN_ID.get(piece)
        if existing is not None:
            tokens.append(int(existing))
            continue
        digest = int.from_bytes(hashlib.sha256(piece.encode("utf-8")).digest()[:4], "big")
        tok_id = (digest % 90000) + 1000
        # Resolve rare hash collision if different string occupies tok_id
        while str(tok_id) in _TOKEN_ID_TO_STR and _TOKEN_ID_TO_STR[str(tok_id)] != piece:
            tok_id += 1
        _TOKEN_ID_TO_STR[str(tok_id)] = piece
        _STR_TO_TOKEN_ID[piece] = tok_id
        tokens.append(tok_id)
    return tokens


def simple_detokenize(token_ids: list[int]) -> str:
    """
    Reconstructs the exact original text for token IDs produced by `simple_tokenize`,
    and falls back to deterministic vocabulary words for arbitrary unseen integer IDs.
    """
    if not token_ids:
        return ""
    fallback_words = MOCK_RESPONSES[0]["content"].split()
    out_pieces: list[str] = []
    for idx, tid in enumerate(token_ids):
        if tid == Constants.BOS_TOKEN_ID:
            continue
        mapped = _TOKEN_ID_TO_STR.get(str(tid))
        if mapped is not None:
            out_pieces.append(str(mapped))
        else:
            word = fallback_words[abs(tid) % len(fallback_words)]
            out_pieces.append(word if idx == 0 or not out_pieces else f" {word}")
    return "".join(out_pieces)


def synthesize_from_json_schema(
    schema: dict[str, Any] | None,
    fallback_obj: dict[str, Any] | None = None,
    root_schema: dict[str, Any] | None = None,
    rng: random.Random | None = None,
    depth: int = 0,
) -> Any:
    """
    Recursively synthesizes a valid JSON value conforming to a user-provided JSON Schema
    (`response_format.json_schema.schema`, `guided_json`, or tool `parameters`).
    """
    active_rng = rng or random
    fallback = fallback_obj or MOCK_RESPONSES[0]["structured_json"]
    if not schema or not isinstance(schema, dict) or depth > 8:
        return fallback

    root = root_schema or schema

    # 1. Resolve local $ref (#/$defs/Name or #/definitions/Name)
    if "$ref" in schema and isinstance(schema["$ref"], str):
        ref_path = schema["$ref"]
        if ref_path.startswith("#/"):
            target: Any = root
            for part in ref_path.lstrip("#/").split("/"):
                if isinstance(target, dict):
                    target = target.get(part)
                else:
                    target = None
                    break
            if isinstance(target, dict):
                return synthesize_from_json_schema(target, fallback, root, active_rng, depth + 1)

    # 2. Handle const and enum
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return active_rng.choice(schema["enum"])

    # 3. Handle anyOf / oneOf / allOf
    for combiner in ("anyOf", "oneOf"):
        if combiner in schema and isinstance(schema[combiner], list) and schema[combiner]:
            non_null = [s for s in schema[combiner] if isinstance(s, dict) and s.get("type") != "null"]
            chosen_sub = non_null[0] if non_null else schema[combiner][0]
            return synthesize_from_json_schema(chosen_sub, fallback, root, active_rng, depth + 1)

    if "allOf" in schema and isinstance(schema["allOf"], list) and schema["allOf"]:
        merged: dict[str, Any] = {"type": "object", "properties": {}}
        for sub in schema["allOf"]:
            if isinstance(sub, dict) and isinstance(sub.get("properties"), dict):
                merged["properties"].update(sub["properties"])
        return synthesize_from_json_schema(merged, fallback, root, active_rng, depth + 1)

    # 4. Determine primitive or composite type
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), "string")
    if not schema_type:
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"
        else:
            schema_type = "string"

    if schema_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return fallback if depth == 0 else {"key": "value"}
        result_obj: dict[str, Any] = {}
        for prop_name, prop_spec in properties.items():
            if not isinstance(prop_spec, dict):
                result_obj[prop_name] = "sample_value"
                continue
            if prop_name in fallback and not any(k in prop_spec for k in ("properties", "items", "enum", "const", "$ref")):
                val_candidate = fallback[prop_name]
                expected_t = prop_spec.get("type")
                if (
                    (expected_t == "string" and isinstance(val_candidate, str))
                    or (expected_t == "integer" and isinstance(val_candidate, int))
                    or (expected_t == "number" and isinstance(val_candidate, (int, float)))
                    or (expected_t == "boolean" and isinstance(val_candidate, bool))
                ):
                    result_obj[prop_name] = val_candidate
                    continue
            result_obj[prop_name] = _synthesize_named_property(prop_name, prop_spec, fallback, root, active_rng, depth + 1)
        return result_obj

    if schema_type == "array":
        item_schema = schema.get("items")
        min_items = int(schema.get("minItems", 1))
        max_items = int(schema.get("maxItems", max(2, min_items)))
        count = max(min_items, min(2, max_items))
        if isinstance(item_schema, dict):
            return [
                synthesize_from_json_schema(item_schema, fallback, root, active_rng, depth + 1)
                for _ in range(count)
            ]
        return ["item_alpha", "item_beta"][:count]

    if schema_type == "integer":
        low = int(schema.get("minimum", 1))
        high = int(schema.get("maximum", max(low + 100, 42)))
        return max(low, min(42, high))

    if schema_type == "number":
        low_f = float(schema.get("minimum", 0.0))
        high_f = float(schema.get("maximum", max(low_f + 1.0, 1.0)))
        return round(max(low_f, min(0.98, high_f)), 4)

    if schema_type == "boolean":
        return True

    if schema_type == "null":
        return None

    return schema.get("default", "sample_value")


def _synthesize_named_property(
    prop_name: str,
    prop_spec: dict[str, Any],
    fallback: dict[str, Any],
    root: dict[str, Any],
    rng: random.Random,
    depth: int,
) -> Any:
    prop_type = prop_spec.get("type")
    if isinstance(prop_type, list):
        prop_type = next((t for t in prop_type if t != "null"), "string")

    if (
        prop_type in ("object", "array", "integer", "number", "boolean", "null")
        or "enum" in prop_spec
        or "const" in prop_spec
        or "$ref" in prop_spec
        or "anyOf" in prop_spec
        or "oneOf" in prop_spec
    ):
        return synthesize_from_json_schema(prop_spec, fallback, root, rng, depth)

    lower_name = prop_name.lower()
    if "city" in lower_name or "location" in lower_name:
        return "San Francisco, CA"
    if "query" in lower_name or "search" in lower_name:
        return str(fallback.get("query", "disaggregated prefill decode architecture"))
    if "model" in lower_name:
        return Constants.DEFAULT_CHAT_MODEL
    if "status" in lower_name:
        return "optimal"
    if "name" in lower_name or "title" in lower_name:
        return f"mock_{prop_name}"
    return f"mock_{prop_name}_value"


def synthesize_tool_arguments(
    parameters_schema: dict[str, Any] | None,
    fallback_args: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    Generates realistic English arguments matching a JSON Schema `parameters` object
    when provided, falling back to `fallback_args` from MOCK_RESPONSES.
    """
    if not parameters_schema or not isinstance(parameters_schema, dict):
        return fallback_args
    synthesized = synthesize_from_json_schema(parameters_schema, fallback_args, parameters_schema, rng)
    return synthesized if isinstance(synthesized, dict) else fallback_args


def build_tool_calls(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    sample_entry: dict[str, Any],
    parallel_tool_calls: bool | None = True,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """
    Builds OpenAI-compatible tool_calls list if tools are supplied and tool_choice
    is not 'none'. Supports multiple parallel tool calls when `parallel_tool_calls` is True.
    """
    if not tools or tool_choice == "none":
        return []

    active_rng = rng or random
    selected_tools: list[dict[str, Any]] = []

    if isinstance(tool_choice, dict):
        target_name = (
            tool_choice.get("function", {}).get("name")
            or tool_choice.get("name")
        )
        if target_name:
            for t in tools:
                fn = t.get("function", t)
                if fn.get("name") == target_name:
                    selected_tools = [t]
                    break
        if not selected_tools:
            selected_tools = [tools[0]]
    elif parallel_tool_calls is False or len(tools) == 1:
        selected_tools = [tools[0]]
    else:
        selected_tools = tools[: Constants.MAX_PARALLEL_TOOL_CALLS]

    built_calls: list[dict[str, Any]] = []
    for idx, chosen_tool in enumerate(selected_tools):
        fn_spec = chosen_tool.get("function", chosen_tool)
        fn_name = fn_spec.get("name", f"mock_function_{idx}")
        fn_params = fn_spec.get("parameters")
        args_dict = synthesize_tool_arguments(fn_params, sample_entry["tool_call_arguments"], active_rng)
        call_hex = f"{active_rng.getrandbits(96):024x}" if rng is not None else uuid.uuid4().hex[:24]
        built_calls.append(
            {
                "id": f"call_{call_hex}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(args_dict),
                },
            }
        )
    return built_calls


def compute_relevance_score(query: str, document: str) -> float:
    """
    Computes a deterministic lexical + character 3-gram similarity score in [0.05, 0.99]
    so `/v1/rerank` and `/v1/score` rank genuinely matching documents higher than unrelated ones.
    """
    q_clean = query.strip().lower()
    d_clean = document.strip().lower()
    if not q_clean or not d_clean:
        return 0.05
    if q_clean == d_clean:
        return 0.99

    q_tokens = set(re.findall(r"[a-z0-9]+", q_clean))
    d_tokens = set(re.findall(r"[a-z0-9]+", d_clean))
    token_overlap = (len(q_tokens & d_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0

    def _trigrams(s: str) -> set[str]:
        padded = f"  {s}  "
        return {padded[i : i + 3] for i in range(max(1, len(padded) - 2))}

    q_tri = _trigrams(q_clean)
    d_tri = _trigrams(d_clean)
    jaccard_tri = len(q_tri & d_tri) / max(1, len(q_tri | d_tri))

    substring_bonus = 0.25 if q_clean in d_clean else 0.0
    raw_score = 0.10 + (0.55 * token_overlap) + (0.30 * jaccard_tri) + substring_bonus
    return round(max(0.05, min(0.99, raw_score)), 4)


def generate_embedding_vector(
    dimensions: int = Constants.DEFAULT_EMBEDDING_DIMENSIONS,
    encoding_format: str = "float",
    rng: random.Random | None = None,
) -> list[float] | str:
    """
    Generates a normalized unit-length embedding vector in either `float` list
    or `base64` encoded float32 bytes. Bounds maximum dimension to Constants.MAX_EMBEDDING_DIMENSIONS
    and supports deterministic seeding via `rng`.
    """
    active_rng = rng or random
    safe_dims = max(1, min(dimensions, Constants.MAX_EMBEDDING_DIMENSIONS))
    raw = [active_rng.uniform(-1.0, 1.0) for _ in range(safe_dims)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    normalized = [round(x / norm, 6) for x in raw]

    if encoding_format == "base64":
        packed = struct.pack(f"<{safe_dims}f", *normalized)
        return base64.b64encode(packed).decode("ascii")
    return normalized


# Pre-compute default WAV bytes once at module load to avoid repeated allocations
def generate_wav_bytes(
    duration_ms: int = Constants.DEFAULT_WAV_DURATION_MS,
    sample_rate: int = Constants.DEFAULT_WAV_SAMPLE_RATE,
) -> bytes:
    """Generates a valid minimal RIFF/WAVE PCM audio file in memory."""
    safe_duration_ms = max(10, min(duration_ms, 1000))
    num_samples = max(1, int(sample_rate * (safe_duration_ms / 1000.0)))
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * block_align

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for i in range(num_samples):
        sample_val = int(4000.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        buf.write(struct.pack("<h", sample_val))
    return buf.getvalue()


CACHED_WAV_BYTES: bytes = generate_wav_bytes()
