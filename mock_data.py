"""
Curated 2026 State-of-the-Art English list[dict] datasets, bounded memory-safe stores,
and generator utilities for the OpenAI & vLLM compatible FastAPI mock server.
All model names, engine versions, and store bounds are sourced from `Constants`.
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
import struct
from collections import OrderedDict
from typing import Any

from constants import Constants

# ---------------------------------------------------------------------------
# 0. Memory-Safe Bounded Store Containers (Prevents OOM / Memory Leaks)
# ---------------------------------------------------------------------------
DEFAULT_MAX_STORE_SIZE = Constants.MAX_STORE_SIZE
MAX_STORED_FILE_CONTENT_BYTES = Constants.MAX_STORED_FILE_BYTES


class BoundedListStore(list):
    """
    A list[dict] subclass that enforces a strict upper bound on element count.
    When new items are appended beyond `max_size`, the oldest items
    are automatically evicted in FIFO order to prevent unbounded memory growth.
    """

    def __init__(self, initial: list[dict[str, Any]] | None = None, max_size: int = Constants.MAX_STORE_SIZE) -> None:
        super().__init__(initial or [])
        self.max_size = max(10, max_size)
        self._trim()

    def _trim(self) -> None:
        while len(self) > self.max_size:
            super().pop(0)

    def append(self, item: dict[str, Any]) -> None:
        super().append(item)
        self._trim()

    def extend(self, iterable: Any) -> None:
        super().extend(iterable)
        self._trim()


class BoundedDictStore(OrderedDict):
    """
    An OrderedDict subclass with strict LRU/FIFO capacity eviction.
    Guarantees O(1) bounded memory usage even under continuous load testing.
    """

    def __init__(self, initial: dict[str, Any] | None = None, max_size: int = Constants.MAX_STORE_SIZE) -> None:
        super().__init__()
        self.max_size = max(10, max_size)
        if initial:
            for k, v in initial.items():
                self[k] = v

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.max_size:
            self.popitem(last=False)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
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
    ]
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
# 6. Memory-Bounded Stateful Stores (Initialized with 2026 SOTA English dicts)
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
    ]
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
    ]
)

MOCK_RESPONSES_STORE: BoundedDictStore = BoundedDictStore()

MOCK_CHAT_COMPLETIONS_STORE: BoundedDictStore = BoundedDictStore()

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
    ]
)

MOCK_UPLOADS_STORE: BoundedDictStore = BoundedDictStore()

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
    ]
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
    ]
)

MOCK_THREADS_STORE: BoundedDictStore = BoundedDictStore()
MOCK_THREAD_MESSAGES_STORE: BoundedDictStore = BoundedDictStore()
MOCK_THREAD_RUNS_STORE: BoundedDictStore = BoundedDictStore()


# ---------------------------------------------------------------------------
# 7. Generator & Helper Functions
# ---------------------------------------------------------------------------
def get_random_response_dict() -> dict[str, Any]:
    """Selects a random English response dictionary from MOCK_RESPONSES."""
    return random.choice(MOCK_RESPONSES)


def get_random_transcription_dict() -> dict[str, Any]:
    """Selects a random English transcription dictionary from MOCK_TRANSCRIPTIONS."""
    return random.choice(MOCK_TRANSCRIPTIONS)


def get_random_image_dict() -> dict[str, Any]:
    """Selects a random English image prompt dictionary from MOCK_IMAGE_PROMPTS."""
    return random.choice(MOCK_IMAGE_PROMPTS)


def get_random_classification_dict() -> dict[str, Any]:
    """Selects a random classification dictionary from MOCK_CLASSIFICATIONS."""
    return random.choice(MOCK_CLASSIFICATIONS)


def estimate_tokens(text: str) -> int:
    """Roughly estimates token count for English text."""
    if not text:
        return 1
    words = text.split()
    return max(1, int(len(words) * 1.3))


def simple_tokenize(text: str) -> list[int]:
    """Deterministic lightweight tokenizer converting English text to token IDs."""
    if not text:
        return []
    tokens: list[int] = []
    parts = text.split(" ")
    for idx, part in enumerate(parts):
        piece = part if idx == 0 else " " + part
        if piece:
            token_id = (sum(ord(ch) * (i + 1) for i, ch in enumerate(piece)) % 50000) + 256
            tokens.append(token_id)
    return tokens


def simple_detokenize(token_ids: list[int]) -> str:
    """Produces a clean English sentence from token IDs."""
    if not token_ids:
        return ""
    sample = get_random_response_dict()["content"]
    words = sample.split()
    selected = [words[i % len(words)] for i in range(len(token_ids))]
    return " ".join(selected)


def synthesize_tool_arguments(parameters_schema: dict[str, Any] | None, fallback_args: dict[str, Any]) -> dict[str, Any]:
    """
    Generates realistic English arguments matching a JSON Schema `parameters` object
    when provided, falling back to `fallback_args` from MOCK_RESPONSES.
    """
    if not parameters_schema or not isinstance(parameters_schema, dict):
        return fallback_args

    properties = parameters_schema.get("properties")
    if not properties or not isinstance(properties, dict):
        return fallback_args

    synthesized: dict[str, Any] = {}
    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            synthesized[prop_name] = "sample_value"
            continue

        if "enum" in prop_spec and isinstance(prop_spec["enum"], list) and prop_spec["enum"]:
            synthesized[prop_name] = random.choice(prop_spec["enum"])
            continue

        prop_type = prop_spec.get("type", "string")
        if isinstance(prop_type, list):
            prop_type = next((t for t in prop_type if t != "null"), "string")

        if prop_type == "string":
            if "city" in prop_name.lower() or "location" in prop_name.lower():
                synthesized[prop_name] = "San Francisco, CA"
            elif "query" in prop_name.lower() or "search" in prop_name.lower():
                synthesized[prop_name] = fallback_args.get("query", "disaggregated prefill decode architecture")
            elif "model" in prop_name.lower():
                synthesized[prop_name] = Constants.DEFAULT_CHAT_MODEL
            else:
                synthesized[prop_name] = f"mock_{prop_name}_value"
        elif prop_type == "integer":
            synthesized[prop_name] = 42
        elif prop_type == "number":
            synthesized[prop_name] = 0.98
        elif prop_type == "boolean":
            synthesized[prop_name] = True
        elif prop_type == "array":
            synthesized[prop_name] = ["item_alpha", "item_beta"]
        elif prop_type == "object":
            synthesized[prop_name] = {"key": "value"}
        else:
            synthesized[prop_name] = "sample_value"

    return synthesized


def build_tool_calls(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    sample_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Builds OpenAI-compatible tool_calls list if tools are supplied and tool_choice
    is not 'none'.
    """
    import uuid

    if not tools or tool_choice == "none":
        return []

    chosen_tool = tools[0]
    if isinstance(tool_choice, dict):
        target_name = (
            tool_choice.get("function", {}).get("name")
            or tool_choice.get("name")
        )
        if target_name:
            for t in tools:
                fn = t.get("function", t)
                if fn.get("name") == target_name:
                    chosen_tool = t
                    break

    fn_spec = chosen_tool.get("function", chosen_tool)
    fn_name = fn_spec.get("name", "mock_function")
    fn_params = fn_spec.get("parameters")
    args_dict = synthesize_tool_arguments(fn_params, sample_entry["tool_call_arguments"])

    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps(args_dict),
            },
        }
    ]


def generate_embedding_vector(
    dimensions: int = Constants.DEFAULT_EMBEDDING_DIMENSIONS,
    encoding_format: str = "float",
) -> list[float] | str:
    """
    Generates a normalized unit-length embedding vector in either `float` list
    or `base64` encoded float32 bytes. Bounds maximum dimension to Constants.MAX_EMBEDDING_DIMENSIONS
    to prevent excessive per-request memory allocation.
    """
    safe_dims = max(1, min(dimensions, Constants.MAX_EMBEDDING_DIMENSIONS))
    raw = [random.uniform(-1.0, 1.0) for _ in range(safe_dims)]
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
