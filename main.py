"""
OpenAI & vLLM Compatible Latest Mock FastAPI Server.
Generates responses dynamically from curated English list[dict] datasets in mock_data.py.
All versions, model names, and resource bounds are sourced from `Constants`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import random
import sys
import time
from typing import Annotated, Any
import uuid

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from constants import Constants
from mock_data import (
    BoundedListStore,
    CACHED_WAV_BYTES,
    MAX_STORED_FILE_CONTENT_BYTES,
    MOCK_ASSISTANTS_STORE,
    MOCK_BATCHES_STORE,
    MOCK_CHAT_COMPLETIONS_STORE,
    MOCK_FILES_STORE,
    MOCK_FINE_TUNING_STORE,
    MOCK_MODELS,
    MOCK_RESPONSES_STORE,
    MOCK_THREAD_MESSAGES_STORE,
    MOCK_THREAD_RUNS_STORE,
    MOCK_THREADS_STORE,
    MOCK_UPLOADS_STORE,
    MOCK_VECTOR_STORES,
    MOCK_VOICES,
    build_tool_calls,
    compute_relevance_score,
    estimate_tokens,
    generate_embedding_vector,
    get_active_state_db_path,
    get_random_classification_dict,
    get_random_image_dict,
    get_random_response_dict,
    get_random_transcription_dict,
    get_rng,
    simple_detokenize,
    simple_tokenize,
    synthesize_from_json_schema,
)
from schemas import (
    AssistantCreateRequest,
    AudioSpeechRequest,
    BatchCreateRequest,
    ChatCompletionRequest,
    ClassifyRequest,
    CollectiveRpcRequest,
    CompletionRequest,
    DetokenizeRequest,
    EmbeddingRequest,
    FineTuningJobCreateRequest,
    ImageGenerationRequest,
    LoadLoraAdapterRequest,
    ModerationRequest,
    PoolingRequest,
    RerankRequest,
    ResponsesRequest,
    ScaleElasticEPRequest,
    ScoreRequest,
    ThreadCreateRequest,
    ThreadMessageCreateRequest,
    ThreadRunCreateRequest,
    TokenizeRequest,
    UnloadLoraAdapterRequest,
    UploadCompleteRequest,
    UploadCreateRequest,
    VectorStoreCreateRequest,
    VectorStoreSearchRequest,
)


# ---------------------------------------------------------------------------
# 0. Container-Native Stdout Streaming Logger (JSON or Text, No File I/O)
# ---------------------------------------------------------------------------
class StdoutStreamingFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON or human-readable text for Docker
    and Kubernetes stdout log collectors (FluentBit, Vector, Promtail, Datadog).
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        log_fmt = os.environ.get("MOCK_LOG_FORMAT", Constants.LOG_FORMAT).strip().lower()
        meta: dict[str, Any] = getattr(record, "structured_data", {})

        if log_fmt == "json":
            payload: dict[str, Any] = {
                "timestamp": ts,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "vllm_version": Constants.VLLM_VERSION,
                **meta,
            }
            return json.dumps(payload, separators=(",", ":"))

        req_id = meta.get("request_id", "-")
        client_ip = meta.get("client_ip", "-")
        return f"{ts} | {record.levelname} | request_id={req_id} | client_ip={client_ip} | {record.getMessage()}"


def configure_stdout_logger() -> logging.Logger:
    """Configures a non-blocking stdout stream logger with zero file handlers."""
    log = logging.getLogger(Constants.LOGGER_NAME)
    level_name = os.environ.get("MOCK_LOG_LEVEL", Constants.LOG_LEVEL).upper()
    log.setLevel(getattr(logging, level_name, logging.INFO))
    log.propagate = False
    if not log.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(StdoutStreamingFormatter())
        log.addHandler(handler)
    return log


logger = configure_stdout_logger()


def _is_logging_enabled() -> bool:
    raw = os.environ.get("MOCK_ENABLE_LOGGING")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return Constants.ENABLE_LOGGING


def _is_observability_enabled() -> bool:
    raw = os.environ.get("MOCK_ENABLE_OBSERVABILITY")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return Constants.ENABLE_OBSERVABILITY_MIDDLEWARE


def extract_real_client_ip(request: Request) -> str:
    """
    Extracts the originating client IP behind reverse proxies, Kubernetes Ingress,
    Cloudflare, or AWS ALB using Constants.REAL_IP_HEADERS priority order.
    """
    for header_name in Constants.REAL_IP_HEADERS:
        raw_val = request.headers.get(header_name)
        if raw_val:
            first_hop = raw_val.split(",")[0].strip()
            if first_hop:
                return first_hop

    forwarded = request.headers.get("Forwarded")
    if forwarded:
        for part in forwarded.split(";"):
            part_clean = part.strip()
            if part_clean.lower().startswith("for="):
                val = part_clean[4:].split(",")[0].strip().strip('"').strip("[]")
                if val:
                    return val

    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def extract_or_generate_request_id(request: Request) -> str:
    """Reads X-Request-ID / X-Correlation-ID from incoming headers or generates a new ID."""
    incoming = (
        request.headers.get(Constants.HEADER_REQUEST_ID)
        or request.headers.get(Constants.HEADER_CORRELATION_ID)
    )
    if incoming and incoming.strip():
        return incoming.strip()[:128]
    return f"req_{uuid.uuid4().hex[:24]}"


# Server runtime state for vLLM control endpoints
SERVER_STATE: dict[str, Any] = {
    "is_sleeping": False,
    "sleep_level": 0,
    "server_load": 0,
    "total_requests": 0,
    "prefix_cache_resets": 0,
    "mm_cache_resets": 0,
    "is_profiling": False,
    "data_parallel_size": 1,
    "vllm_version": Constants.VLLM_VERSION,
}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Modern FastAPI lifespan handler managing server initialization, multi-replica
    state consistency checks, and graceful shutdown.
    """
    SERVER_STATE["vllm_version"] = Constants.VLLM_VERSION
    SERVER_STATE["is_sleeping"] = False

    db_path = get_active_state_db_path()
    if not db_path:
        logger.warning(
            "MOCK_STATE_DB_PATH is not configured; stateful stores are running in per-process memory. "
            "Set replicas=1 (or sessionAffinity=ClientIP) in Kubernetes, or set MOCK_STATE_DB_PATH "
            "to a shared SQLite file path for multi-replica state consistency.",
            extra={
                "structured_data": {
                    "event": "startup_state_check",
                    "state_backend": "in_memory",
                    "recommendation": "Set replicas=1 or configure MOCK_STATE_DB_PATH",
                }
            },
        )
    else:
        logger.info(
            "SQLite shared state backend enabled at %s",
            db_path,
            extra={"structured_data": {"event": "startup_state_check", "state_backend": "sqlite", "db_path": db_path}},
        )
    for handler in logger.handlers:
        handler.flush()
    yield


app = FastAPI(
    title=Constants.APP_TITLE,
    description=Constants.APP_DESCRIPTION,
    version=Constants.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Constants.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        Constants.HEADER_REQUEST_ID,
        Constants.HEADER_PROCESS_TIME_MS,
        Constants.HEADER_RESPONSE_TIME,
        Constants.HEADER_CLIENT_IP,
    ],
)

PUBLIC_PATHS = {
    "/health",
    "/ping",
    "/version",
    "/metrics",
    "/load",
    "/is_sleeping",
    "/server_info",
    "/v1/server_info",
    "/docs",
    "/openapi.json",
}


def _finalize_observability_and_log(
    request: Request,
    response: Response,
    request_id: str,
    client_ip: str,
    start_perf: float,
) -> Response:
    elapsed_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)

    if _is_observability_enabled():
        response.headers[Constants.HEADER_REQUEST_ID] = request_id
        response.headers[Constants.HEADER_CLIENT_IP] = client_ip
        response.headers[Constants.HEADER_PROCESS_TIME_MS] = f"{elapsed_ms:.3f}"
        response.headers[Constants.HEADER_RESPONSE_TIME] = f"{elapsed_ms:.3f}ms"

    if _is_logging_enabled():
        logger.info(
            "%s %s -> %d (%.3f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={
                "structured_data": {
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.url.query) if request.url.query else None,
                    "status_code": response.status_code,
                    "response_time_ms": elapsed_ms,
                    "user_agent": request.headers.get("User-Agent"),
                }
            },
        )
        for handler in logger.handlers:
            handler.flush()

    return response


def _maybe_build_chaos_error_response(request: Request) -> JSONResponse | None:
    """
    Evaluates optional Chaos / Fault-Injection headers (`X-Mock-Error-Status`, `X-Mock-Error-Rate`)
    or environment variable (`MOCK_ERROR_RATE`) to simulate 429, 500, 502, 503, or 504 errors.
    """
    if request.url.path in ("/health", "/ping"):
        return None

    status_header = request.headers.get(Constants.HEADER_MOCK_ERROR_STATUS, "").strip()
    rate_header = request.headers.get(
        Constants.HEADER_MOCK_ERROR_RATE,
        os.environ.get("MOCK_ERROR_RATE", str(Constants.DEFAULT_ERROR_RATE)),
    ).strip()

    forced_status: int | None = None
    if status_header:
        try:
            forced_status = int(status_header)
        except ValueError:
            forced_status = None

    error_rate = 0.0
    try:
        error_rate = max(0.0, min(1.0, float(rate_header)))
    except ValueError:
        error_rate = 0.0

    if forced_status is None and error_rate > 0.0 and random.random() < error_rate:
        forced_status = status.HTTP_503_SERVICE_UNAVAILABLE

    if forced_status is None or forced_status < 400:
        return None

    custom_msg = request.headers.get(Constants.HEADER_MOCK_ERROR_MESSAGE)
    if forced_status == status.HTTP_429_TOO_MANY_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": "1"},
            content={
                "error": {
                    "message": custom_msg or "Rate limit reached for requests on mock server.",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )

    return JSONResponse(
        status_code=forced_status,
        content={
            "error": {
                "message": custom_msg or f"Simulated upstream server error (HTTP {forced_status}).",
                "type": "server_error",
                "param": None,
                "code": f"mock_http_{forced_status}",
            }
        },
    )


def _is_malformed_sse_requested(request: Request) -> bool:
    val = request.headers.get(Constants.HEADER_MOCK_MALFORMED_SSE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Middleware: Real-IP, X-Request-ID, Response-Time, Chaos, Auth & Latency
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_auth_and_latency_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start_perf = time.perf_counter()
    SERVER_STATE["total_requests"] += 1

    request_id = extract_or_generate_request_id(request)
    client_ip = extract_real_client_ip(request)
    request.state.request_id = request_id
    request.state.client_ip = client_ip

    # 1. Optional API Key Check
    expected_key = os.environ.get("MOCK_API_KEY", "").strip()
    if expected_key and request.url.path not in PUBLIC_PATHS:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header.removeprefix("Bearer ").strip() != expected_key:
            unauthorized_res = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "message": "Incorrect API key provided or missing Bearer token.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
            return _finalize_observability_and_log(
                request=request,
                response=unauthorized_res,
                request_id=request_id,
                client_ip=client_ip,
                start_perf=start_perf,
            )

    # 2. Simulated Request Latency (Header X-Mock-Latency-Ms or Env MOCK_LATENCY_MS)
    latency_str = request.headers.get("X-Mock-Latency-Ms", os.environ.get("MOCK_LATENCY_MS", "0"))
    try:
        latency_ms = max(0, int(latency_str))
    except ValueError:
        latency_ms = 0

    if latency_ms > 0:
        await asyncio.sleep(latency_ms / 1000.0)

    # 3. Optional Chaos / Error Simulation (429, 500, 503, etc.)
    chaos_res = _maybe_build_chaos_error_response(request)
    if chaos_res is not None:
        return _finalize_observability_and_log(
            request=request,
            response=chaos_res,
            request_id=request_id,
            client_ip=client_ip,
            start_perf=start_perf,
        )

    SERVER_STATE["server_load"] += 1
    try:
        response = await call_next(request)
    finally:
        SERVER_STATE["server_load"] = max(0, SERVER_STATE["server_load"] - 1)

    return _finalize_observability_and_log(
        request=request,
        response=response,
        request_id=request_id,
        client_ip=client_ip,
        start_perf=start_perf,
    )


def _get_stream_delay_sec(request: Request) -> float:
    delay_str = request.headers.get("X-Mock-Stream-Delay-Ms", os.environ.get("MOCK_STREAM_DELAY_MS", "0"))
    try:
        return max(0.0, int(delay_str) / 1000.0)
    except ValueError:
        return 0.0


def _should_include_reasoning(model: str, explicit_flag: bool | None = None, reasoning_effort: str | None = None) -> bool:
    if explicit_flag is not None:
        return explicit_flag
    env_flag = os.environ.get("MOCK_INCLUDE_REASONING", "true").lower()
    if env_flag in ("1", "true", "yes", "always"):
        return True
    model_lower = model.lower()
    return any(tag in model_lower for tag in ("r1", "r2", "o1", "o3", "reason", "qwq", "thinking", "opus", "sol", "astra", "argon")) or bool(reasoning_effort)


# ---------------------------------------------------------------------------
# 1. vLLM Health, Version, Load, Metrics & Control Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Response:
    return Response(status_code=status.HTTP_200_OK)


@app.get("/ping")
@app.post("/ping")
async def ping() -> Response:
    return Response(status_code=status.HTTP_200_OK)


@app.get("/version")
async def get_version() -> dict[str, str]:
    return {"version": SERVER_STATE["vllm_version"]}


@app.get("/load")
async def get_load() -> dict[str, int]:
    return {"server_load": SERVER_STATE["server_load"]}


@app.get("/server_info")
@app.get("/v1/server_info")
async def get_server_info() -> dict[str, Any]:
    return {
        "vllm_version": SERVER_STATE["vllm_version"],
        "is_sleeping": SERVER_STATE["is_sleeping"],
        "sleep_level": SERVER_STATE["sleep_level"],
        "server_load": SERVER_STATE["server_load"],
        "total_requests": SERVER_STATE["total_requests"],
        "data_parallel_size": SERVER_STATE["data_parallel_size"],
        "state_backend": "sqlite" if get_active_state_db_path() else "in_memory",
        "models": [m["id"] for m in MOCK_MODELS],
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics() -> str:
    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running on GPU.",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="mock"}} {SERVER_STATE["server_load"]}',
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE vllm:num_requests_waiting gauge",
        'vllm:num_requests_waiting{model_name="mock"} 0',
        "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.",
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        'vllm:gpu_cache_usage_perc{model_name="mock"} 0.15',
        "# HELP vllm:request_success_total Count of successfully processed requests.",
        "# TYPE vllm:request_success_total counter",
        f'vllm:request_success_total{{model_name="mock"}} {SERVER_STATE["total_requests"]}',
    ]
    return "\n".join(lines) + "\n"


@app.post("/sleep")
async def sleep_engine(level: Annotated[int, Query()] = 1) -> Response:
    SERVER_STATE["is_sleeping"] = True
    SERVER_STATE["sleep_level"] = level
    return Response(status_code=status.HTTP_200_OK)


@app.post("/wake_up")
async def wake_up_engine(tags: Annotated[list[str] | None, Query()] = None) -> Response:
    _ = tags
    SERVER_STATE["is_sleeping"] = False
    SERVER_STATE["sleep_level"] = 0
    return Response(status_code=status.HTTP_200_OK)


@app.get("/is_sleeping")
async def is_sleeping() -> dict[str, bool]:
    return {"is_sleeping": SERVER_STATE["is_sleeping"]}


@app.post("/reset_prefix_cache")
async def reset_prefix_cache() -> Response:
    SERVER_STATE["prefix_cache_resets"] += 1
    return Response(status_code=status.HTTP_200_OK)


@app.post("/reset_mm_cache")
@app.post("/reset_encoder_cache")
async def reset_mm_cache() -> Response:
    SERVER_STATE["mm_cache_resets"] += 1
    return Response(status_code=status.HTTP_200_OK)


@app.post("/start_profile")
async def start_profile() -> Response:
    SERVER_STATE["is_profiling"] = True
    return Response(status_code=status.HTTP_200_OK)


@app.post("/stop_profile")
async def stop_profile() -> Response:
    SERVER_STATE["is_profiling"] = False
    return Response(status_code=status.HTTP_200_OK)


@app.post("/scale_elastic_ep")
async def scale_elastic_ep(req: ScaleElasticEPRequest) -> dict[str, Any]:
    SERVER_STATE["data_parallel_size"] = req.new_data_parallel_size
    return {"status": "ok", "new_data_parallel_size": req.new_data_parallel_size}


@app.post("/collective_rpc")
async def collective_rpc(req: CollectiveRpcRequest) -> dict[str, Any]:
    return {"status": "ok", "method": req.method, "results": [{"worker_id": 0, "status": "success"}]}


# ---------------------------------------------------------------------------
# 2. vLLM LoRA Adapter Management & Models (/v1/models)
# ---------------------------------------------------------------------------
@app.post("/v1/load_lora_adapter")
async def load_lora_adapter(req: LoadLoraAdapterRequest) -> PlainTextResponse:
    existing_ids = {m["id"] for m in MOCK_MODELS}
    if req.lora_name not in existing_ids:
        MOCK_MODELS.append(
            {
                "id": req.lora_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm-lora",
                "root": req.lora_path,
                "parent": MOCK_MODELS[0]["id"] if MOCK_MODELS else Constants.DEFAULT_CHAT_MODEL,
                "max_model_len": Constants.DEFAULT_LORA_MAX_MODEL_LEN,
            }
        )
    return PlainTextResponse(f"Success: LoRA adapter '{req.lora_name}' added successfully.")


@app.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(req: UnloadLoraAdapterRequest) -> PlainTextResponse:
    MOCK_MODELS[:] = [m for m in MOCK_MODELS if m["id"] != req.lora_name]
    return PlainTextResponse(f"Success: LoRA adapter '{req.lora_name}' removed successfully.")


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": list(MOCK_MODELS),
    }


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str) -> dict[str, Any]:
    for model in MOCK_MODELS:
        if model["id"] == model_id:
            return model
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "vllm-mock",
        "root": model_id,
        "parent": None,
        "max_model_len": Constants.DEFAULT_MAX_MODEL_LEN,
    }


@app.delete("/v1/models/{model_id:path}")
async def delete_model(model_id: str) -> dict[str, Any]:
    MOCK_MODELS[:] = [m for m in MOCK_MODELS if m["id"] != model_id]
    return {
        "id": model_id,
        "object": "model",
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# 3. Chat Completions (/v1/chat/completions)
# ---------------------------------------------------------------------------
def _extract_json_schema_from_request(
    response_format: dict[str, Any] | None,
    guided_json: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        js_spec = response_format.get("json_schema")
        if isinstance(js_spec, dict) and isinstance(js_spec.get("schema"), dict):
            return js_spec["schema"]
        if isinstance(response_format.get("schema"), dict):
            return response_format["schema"]
    if guided_json is not None:
        if isinstance(guided_json, dict):
            return guided_json
        if isinstance(guided_json, str):
            try:
                parsed = json.loads(guided_json)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None
    return None


def _resolve_chat_output(
    req: ChatCompletionRequest,
    sample: dict[str, Any],
    rng: random.Random | None = None,
) -> str:
    active_rng = rng or random
    if req.guided_choice and isinstance(req.guided_choice, list):
        return active_rng.choice(req.guided_choice)

    explicit_schema = _extract_json_schema_from_request(req.response_format, req.guided_json)
    if explicit_schema is not None:
        synthesized = synthesize_from_json_schema(
            explicit_schema,
            sample["structured_json"],
            explicit_schema,
            active_rng,
        )
        return json.dumps(synthesized)

    rf_type = (req.response_format or {}).get("type")
    if rf_type in ("json_object", "json_schema") or req.guided_json is not None:
        return json.dumps(sample["structured_json"])

    return sample["content"]


def _build_logprobs(
    text: str,
    enabled: bool | None,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    active_rng = rng or random
    words = text.split(" ")
    content_logprobs = []
    for idx, word in enumerate(words):
        token_str = word if idx == 0 else f" {word}"
        content_logprobs.append(
            {
                "token": token_str,
                "logprob": round(active_rng.uniform(-0.45, -0.01), 4),
                "bytes": list(token_str.encode("utf-8")),
                "top_logprobs": [
                    {
                        "token": token_str,
                        "logprob": -0.05,
                        "bytes": list(token_str.encode("utf-8")),
                    }
                ],
            }
        )
    return {"content": content_logprobs, "refusal": None}


@app.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest, request: Request) -> Response:
    created_ts = int(time.time())
    prompt_text = " ".join(
        str(m.get("content", "") if isinstance(m, dict) else (m.content or ""))
        for m in req.messages
    )
    rng = get_rng(req.seed, f"{req.model}:{prompt_text}")
    comp_hex = f"{rng.getrandbits(96):024x}" if req.seed is not None else uuid.uuid4().hex[:24]
    completion_id = f"chatcmpl-{comp_hex}"
    system_fp = Constants.SYSTEM_FINGERPRINT_CHAT

    prompt_tokens = estimate_tokens(prompt_text)
    include_reasoning = _should_include_reasoning(req.model, req.include_reasoning, req.reasoning_effort)

    if req.stream:
        delay_sec = _get_stream_delay_sec(request)
        malformed_sse = _is_malformed_sse_requested(request)
        return StreamingResponse(
            _stream_chat_completion(
                req=req,
                completion_id=completion_id,
                created_ts=created_ts,
                system_fp=system_fp,
                prompt_tokens=prompt_tokens,
                include_reasoning=include_reasoning,
                delay_sec=delay_sec,
                rng=rng,
                malformed_sse=malformed_sse,
            ),
            media_type="text/event-stream",
        )

    choices: list[dict[str, Any]] = []
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    safe_n = min(max(1, req.n), Constants.MAX_CHOICES_N)

    for idx in range(safe_n):
        sample = get_random_response_dict(rng)
        tool_calls = build_tool_calls(req.tools, req.tool_choice, sample, req.parallel_tool_calls, rng)
        reasoning_text = sample["reasoning_content"] if include_reasoning else None

        if tool_calls:
            content_text = None
            finish_reason = "tool_calls"
            comp_tokens = estimate_tokens(json.dumps(tool_calls))
        else:
            content_text = _resolve_chat_output(req, sample, rng)
            finish_reason = "stop"
            comp_tokens = estimate_tokens(content_text)

        r_tokens = estimate_tokens(reasoning_text) if reasoning_text else 0
        total_reasoning_tokens += r_tokens
        total_completion_tokens += comp_tokens + r_tokens

        message_obj: dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "refusal": None,
        }
        if reasoning_text:
            message_obj["reasoning_content"] = reasoning_text
        if tool_calls:
            message_obj["tool_calls"] = tool_calls

        choices.append(
            {
                "index": idx,
                "message": message_obj,
                "logprobs": _build_logprobs(content_text or "", req.logprobs, rng),
                "finish_reason": finish_reason,
                "stop_reason": None,
            }
        )

    response_payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": system_fp,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": total_reasoning_tokens,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
    }
    MOCK_CHAT_COMPLETIONS_STORE[completion_id] = {
        **response_payload,
        "input_messages": [
            m if isinstance(m, dict) else m.model_dump()
            for m in req.messages[: Constants.MAX_STORED_INPUT_MESSAGES]
        ],
    }
    return JSONResponse(content=response_payload)


@app.get("/v1/chat/completions")
async def list_chat_completions() -> dict[str, Any]:
    data = [{k: v for k, v in item.items() if k != "input_messages"} for item in MOCK_CHAT_COMPLETIONS_STORE.values()]
    return {"object": "list", "data": data, "has_more": False}


@app.get("/v1/chat/completions/{completion_id}")
async def get_chat_completion(completion_id: str) -> dict[str, Any]:
    if completion_id in MOCK_CHAT_COMPLETIONS_STORE:
        return {k: v for k, v in MOCK_CHAT_COMPLETIONS_STORE[completion_id].items() if k != "input_messages"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat completion {completion_id} not found")


@app.get("/v1/chat/completions/{completion_id}/messages")
async def get_chat_completion_messages(completion_id: str) -> dict[str, Any]:
    if completion_id in MOCK_CHAT_COMPLETIONS_STORE:
        msgs = MOCK_CHAT_COMPLETIONS_STORE[completion_id].get("input_messages", [])
        return {"object": "list", "data": msgs, "has_more": False}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat completion {completion_id} not found")


@app.post("/v1/chat/completions/{completion_id}")
async def update_chat_completion(completion_id: str, body: dict[str, Any]) -> dict[str, Any]:
    if completion_id not in MOCK_CHAT_COMPLETIONS_STORE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat completion {completion_id} not found")
    updated = dict(MOCK_CHAT_COMPLETIONS_STORE[completion_id])
    updated["metadata"] = body.get("metadata", {})
    MOCK_CHAT_COMPLETIONS_STORE[completion_id] = updated
    return {k: v for k, v in updated.items() if k != "input_messages"}


@app.delete("/v1/chat/completions/{completion_id}")
async def delete_chat_completion(completion_id: str) -> dict[str, Any]:
    MOCK_CHAT_COMPLETIONS_STORE.pop(completion_id, None)
    return {"object": "chat.completion.deleted", "id": completion_id, "deleted": True}


async def _stream_chat_completion(
    req: ChatCompletionRequest,
    completion_id: str,
    created_ts: int,
    system_fp: str,
    prompt_tokens: int,
    include_reasoning: bool,
    delay_sec: float,
    rng: random.Random | None = None,
    malformed_sse: bool = False,
) -> AsyncGenerator[str, None]:
    sample = get_random_response_dict(rng)
    tool_calls = build_tool_calls(req.tools, req.tool_choice, sample, req.parallel_tool_calls, rng)
    reasoning_text = sample["reasoning_content"] if include_reasoning else None

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": system_fp,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    if malformed_sse:
        yield 'data: {"id": "truncated_malformed_sse_frame"\n\n'
        return

    reasoning_tokens = 0
    if reasoning_text:
        r_words = reasoning_text.split(" ")
        reasoning_tokens = len(r_words)
        for i, word in enumerate(r_words):
            piece = word if i == 0 else f" {word}"
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "system_fingerprint": system_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": piece},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)

    completion_tokens = 0
    if tool_calls:
        for tc_idx, tc in enumerate(tool_calls):
            args_str = tc["function"]["arguments"]
            completion_tokens += estimate_tokens(args_str)
            # Emit tool call header + arguments delta with proper tool index
            tc_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "system_fingerprint": system_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tc_idx,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": args_str,
                                    },
                                }
                            ]
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(tc_chunk)}\n\n"
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)
        finish_reason = "tool_calls"
    else:
        output_text = _resolve_chat_output(req, sample, rng)
        words = output_text.split(" ")
        completion_tokens = len(words)
        for i, word in enumerate(words):
            piece = word if i == 0 else f" {word}"
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "system_fingerprint": system_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)
        finish_reason = "stop"

    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": system_fp,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"

    if req.stream_options and req.stream_options.include_usage:
        total_comp = completion_tokens + reasoning_tokens
        usage_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": req.model,
            "system_fingerprint": system_fp,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_comp,
                "total_tokens": prompt_tokens + total_comp,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 4. Legacy Text Completions (/v1/completions)
# ---------------------------------------------------------------------------
@app.post("/v1/completions")
async def create_completion(req: CompletionRequest, request: Request) -> Response:
    created_ts = int(time.time())
    system_fp = Constants.SYSTEM_FINGERPRINT_COMPLETION

    prompts: list[str]
    if isinstance(req.prompt, str):
        prompts = [req.prompt]
    elif isinstance(req.prompt, list) and all(isinstance(p, str) for p in req.prompt):
        prompts = req.prompt[: Constants.MAX_BATCH_INPUT_ITEMS]  # type: ignore[assignment]
    else:
        prompts = ["token_input"]

    rng = get_rng(req.seed, f"{req.model}:{'|'.join(prompts)}")
    comp_hex = f"{rng.getrandbits(96):024x}" if req.seed is not None else uuid.uuid4().hex[:24]
    comp_id = f"cmpl-{comp_hex}"
    prompt_tokens = sum(estimate_tokens(p) for p in prompts)

    if req.stream:
        delay_sec = _get_stream_delay_sec(request)
        malformed_sse = _is_malformed_sse_requested(request)
        return StreamingResponse(
            _stream_text_completion(
                req=req,
                comp_id=comp_id,
                created_ts=created_ts,
                system_fp=system_fp,
                prompt_tokens=prompt_tokens,
                delay_sec=delay_sec,
                rng=rng,
                malformed_sse=malformed_sse,
            ),
            media_type="text/event-stream",
        )

    choices: list[dict[str, Any]] = []
    completion_tokens = 0
    idx = 0
    safe_n = min(max(1, req.n), Constants.MAX_CHOICES_N)
    for prompt_str in prompts:
        for _ in range(safe_n):
            sample = get_random_response_dict(rng)
            text = sample["completion"].lstrip()
            if req.echo and prompt_str:
                text = f"{prompt_str} {text}"
            if req.suffix:
                text = f"{text}{req.suffix}"

            completion_tokens += estimate_tokens(text)
            choices.append(
                {
                    "text": text,
                    "index": idx,
                    "logprobs": (
                        {"tokens": text.split(), "token_logprobs": [-0.1] * len(text.split())}
                        if req.logprobs
                        else None
                    ),
                    "finish_reason": "stop",
                    "stop_reason": None,
                }
            )
            idx += 1

    return JSONResponse(
        content={
            "id": comp_id,
            "object": "text_completion",
            "created": created_ts,
            "model": req.model,
            "system_fingerprint": system_fp,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


async def _stream_text_completion(
    req: CompletionRequest,
    comp_id: str,
    created_ts: int,
    system_fp: str,
    prompt_tokens: int,
    delay_sec: float,
    rng: random.Random | None = None,
    malformed_sse: bool = False,
) -> AsyncGenerator[str, None]:
    if malformed_sse:
        yield 'data: {"id": "truncated_malformed_sse_frame"\n\n'
        return

    sample = get_random_response_dict(rng)
    text = sample["completion"].lstrip()
    words = text.split(" ")

    for i, word in enumerate(words):
        piece = word if i == 0 else f" {word}"
        chunk = {
            "id": comp_id,
            "object": "text_completion",
            "created": created_ts,
            "model": req.model,
            "system_fingerprint": system_fp,
            "choices": [
                {
                    "text": piece,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop" if i == len(words) - 1 else None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)

    if req.stream_options and req.stream_options.include_usage:
        usage_chunk = {
            "id": comp_id,
            "object": "text_completion",
            "created": created_ts,
            "model": req.model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(words),
                "total_tokens": prompt_tokens + len(words),
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 5. OpenAI / vLLM Latest Responses API (/v1/responses)
# ---------------------------------------------------------------------------
@app.post("/v1/responses")
async def create_response(req: ResponsesRequest, request: Request) -> Response:
    created_ts = int(time.time())
    input_str = req.input if isinstance(req.input, str) else json.dumps(req.input)
    rng = get_rng(req.seed, f"{req.model}:{input_str}")
    resp_hex = f"{rng.getrandbits(96):024x}" if req.seed is not None else uuid.uuid4().hex[:24]
    resp_id = f"resp_{resp_hex}"
    sample = get_random_response_dict(rng)
    input_tokens = estimate_tokens(input_str)

    text_format = (req.text or {}).get("format", {})
    text_format_type = text_format.get("type") if isinstance(text_format, dict) else None
    if text_format_type == "json_schema" and isinstance(text_format.get("schema"), dict):
        synthesized = synthesize_from_json_schema(
            text_format["schema"],
            sample["structured_json"],
            text_format["schema"],
            rng,
        )
        output_text = json.dumps(synthesized)
    elif text_format_type in ("json_object", "json_schema"):
        output_text = json.dumps(sample["structured_json"])
    else:
        output_text = sample["content"]

    reasoning_text = sample["reasoning_content"]
    output_tokens = estimate_tokens(output_text) + estimate_tokens(reasoning_text)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    rs_id = f"rs_{uuid.uuid4().hex[:24]}"

    output_items: list[dict[str, Any]] = [
        {
            "id": rs_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning_text}],
        }
    ]

    tool_calls = build_tool_calls(req.tools, req.tool_choice, sample, req.parallel_tool_calls, rng)
    if tool_calls:
        for tc in tool_calls:
            output_items.append(
                {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "status": "completed",
                }
            )
    else:
        output_items.append(
            {
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        )

    response_obj: dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": created_ts,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": req.instructions,
        "max_output_tokens": req.max_output_tokens,
        "model": req.model,
        "output": output_items,
        "output_text": "" if tool_calls else output_text,
        "parallel_tool_calls": req.parallel_tool_calls if req.parallel_tool_calls is not None else True,
        "previous_response_id": req.previous_response_id,
        "reasoning": req.reasoning or {"effort": "medium", "summary": "auto"},
        "store": req.store,
        "temperature": req.temperature,
        "text": req.text or {"format": {"type": "text"}},
        "tool_choice": req.tool_choice or "auto",
        "tools": req.tools or [],
        "top_p": req.top_p,
        "truncation": "disabled",
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": estimate_tokens(reasoning_text)},
            "total_tokens": input_tokens + output_tokens,
        },
        "user": None,
        "metadata": req.metadata or {},
    }

    if req.store:
        MOCK_RESPONSES_STORE[resp_id] = response_obj

    if req.stream:
        delay_sec = _get_stream_delay_sec(request)
        malformed_sse = _is_malformed_sse_requested(request)
        return StreamingResponse(
            _stream_responses_api(response_obj, output_text, delay_sec, malformed_sse),
            media_type="text/event-stream",
        )

    return JSONResponse(content=response_obj)


async def _stream_responses_api(
    response_obj: dict[str, Any],
    output_text: str,
    delay_sec: float,
    malformed_sse: bool = False,
) -> AsyncGenerator[str, None]:
    created_event = {"type": "response.created", "response": {**response_obj, "status": "in_progress"}}
    yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"

    in_progress_event = {"type": "response.in_progress", "response": {**response_obj, "status": "in_progress"}}
    yield f"event: response.in_progress\ndata: {json.dumps(in_progress_event)}\n\n"

    if malformed_sse:
        yield 'event: response.output_text.delta\ndata: {"truncated_json": \n\n'
        return

    for out_idx, item in enumerate(response_obj.get("output", [])):
        added_event = {
            "type": "response.output_item.added",
            "output_index": out_idx,
            "item": item,
        }
        yield f"event: response.output_item.added\ndata: {json.dumps(added_event)}\n\n"

        item_type = item.get("type")
        if item_type == "function_call":
            args_str = str(item.get("arguments", "{}"))
            chunk_size = max(1, len(args_str) // 2)
            for start in range(0, len(args_str), chunk_size):
                piece = args_str[start : start + chunk_size]
                fc_delta = {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": out_idx,
                    "call_id": item["call_id"],
                    "delta": piece,
                }
                yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(fc_delta)}\n\n"
                if delay_sec > 0:
                    await asyncio.sleep(delay_sec)

            fc_done = {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": out_idx,
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": args_str,
            }
            yield f"event: response.function_call_arguments.done\ndata: {json.dumps(fc_done)}\n\n"

        elif item_type == "message":
            for i, word in enumerate(output_text.split(" ")):
                delta = word if i == 0 else f" {word}"
                delta_event = {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": out_idx,
                    "content_index": 0,
                    "delta": delta,
                }
                yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"
                if delay_sec > 0:
                    await asyncio.sleep(delay_sec)

            done_text_event = {
                "type": "response.output_text.done",
                "item_id": item["id"],
                "output_index": out_idx,
                "content_index": 0,
                "text": output_text,
            }
            yield f"event: response.output_text.done\ndata: {json.dumps(done_text_event)}\n\n"

        done_item_event = {
            "type": "response.output_item.done",
            "output_index": out_idx,
            "item": item,
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(done_item_event)}\n\n"

    completed_event = {"type": "response.completed", "response": response_obj}
    yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/responses/{response_id}")
async def get_response_by_id(response_id: str) -> dict[str, Any]:
    if response_id in MOCK_RESPONSES_STORE:
        return MOCK_RESPONSES_STORE[response_id]
    sample = get_random_response_dict()
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": Constants.DEFAULT_RESPONSES_MODEL,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": sample["content"], "annotations": []}],
            }
        ],
        "output_text": sample["content"],
    }


@app.delete("/v1/responses/{response_id}")
async def delete_response_by_id(response_id: str) -> dict[str, Any]:
    MOCK_RESPONSES_STORE.pop(response_id, None)
    return {"id": response_id, "object": "response", "deleted": True}


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response_by_id(response_id: str) -> dict[str, Any]:
    if response_id in MOCK_RESPONSES_STORE:
        updated = dict(MOCK_RESPONSES_STORE[response_id])
        updated["status"] = "cancelled"
        MOCK_RESPONSES_STORE[response_id] = updated
        return updated
    return {"id": response_id, "object": "response", "status": "cancelled"}


@app.get("/v1/responses/{response_id}/input_items")
async def list_response_input_items(response_id: str) -> dict[str, Any]:
    _ = response_id
    sample = get_random_response_dict()
    return {
        "object": "list",
        "data": [
            {
                "id": f"msg_in_{uuid.uuid4().hex[:16]}",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": sample["content"]}],
            }
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# 6. Embeddings (/v1/embeddings, /v2/embeddings, /embeddings) &
#    vLLM Pooling (/pooling, /v1/pooling, /v2/pooling)
# ---------------------------------------------------------------------------
@app.post("/v1/embeddings")
@app.post("/v2/embeddings")
@app.post("/embeddings")
async def create_embeddings(req: EmbeddingRequest) -> dict[str, Any]:
    dims = req.dimensions or Constants.DEFAULT_EMBEDDING_DIMENSIONS
    if isinstance(req.input, str):
        items = [req.input]
    elif isinstance(req.input, list) and req.input and isinstance(req.input[0], int):
        items = ["token_list"]
    elif isinstance(req.input, list):
        items = [str(x) for x in req.input[: Constants.MAX_BATCH_INPUT_ITEMS]]
    else:
        items = ["input"]

    data = []
    total_tokens = 0
    for idx, item in enumerate(items):
        item_rng = get_rng(req.seed, f"{req.model}:{item}:{idx}")
        total_tokens += estimate_tokens(item)
        data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": generate_embedding_vector(
                    dimensions=dims,
                    encoding_format=req.encoding_format,
                    rng=item_rng,
                ),
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.post("/pooling")
@app.post("/v1/pooling")
@app.post("/v2/pooling")
async def create_pooling(req: PoolingRequest) -> dict[str, Any]:
    dims = req.dimensions or Constants.DEFAULT_POOLING_DIMENSIONS
    inputs = [req.input] if isinstance(req.input, str) else (req.input if isinstance(req.input, list) else ["pool"])
    data = [
        {
            "index": idx,
            "object": "pooling",
            "data": generate_embedding_vector(dimensions=dims, encoding_format=req.encoding_format),
        }
        for idx in range(max(1, min(len(inputs), Constants.MAX_BATCH_INPUT_ITEMS)))
    ]
    return {
        "id": f"pool-{uuid.uuid4().hex[:24]}",
        "object": "list",
        "created": int(time.time()),
        "model": req.model,
        "data": data,
        "usage": {"prompt_tokens": 12, "total_tokens": 12, "completion_tokens": 0},
    }


# ---------------------------------------------------------------------------
# 7. vLLM Score (/score, /v1/score, /v2/score),
#    Rerank (/rerank, /v1/rerank, /v2/rerank) &
#    Classify (/classify, /v1/classify, /v2/classify)
# ---------------------------------------------------------------------------
@app.post("/v1/score")
@app.post("/v2/score")
@app.post("/score")
async def create_score(req: ScoreRequest) -> dict[str, Any]:
    t1_list = [req.text_1] if isinstance(req.text_1, str) else req.text_1
    t2_list = [req.text_2] if isinstance(req.text_2, str) else req.text_2[: Constants.MAX_BATCH_INPUT_ITEMS]
    data = []
    for idx, t2_item in enumerate(t2_list):
        t1_item = t1_list[idx % len(t1_list)] if t1_list else ""
        data.append(
            {
                "index": idx,
                "object": "score",
                "score": compute_relevance_score(str(t1_item), str(t2_item)),
            }
        )
    return {
        "id": f"score-{uuid.uuid4().hex[:24]}",
        "object": "list",
        "created": int(time.time()),
        "model": req.model,
        "data": data,
        "usage": {"prompt_tokens": 24, "total_tokens": 24},
    }


@app.post("/v1/rerank")
@app.post("/v2/rerank")
@app.post("/rerank")
async def create_rerank(req: RerankRequest) -> dict[str, Any]:
    results = []
    for idx, doc in enumerate(req.documents[: Constants.MAX_BATCH_INPUT_ITEMS]):
        doc_text = doc if isinstance(doc, str) else doc.get("text", json.dumps(doc))
        entry: dict[str, Any] = {
            "index": idx,
            "relevance_score": compute_relevance_score(req.query, doc_text),
        }
        if req.return_documents:
            entry["document"] = {"text": doc_text}
        results.append(entry)

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if req.top_n is not None:
        results = results[: req.top_n]

    return {
        "id": f"rerank-{uuid.uuid4().hex[:24]}",
        "model": req.model,
        "usage": {"prompt_tokens": 32, "total_tokens": 32},
        "results": results,
    }


@app.post("/classify")
@app.post("/v1/classify")
@app.post("/v2/classify")
async def create_classify(req: ClassifyRequest) -> dict[str, Any]:
    items = [req.input] if isinstance(req.input, str) else req.input[: Constants.MAX_BATCH_INPUT_ITEMS]
    data = []
    for idx, item_str in enumerate(items):
        rng = get_rng(None, str(item_str))
        sample_cls = get_random_classification_dict(rng)
        data.append(
            {
                "index": idx,
                "label": sample_cls["label"],
                "probs": sample_cls["probs"],
                "num_classes": len(sample_cls["probs"]),
            }
        )
    return {
        "id": f"classify-{uuid.uuid4().hex[:24]}",
        "object": "list",
        "created": int(time.time()),
        "model": req.model,
        "data": data,
        "usage": {"prompt_tokens": 16, "total_tokens": 16, "completion_tokens": 0},
    }


# ---------------------------------------------------------------------------
# 8. vLLM Tokenizer Endpoints (/tokenize, /v1/tokenize, /v2/tokenize,
#    /detokenize, /v1/detokenize, /v2/detokenize)
# ---------------------------------------------------------------------------
@app.post("/tokenize")
@app.post("/v1/tokenize")
@app.post("/v2/tokenize")
async def tokenize_text(req: TokenizeRequest) -> dict[str, Any]:
    if req.prompt is not None:
        text = req.prompt
    elif req.messages:
        text = " ".join(str(m.get("content", "")) for m in req.messages)
    else:
        text = ""

    tokens = simple_tokenize(text)
    if req.add_special_tokens:
        tokens = [Constants.BOS_TOKEN_ID] + tokens

    resp: dict[str, Any] = {
        "count": len(tokens),
        "max_model_len": Constants.DEFAULT_MAX_MODEL_LEN,
        "tokens": tokens,
    }
    if req.return_token_strs:
        resp["token_strs"] = [
            "<|begin_of_text|>" if t == Constants.BOS_TOKEN_ID else simple_detokenize([t])
            for t in tokens
        ]
    return resp


@app.post("/detokenize")
@app.post("/v1/detokenize")
@app.post("/v2/detokenize")
async def detokenize_tokens(req: DetokenizeRequest) -> dict[str, str]:
    prompt_str = simple_detokenize(req.tokens)
    return {"prompt": prompt_str}


# ---------------------------------------------------------------------------
# 9. OpenAI & vLLM Audio Endpoints (/v1/audio/speech, /v1/audio/voices,
#    /v1/audio/transcriptions, /v1/audio/translations)
# ---------------------------------------------------------------------------
@app.get("/v1/audio/voices")
async def list_audio_voices() -> dict[str, Any]:
    return {"voices": MOCK_VOICES}


@app.post("/v1/audio/speech")
async def create_speech(req: AudioSpeechRequest) -> Response:
    media_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
    }
    return Response(content=CACHED_WAV_BYTES, media_type=media_map.get(req.response_format, "audio/wav"))


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = Constants.DEFAULT_TRANSCRIPTION_MODEL,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
) -> Response:
    _ = (model, prompt, temperature)
    await file.close()
    sample = get_random_transcription_dict()
    if response_format in ("text", "srt", "vtt"):
        return PlainTextResponse(sample["text"])
    if response_format == "verbose_json":
        return JSONResponse(
            content={
                "task": "transcribe",
                "language": language or sample["language"],
                "duration": sample["duration"],
                "text": sample["text"],
                "segments": sample["segments"],
            }
        )
    return JSONResponse(content={"text": sample["text"]})


@app.post("/v1/audio/translations")
async def create_translation(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = Constants.DEFAULT_TRANSCRIPTION_MODEL,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
) -> Response:
    _ = (model, prompt, temperature)
    await file.close()
    sample = get_random_transcription_dict()
    if response_format in ("text", "srt", "vtt"):
        return PlainTextResponse(sample["text"])
    return JSONResponse(content={"text": sample["text"]})


# ---------------------------------------------------------------------------
# 10. OpenAI Image Endpoints (/v1/images/generations, /v1/images/edits,
#     /v1/images/variations)
# ---------------------------------------------------------------------------
@app.post("/v1/images/generations")
async def create_image_generation(req: ImageGenerationRequest) -> dict[str, Any]:
    data = []
    safe_n = min(max(1, req.n), Constants.MAX_CHOICES_N)
    for _ in range(safe_n):
        sample = get_random_image_dict()
        if req.response_format == "b64_json":
            data.append({"b64_json": sample["b64_json"], "revised_prompt": sample["revised_prompt"]})
        else:
            data.append({"url": sample["url"], "revised_prompt": sample["revised_prompt"]})
    return {"created": int(time.time()), "data": data}


@app.post("/v1/images/edits")
@app.post("/v1/images/variations")
async def create_image_edit_or_variation(
    n: Annotated[int, Form()] = 1,
    response_format: Annotated[str, Form()] = "url",
) -> dict[str, Any]:
    data = []
    safe_n = min(max(1, n), Constants.MAX_CHOICES_N)
    for _ in range(safe_n):
        sample = get_random_image_dict()
        if response_format == "b64_json":
            data.append({"b64_json": sample["b64_json"], "revised_prompt": sample["revised_prompt"]})
        else:
            data.append({"url": sample["url"], "revised_prompt": sample["revised_prompt"]})
    return {"created": int(time.time()), "data": data}


# ---------------------------------------------------------------------------
# 11. OpenAI Moderations Endpoint (/v1/moderations)
# ---------------------------------------------------------------------------
@app.post("/v1/moderations")
async def create_moderation(req: ModerationRequest) -> dict[str, Any]:
    inputs = [req.input] if isinstance(req.input, str) else req.input[: Constants.MAX_BATCH_INPUT_ITEMS]
    results = []
    for _ in inputs:
        results.append(
            {
                "flagged": False,
                "categories": {
                    "sexual": False,
                    "hate": False,
                    "harassment": False,
                    "self-harm": False,
                    "sexual/minors": False,
                    "hate/threatening": False,
                    "violence/graphic": False,
                    "self-harm/intent": False,
                    "self-harm/instructions": False,
                    "harassment/threatening": False,
                    "violence": False,
                },
                "category_scores": {
                    "sexual": 0.0001,
                    "hate": 0.0001,
                    "harassment": 0.0002,
                    "self-harm": 0.0001,
                    "sexual/minors": 0.0001,
                    "hate/threatening": 0.0001,
                    "violence/graphic": 0.0001,
                    "self-harm/intent": 0.0001,
                    "self-harm/instructions": 0.0001,
                    "harassment/threatening": 0.0001,
                    "violence": 0.0002,
                },
            }
        )
    return {
        "id": f"modr-{uuid.uuid4().hex[:24]}",
        "model": req.model,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 12. OpenAI Files Endpoints (/v1/files) with Explicit 413 Size Enforcement
# ---------------------------------------------------------------------------
@app.get("/v1/files")
async def list_files(purpose: Annotated[str | None, Query()] = None) -> dict[str, Any]:
    files = [
        {k: v for k, v in f.items() if k != "content"}
        for f in MOCK_FILES_STORE
        if purpose is None or f.get("purpose") == purpose
    ]
    return {"object": "list", "data": files, "has_more": False}


@app.post("/v1/files", response_model=None)
async def upload_file(
    file: Annotated[UploadFile, File()],
    purpose: Annotated[str, Form()] = "batch",
) -> Response | dict[str, Any]:
    max_allowed = int(os.environ.get("MOCK_MAX_FILE_BYTES", str(MAX_STORED_FILE_CONTENT_BYTES)))
    raw_bytes = await file.read(max_allowed + 1)
    await file.close()
    if len(raw_bytes) > max_allowed:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "error": {
                    "message": (
                        f"Uploaded file exceeds maximum allowed size of {max_allowed} bytes. "
                        "Increase MOCK_MAX_FILE_BYTES if larger mock files are required."
                    ),
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "file_too_large",
                }
            },
        )
    file_entry = {
        "id": f"file-{uuid.uuid4().hex[:20]}",
        "object": "file",
        "bytes": len(raw_bytes),
        "created_at": int(time.time()),
        "filename": file.filename or "uploaded_data.jsonl",
        "purpose": purpose,
        "status": "processed",
        "content": raw_bytes.decode("utf-8", errors="replace"),
    }
    MOCK_FILES_STORE.append(file_entry)
    return {k: v for k, v in file_entry.items() if k != "content"}


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str) -> dict[str, Any]:
    for f in MOCK_FILES_STORE:
        if f["id"] == file_id:
            return {k: v for k, v in f.items() if k != "content"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File {file_id} not found")


@app.get("/v1/files/{file_id}/content", response_class=PlainTextResponse)
async def retrieve_file_content(file_id: str) -> str:
    for f in MOCK_FILES_STORE:
        if f["id"] == file_id:
            return str(f.get("content", ""))
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File {file_id} not found")


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str) -> dict[str, Any]:
    MOCK_FILES_STORE[:] = [f for f in MOCK_FILES_STORE if f["id"] != file_id]
    return {"id": file_id, "object": "file", "deleted": True}


# ---------------------------------------------------------------------------
# 13. OpenAI Batches Endpoints (/v1/batches)
# ---------------------------------------------------------------------------
@app.get("/v1/batches")
async def list_batches() -> dict[str, Any]:
    return {"object": "list", "data": list(MOCK_BATCHES_STORE), "has_more": False}


@app.post("/v1/batches")
async def create_batch(req: BatchCreateRequest) -> dict[str, Any]:
    now = int(time.time())
    batch_entry: dict[str, Any] = {
        "id": f"batch_{uuid.uuid4().hex[:20]}",
        "object": "batch",
        "endpoint": req.endpoint,
        "errors": None,
        "input_file_id": req.input_file_id,
        "completion_window": req.completion_window,
        "status": "in_progress",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": now,
        "in_progress_at": now,
        "expires_at": now + 86400,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 10, "completed": 0, "failed": 0},
        "metadata": req.metadata or {},
    }
    MOCK_BATCHES_STORE.append(batch_entry)
    return batch_entry


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str) -> dict[str, Any]:
    for b in MOCK_BATCHES_STORE:
        if b["id"] == batch_id:
            return b
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch {batch_id} not found")


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str) -> dict[str, Any]:
    items = list(MOCK_BATCHES_STORE)
    for idx, b in enumerate(items):
        if b["id"] == batch_id:
            updated = dict(b)
            updated["status"] = "cancelled"
            updated["cancelled_at"] = int(time.time())
            MOCK_BATCHES_STORE[idx] = updated
            return updated
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch {batch_id} not found")


# ---------------------------------------------------------------------------
# 14. OpenAI Fine-Tuning Endpoints (/v1/fine_tuning/jobs)
# ---------------------------------------------------------------------------
@app.get("/v1/fine_tuning/jobs")
async def list_fine_tuning_jobs() -> dict[str, Any]:
    return {"object": "list", "data": list(MOCK_FINE_TUNING_STORE), "has_more": False}


@app.post("/v1/fine_tuning/jobs")
async def create_fine_tuning_job(req: FineTuningJobCreateRequest) -> dict[str, Any]:
    now = int(time.time())
    job_id = f"ftjob-{uuid.uuid4().hex[:16]}"
    job_entry: dict[str, Any] = {
        "id": job_id,
        "object": "fine_tuning.job",
        "created_at": now,
        "finished_at": None,
        "model": req.model,
        "fine_tuned_model": f"ft:{req.model}:mock-org::{req.suffix or 'custom'}",
        "organization_id": Constants.ORGANIZATION_ID,
        "result_files": [],
        "status": "running",
        "validation_file": req.validation_file,
        "training_file": req.training_file,
        "hyperparameters": req.hyperparameters or {"n_epochs": "auto"},
        "trained_tokens": None,
        "error": {},
        "seed": req.seed or 42,
    }
    MOCK_FINE_TUNING_STORE.append(job_entry)
    return job_entry


@app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}")
async def retrieve_fine_tuning_job(fine_tuning_job_id: str) -> dict[str, Any]:
    for job in MOCK_FINE_TUNING_STORE:
        if job["id"] == fine_tuning_job_id:
            return job
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Fine-tuning job {fine_tuning_job_id} not found")


@app.post("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel")
async def cancel_fine_tuning_job(fine_tuning_job_id: str) -> dict[str, Any]:
    items = list(MOCK_FINE_TUNING_STORE)
    for idx, job in enumerate(items):
        if job["id"] == fine_tuning_job_id:
            updated = dict(job)
            updated["status"] = "cancelled"
            MOCK_FINE_TUNING_STORE[idx] = updated
            return updated
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Fine-tuning job {fine_tuning_job_id} not found")


@app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}/events")
async def list_fine_tuning_events(fine_tuning_job_id: str) -> dict[str, Any]:
    _ = fine_tuning_job_id
    sample = get_random_response_dict()
    return {
        "object": "list",
        "data": [
            {
                "id": f"ftevent-{uuid.uuid4().hex[:16]}",
                "object": "fine_tuning.job.event",
                "created_at": int(time.time()),
                "level": "info",
                "message": f"Fine-tuning step completed: {sample['content'][:80]}",
                "type": "message",
            }
        ],
        "has_more": False,
    }


@app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints")
async def list_fine_tuning_checkpoints(fine_tuning_job_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": f"ftckpt_{uuid.uuid4().hex[:16]}",
                "object": "fine_tuning.job.checkpoint",
                "created_at": int(time.time()),
                "fine_tuned_model_checkpoint": f"ft:{fine_tuning_job_id}:ckpt-step-100",
                "fine_tuning_job_id": fine_tuning_job_id,
                "metrics": {"step": 100, "train_loss": 0.142, "valid_loss": 0.168},
                "step_number": 100,
            }
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# 15. OpenAI Uploads Endpoints (/v1/uploads)
# ---------------------------------------------------------------------------
@app.post("/v1/uploads")
async def create_upload(req: UploadCreateRequest) -> dict[str, Any]:
    upload_id = f"upload_{uuid.uuid4().hex[:20]}"
    now = int(time.time())
    upload_obj = {
        "id": upload_id,
        "object": "upload",
        "bytes": req.bytes,
        "created_at": now,
        "filename": req.filename,
        "purpose": req.purpose,
        "status": "pending",
        "expires_at": now + 3600,
    }
    MOCK_UPLOADS_STORE[upload_id] = upload_obj
    return upload_obj


@app.post("/v1/uploads/{upload_id}/parts", response_model=None)
async def add_upload_part(
    upload_id: str,
    data: Annotated[UploadFile, File()],
) -> Response | dict[str, Any]:
    max_allowed = int(os.environ.get("MOCK_MAX_FILE_BYTES", str(MAX_STORED_FILE_CONTENT_BYTES)))
    chunk = await data.read(max_allowed + 1)
    await data.close()
    if len(chunk) > max_allowed:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error": {
                    "message": f"Upload part exceeds maximum allowed size of {max_allowed} bytes.",
                    "type": "invalid_request_error",
                    "param": "data",
                    "code": "part_too_large",
                }
            },
        )
    return {
        "id": f"part_{uuid.uuid4().hex[:20]}",
        "object": "upload.part",
        "created_at": int(time.time()),
        "upload_id": upload_id,
    }


@app.post("/v1/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, req: UploadCompleteRequest) -> dict[str, Any]:
    _ = req
    upload_obj = dict(
        MOCK_UPLOADS_STORE.get(
            upload_id,
            {
                "id": upload_id,
                "object": "upload",
                "bytes": 1024,
                "created_at": int(time.time()),
                "filename": "completed_upload.jsonl",
                "purpose": "batch",
            },
        )
    )
    upload_obj["status"] = "completed"
    upload_obj["file"] = {
        "id": f"file-{uuid.uuid4().hex[:20]}",
        "object": "file",
        "bytes": upload_obj["bytes"],
        "created_at": int(time.time()),
        "filename": upload_obj["filename"],
        "purpose": upload_obj["purpose"],
    }
    MOCK_UPLOADS_STORE[upload_id] = upload_obj
    return upload_obj


@app.post("/v1/uploads/{upload_id}/cancel")
async def cancel_upload(upload_id: str) -> dict[str, Any]:
    upload_obj = dict(MOCK_UPLOADS_STORE.get(upload_id, {"id": upload_id, "object": "upload"}))
    upload_obj["status"] = "cancelled"
    MOCK_UPLOADS_STORE[upload_id] = upload_obj
    return upload_obj


# ---------------------------------------------------------------------------
# 16. OpenAI Vector Stores Endpoints (/v1/vector_stores)
# ---------------------------------------------------------------------------
@app.get("/v1/vector_stores")
async def list_vector_stores() -> dict[str, Any]:
    return {"object": "list", "data": list(MOCK_VECTOR_STORES), "has_more": False}


@app.post("/v1/vector_stores")
async def create_vector_store(req: VectorStoreCreateRequest) -> dict[str, Any]:
    now = int(time.time())
    vs_obj = {
        "id": f"vs_{uuid.uuid4().hex[:20]}",
        "object": "vector_store",
        "created_at": now,
        "name": req.name,
        "usage_bytes": 16384,
        "file_counts": {"in_progress": 0, "completed": len(req.file_ids), "failed": 0, "cancelled": 0, "total": len(req.file_ids)},
        "status": "completed",
        "last_active_at": now,
        "metadata": req.metadata or {},
    }
    MOCK_VECTOR_STORES.append(vs_obj)
    return vs_obj


@app.get("/v1/vector_stores/{vector_store_id}")
async def retrieve_vector_store(vector_store_id: str) -> dict[str, Any]:
    for vs in MOCK_VECTOR_STORES:
        if vs["id"] == vector_store_id:
            return vs
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Vector store {vector_store_id} not found")


@app.delete("/v1/vector_stores/{vector_store_id}")
async def delete_vector_store(vector_store_id: str) -> dict[str, Any]:
    MOCK_VECTOR_STORES[:] = [vs for vs in MOCK_VECTOR_STORES if vs["id"] != vector_store_id]
    return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}


@app.post("/v1/vector_stores/{vector_store_id}/search")
async def search_vector_store(vector_store_id: str, req: VectorStoreSearchRequest) -> dict[str, Any]:
    _ = vector_store_id
    sample = get_random_response_dict()
    return {
        "object": "vector_store.search_results.page",
        "search_query": req.query if isinstance(req.query, str) else " ".join(req.query),
        "data": [
            {
                "file_id": Constants.SEED_FILE_EVAL_ID,
                "filename": Constants.SEED_FILE_EVAL_NAME,
                "score": 0.96,
                "attributes": {"category": sample["category"]},
                "content": [{"type": "text", "text": sample["content"]}],
            }
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# 17. OpenAI Assistants, Threads, Messages & Runs (/v1/assistants, /v1/threads)
# ---------------------------------------------------------------------------
@app.get("/v1/assistants")
async def list_assistants() -> dict[str, Any]:
    return {"object": "list", "data": list(MOCK_ASSISTANTS_STORE), "has_more": False}


@app.post("/v1/assistants")
async def create_assistant(req: AssistantCreateRequest) -> dict[str, Any]:
    asst_obj = {
        "id": f"asst_{uuid.uuid4().hex[:20]}",
        "object": "assistant",
        "created_at": int(time.time()),
        "name": req.name,
        "description": req.description,
        "model": req.model,
        "instructions": req.instructions,
        "tools": req.tools,
        "metadata": req.metadata or {},
    }
    MOCK_ASSISTANTS_STORE.append(asst_obj)
    return asst_obj


@app.get("/v1/assistants/{assistant_id}")
async def retrieve_assistant(assistant_id: str) -> dict[str, Any]:
    for a in MOCK_ASSISTANTS_STORE:
        if a["id"] == assistant_id:
            return a
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Assistant {assistant_id} not found")


@app.delete("/v1/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str) -> dict[str, Any]:
    MOCK_ASSISTANTS_STORE[:] = [a for a in MOCK_ASSISTANTS_STORE if a["id"] != assistant_id]
    return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}


@app.post("/v1/threads")
async def create_thread(req: ThreadCreateRequest) -> dict[str, Any]:
    thread_id = f"thread_{uuid.uuid4().hex[:20]}"
    thread_obj = {
        "id": thread_id,
        "object": "thread",
        "created_at": int(time.time()),
        "metadata": req.metadata or {},
    }
    MOCK_THREADS_STORE[thread_id] = thread_obj
    MOCK_THREAD_MESSAGES_STORE[thread_id] = BoundedListStore(max_size=Constants.MAX_THREAD_SUBSTORE_SIZE)
    return thread_obj


@app.get("/v1/threads/{thread_id}")
async def retrieve_thread(thread_id: str) -> dict[str, Any]:
    if thread_id in MOCK_THREADS_STORE:
        return MOCK_THREADS_STORE[thread_id]
    return {"id": thread_id, "object": "thread", "created_at": int(time.time()), "metadata": {}}


@app.delete("/v1/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, Any]:
    MOCK_THREADS_STORE.pop(thread_id, None)
    MOCK_THREAD_MESSAGES_STORE.pop(thread_id, None)
    MOCK_THREAD_RUNS_STORE.pop(thread_id, None)
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


@app.post("/v1/threads/{thread_id}/messages")
async def create_thread_message(thread_id: str, req: ThreadMessageCreateRequest) -> dict[str, Any]:
    text_val = req.content if isinstance(req.content, str) else json.dumps(req.content)
    msg_obj = {
        "id": f"msg_{uuid.uuid4().hex[:20]}",
        "object": "thread.message",
        "created_at": int(time.time()),
        "thread_id": thread_id,
        "role": req.role,
        "content": [{"type": "text", "text": {"value": text_val, "annotations": []}}],
        "metadata": req.metadata or {},
    }
    existing = list(MOCK_THREAD_MESSAGES_STORE.get(thread_id, []))
    store = BoundedListStore(existing, max_size=Constants.MAX_THREAD_SUBSTORE_SIZE)
    store.append(msg_obj)
    MOCK_THREAD_MESSAGES_STORE[thread_id] = store
    return msg_obj


@app.get("/v1/threads/{thread_id}/messages")
async def list_thread_messages(thread_id: str) -> dict[str, Any]:
    sample = get_random_response_dict()
    msgs = MOCK_THREAD_MESSAGES_STORE.get(thread_id) or [
        {
            "id": f"msg_{uuid.uuid4().hex[:20]}",
            "object": "thread.message",
            "created_at": int(time.time()),
            "thread_id": thread_id,
            "role": "assistant",
            "content": [{"type": "text", "text": {"value": sample["content"], "annotations": []}}],
            "metadata": {},
        }
    ]
    return {"object": "list", "data": list(msgs), "has_more": False}


@app.post("/v1/threads/{thread_id}/runs")
async def create_thread_run(thread_id: str, req: ThreadRunCreateRequest) -> dict[str, Any]:
    sample = get_random_response_dict()
    now = int(time.time())
    run_obj = {
        "id": f"run_{uuid.uuid4().hex[:20]}",
        "object": "thread.run",
        "created_at": now,
        "completed_at": now,
        "thread_id": thread_id,
        "assistant_id": req.assistant_id,
        "status": "completed",
        "model": req.model or Constants.DEFAULT_ASSISTANT_MODEL,
        "instructions": req.instructions or sample["content"],
        "usage": {"prompt_tokens": 20, "completion_tokens": 45, "total_tokens": 65},
    }
    existing = list(MOCK_THREAD_RUNS_STORE.get(thread_id, []))
    store = BoundedListStore(existing, max_size=Constants.MAX_THREAD_SUBSTORE_SIZE)
    store.append(run_obj)
    MOCK_THREAD_RUNS_STORE[thread_id] = store
    return run_obj


@app.get("/v1/threads/{thread_id}/runs/{run_id}")
async def retrieve_thread_run(thread_id: str, run_id: str) -> dict[str, Any]:
    for r in MOCK_THREAD_RUNS_STORE.get(thread_id, []):
        if r["id"] == run_id:
            return r
    return {
        "id": run_id,
        "object": "thread.run",
        "created_at": int(time.time()),
        "thread_id": thread_id,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# 18. vLLM SageMaker / BentoML Invocations (/invocations) &
#     OpenAI / vLLM Realtime WebSocket (/v1/realtime)
# ---------------------------------------------------------------------------
@app.post("/invocations")
async def sagemaker_invocations(payload: dict[str, Any], request: Request) -> Response:
    """vLLM AWS SageMaker compatible inference route."""
    if "messages" in payload:
        chat_req = ChatCompletionRequest(**payload)
        return await create_chat_completion(chat_req, request)
    comp_req = CompletionRequest(**payload)
    return await create_completion(comp_req, request)


@app.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket) -> None:
    """
    OpenAI & vLLM Realtime API WebSocket endpoint supporting session management,
    input_audio_buffer events, conversation.item.create, and full response lifecycle events.
    """
    await websocket.accept()
    session_id = f"sess_{uuid.uuid4().hex[:20]}"
    session_state: dict[str, Any] = {
        "id": session_id,
        "object": "realtime.session",
        "model": websocket.query_params.get("model", Constants.DEFAULT_REALTIME_MODEL),
        "modalities": ["text", "audio"],
        "voice": Constants.DEFAULT_VOICE,
    }
    await websocket.send_json(
        {
            "event_id": f"event_{uuid.uuid4().hex[:16]}",
            "type": "session.created",
            "session": session_state,
        }
    )
    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                client_event = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "Invalid JSON payload."},
                    }
                )
                continue

            ev_type = client_event.get("type", "")
            sample = get_random_response_dict()

            if ev_type == "session.update":
                session_state.update(client_event.get("session", {}))
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "session.updated",
                        "session": session_state,
                    }
                )
            elif ev_type == "input_audio_buffer.append":
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": 0,
                        "item_id": f"item_{uuid.uuid4().hex[:16]}",
                    }
                )
            elif ev_type == "input_audio_buffer.commit":
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "input_audio_buffer.committed",
                        "item_id": f"item_{uuid.uuid4().hex[:16]}",
                    }
                )
            elif ev_type == "input_audio_buffer.clear":
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "input_audio_buffer.cleared",
                    }
                )
            elif ev_type == "conversation.item.create":
                item = client_event.get("item", {"id": f"item_{uuid.uuid4().hex[:16]}", "type": "message"})
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "conversation.item.created",
                        "item": item,
                    }
                )
            elif ev_type == "response.cancel":
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "response.cancelled",
                    }
                )
            else:
                resp_id = f"resp_{uuid.uuid4().hex[:16]}"
                item_id = f"item_{uuid.uuid4().hex[:16]}"
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "response.created",
                        "response": {"id": resp_id, "object": "realtime.response", "status": "in_progress"},
                    }
                )
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "response.output_item.added",
                        "response_id": resp_id,
                        "output_index": 0,
                        "item": {"id": item_id, "type": "message", "role": "assistant"},
                    }
                )
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "response.text.delta",
                        "response_id": resp_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": sample["content"],
                    }
                )
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "response.text.done",
                        "response_id": resp_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": sample["content"],
                    }
                )
                await websocket.send_json(
                    {
                        "event_id": f"event_{uuid.uuid4().hex[:16]}",
                        "type": "response.done",
                        "response": {
                            "id": resp_id,
                            "object": "realtime.response",
                            "status": "completed",
                            "output": [
                                {
                                    "id": item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": sample["content"]}],
                                }
                            ],
                        },
                    }
                )
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=Constants.DEFAULT_HOST, port=Constants.DEFAULT_PORT, reload=True)
