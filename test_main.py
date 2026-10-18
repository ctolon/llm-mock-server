"""
Comprehensive test suite for the OpenAI & vLLM compatible mock FastAPI server.
Tests all endpoints, 2026 SOTA models, centralized Constants, and bounded memory-safe stores
via FastAPI TestClient and the official OpenAI Python SDK.
"""

from __future__ import annotations

import io
import json
from typing import Any
from fastapi.testclient import TestClient
from openai import OpenAI

from constants import Constants
from main import app
from mock_data import (
    BoundedDictStore,
    BoundedListStore,
    MOCK_CLASSIFICATIONS,
    MOCK_IMAGE_PROMPTS,
    MOCK_RESPONSES,
    MOCK_TRANSCRIPTIONS,
)

client = TestClient(app)


def test_vllm_health_version_load_metrics_and_sleep() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/ping").status_code == 200
    assert client.post("/ping").status_code == 200

    ver = client.get("/version")
    assert ver.status_code == 200
    assert ver.json()["version"] == Constants.VLLM_VERSION
    assert Constants.VLLM_VERSION == "0.32.0"

    info = client.get("/server_info")
    assert info.status_code == 200
    assert info.json()["vllm_version"] == Constants.VLLM_VERSION

    load = client.get("/load")
    assert load.status_code == 200
    assert "server_load" in load.json()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "vllm:num_requests_running" in metrics.text

    # Sleep / Wake up / Prefix cache reset
    assert client.post("/sleep?level=2").status_code == 200
    assert client.get("/is_sleeping").json()["is_sleeping"] is True
    assert client.post("/wake_up").status_code == 200
    assert client.get("/is_sleeping").json()["is_sleeping"] is False
    assert client.post("/reset_prefix_cache").status_code == 200


def test_sota_2026_models_and_lora_adapters() -> None:
    res = client.get("/v1/models")
    assert res.status_code == 200
    models_data = res.json()
    assert models_data["object"] == "list"
    model_ids = {m["id"] for m in models_data["data"]}

    expected_sota_models = set(Constants.ALL_SOTA_MODEL_IDS)
    assert expected_sota_models.issubset(model_ids)

    # Load LoRA adapter
    load_res = client.post(
        "/v1/load_lora_adapter",
        json={"lora_name": "sql-lora-2026", "lora_path": "/models/sql-lora-2026"},
    )
    assert load_res.status_code == 200

    model_info = client.get("/v1/models/sql-lora-2026")
    assert model_info.status_code == 200
    assert model_info.json()["id"] == "sql-lora-2026"

    # Unload LoRA adapter
    unload_res = client.post(
        "/v1/unload_lora_adapter",
        json={"lora_name": "sql-lora-2026"},
    )
    assert unload_res.status_code == 200


def test_chat_completions_english_list_of_dict() -> None:
    valid_contents = {item["content"] for item in MOCK_RESPONSES}
    valid_reasonings = {item["reasoning_content"] for item in MOCK_RESPONSES}

    res = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_CLAUDE_OPUS,
            "messages": [{"role": "user", "content": "Explain disaggregated prefill-and-decode serving."}],
            "n": 2,
            "logprobs": True,
            "top_k": 50,
            "repetition_penalty": 1.05,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == Constants.MODEL_CLAUDE_OPUS
    assert len(body["choices"]) == 2
    for choice in body["choices"]:
        msg = choice["message"]
        assert msg["content"] in valid_contents
        assert msg["reasoning_content"] in valid_reasonings
        assert choice["logprobs"] is not None


def test_chat_completions_streaming_with_usage() -> None:
    valid_contents = {item["content"] for item in MOCK_RESPONSES}
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_GEMINI_ARGON,
            "messages": [{"role": "user", "content": "Stream a response."}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"

        assembled_content = ""
        assembled_reasoning = ""
        saw_usage = False
        for line in lines[:-1]:
            payload = json.loads(line.removeprefix("data: "))
            if payload.get("usage"):
                saw_usage = True
            for ch in payload.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content"):
                    assembled_content += delta["content"]
                if delta.get("reasoning_content"):
                    assembled_reasoning += delta["reasoning_content"]

        assert assembled_content in valid_contents
        assert len(assembled_reasoning) > 0
        assert saw_usage is True


def test_chat_completions_tool_calling_and_structured_outputs() -> None:
    # 1. Tool Calling
    res_tools = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_GPT_SOL,
            "messages": [{"role": "user", "content": "Check cluster status in San Francisco"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_cluster_telemetry",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "precision": {"type": "string", "enum": ["mxfp4", "fp8"]},
                            },
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    assert res_tools.status_code == 200
    choice = res_tools.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_cluster_telemetry"
    parsed_args = json.loads(tc["function"]["arguments"])
    assert parsed_args["location"] == "San Francisco, CA"
    assert parsed_args["precision"] in ("mxfp4", "fp8")

    # 2. Structured Outputs (json_object & vLLM guided_json)
    res_json = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_GLM_FLASH,
            "messages": [{"role": "user", "content": "Give JSON output"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert res_json.status_code == 200
    structured = json.loads(res_json.json()["choices"][0]["message"]["content"])
    assert "status" in structured and "confidence_score" in structured


def test_legacy_completions_and_streaming() -> None:
    valid_completions = {item["completion"].lstrip() for item in MOCK_RESPONSES}

    res = client.post(
        "/v1/completions",
        json={"model": Constants.MODEL_QWEN_INSTRUCT, "prompt": "In disaggregated serving,"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] in valid_completions

    # Streaming completion
    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": Constants.MODEL_QWEN_INSTRUCT, "prompt": "Hello", "stream": True},
    ) as stream_res:
        assert stream_res.status_code == 200
        chunks = [l for l in stream_res.iter_lines() if l.startswith("data: ")]
        assert chunks[-1] == "data: [DONE]"


def test_responses_api_sync_stream_and_crud() -> None:
    valid_contents = {item["content"] for item in MOCK_RESPONSES}

    res = client.post(
        "/v1/responses",
        json={"model": Constants.MODEL_GPT_ASTRA, "input": "Tell me about EAGLE-3 speculative decoding."},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "response"
    assert body["output_text"] in valid_contents
    resp_id = body["id"]

    # Retrieve & Delete
    get_res = client.get(f"/v1/responses/{resp_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == resp_id

    del_res = client.delete(f"/v1/responses/{resp_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Streaming Responses API
    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": Constants.MODEL_CLAUDE_SONNET, "input": "Stream test", "stream": True},
    ) as stream_res:
        assert stream_res.status_code == 200
        lines = [l for l in stream_res.iter_lines() if l]
        assert any("response.output_text.delta" in l for l in lines)
        assert lines[-1] == "data: [DONE]"


def test_embeddings_pooling_score_rerank_classify_and_tokenizer() -> None:
    # Embeddings float & base64
    emb_res = client.post(
        "/v1/embeddings",
        json={"model": Constants.MODEL_EMBEDDING, "input": ["hello world", "vllm mock"], "dimensions": 64},
    )
    assert emb_res.status_code == 200
    assert len(emb_res.json()["data"]) == 2
    assert len(emb_res.json()["data"][0]["embedding"]) == 64

    emb_b64 = client.post(
        "/v1/embeddings",
        json={"model": Constants.MODEL_EMBEDDING, "input": "hello", "encoding_format": "base64", "dimensions": 32},
    )
    assert emb_b64.status_code == 200
    assert isinstance(emb_b64.json()["data"][0]["embedding"], str)

    # Pooling
    pool_res = client.post("/pooling", json={"model": Constants.MODEL_RERANKER, "input": "test pooling"})
    assert pool_res.status_code == 200

    # Score
    score_res = client.post(
        "/v1/score",
        json={"model": Constants.MODEL_RERANKER, "text_1": "query", "text_2": ["doc 1", "doc 2"]},
    )
    assert score_res.status_code == 200
    assert len(score_res.json()["data"]) == 2

    # Rerank
    rerank_res = client.post(
        "/v1/rerank",
        json={
            "model": Constants.MODEL_RERANKER,
            "query": "What is MXFP4 quantization?",
            "documents": ["Doc A about databases", "Doc B about MXFP4 in vLLM", "Doc C about networking"],
            "top_n": 2,
        },
    )
    assert rerank_res.status_code == 200
    assert len(rerank_res.json()["results"]) == 2

    # Classify
    cls_res = client.post("/classify", json={"model": Constants.MODEL_RERANKER, "input": ["Route to deep reasoning"]})
    assert cls_res.status_code == 200
    valid_labels = {c["label"] for c in MOCK_CLASSIFICATIONS}
    assert cls_res.json()["data"][0]["label"] in valid_labels

    # Tokenize & Detokenize
    tok_res = client.post("/tokenize", json={"prompt": "MXFP4 quantization halves HBM footprint."})
    assert tok_res.status_code == 200
    tokens = tok_res.json()["tokens"]
    assert len(tokens) > 0

    detok_res = client.post("/detokenize", json={"tokens": tokens})
    assert detok_res.status_code == 200
    assert len(detok_res.json()["prompt"]) > 0


def test_audio_images_moderations_files_and_batches() -> None:
    # Audio Speech
    speech_res = client.post("/v1/audio/speech", json={"model": Constants.MODEL_TTS, "input": "Hello from 2026 mock server."})
    assert speech_res.status_code == 200
    assert speech_res.content.startswith(b"RIFF")

    # Audio Transcriptions
    valid_tx = {t["text"] for t in MOCK_TRANSCRIPTIONS}
    tx_res = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", io.BytesIO(speech_res.content), "audio/wav")},
        data={"model": Constants.MODEL_WHISPER},
    )
    assert tx_res.status_code == 200
    assert tx_res.json()["text"] in valid_tx

    # Images Generations
    valid_prompts = {img["revised_prompt"] for img in MOCK_IMAGE_PROMPTS}
    img_res = client.post("/v1/images/generations", json={"prompt": "An exascale NVLink cluster", "n": 2})
    assert img_res.status_code == 200
    assert len(img_res.json()["data"]) == 2
    assert img_res.json()["data"][0]["revised_prompt"] in valid_prompts

    # Moderations
    mod_res = client.post("/v1/moderations", json={"input": "Safe technical sentence."})
    assert mod_res.status_code == 200
    assert mod_res.json()["results"][0]["flagged"] is False

    # Files upload, list, retrieve, content, delete
    up_res = client.post(
        "/v1/files",
        files={"file": ("test.jsonl", io.BytesIO(b'{"prompt": "hello"}\n'), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert up_res.status_code == 200
    file_id = up_res.json()["id"]

    assert client.get(f"/v1/files/{file_id}").status_code == 200
    assert client.get(f"/v1/files/{file_id}/content").text == '{"prompt": "hello"}\n'

    # Batches create, retrieve, cancel
    batch_res = client.post(
        "/v1/batches",
        json={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert batch_res.status_code == 200
    batch_id = batch_res.json()["id"]

    assert client.get(f"/v1/batches/{batch_id}").status_code == 200
    cancel_res = client.post(f"/v1/batches/{batch_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "cancelled"

    assert client.delete(f"/v1/files/{file_id}").status_code == 200


def test_official_openai_python_sdk_compatibility() -> None:
    sdk_client = OpenAI(
        api_key="mock-key",
        base_url="http://testserver/v1",
        http_client=client,
    )

    models = sdk_client.models.list()
    assert len(models.data) > 0

    chat = sdk_client.chat.completions.create(
        model=Constants.MODEL_GPT_SOL,
        messages=[{"role": "user", "content": "Hello via official OpenAI SDK!"}],
    )
    valid_contents = {item["content"] for item in MOCK_RESPONSES}
    assert chat.choices[0].message.content in valid_contents

    emb = sdk_client.embeddings.create(
        model=Constants.MODEL_EMBEDDING,
        input="Embed via SDK",
    )
    assert len(emb.data[0].embedding) == Constants.DEFAULT_EMBEDDING_DIMENSIONS


def test_auth_and_latency_middleware(monkeypatch: Any) -> None:
    monkeypatch.setenv("MOCK_API_KEY", "secret-test-key")
    # Missing key on /v1/models should return 401
    unauth = client.get("/v1/models")
    assert unauth.status_code == 401

    # Public health endpoint still returns 200 without key
    assert client.get("/health").status_code == 200

    # Valid key + latency header returns 200
    auth_ok = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer secret-test-key",
            "X-Mock-Latency-Ms": "10",
        },
    )
    assert auth_ok.status_code == 200


def test_v2_and_extended_vllm_endpoints() -> None:
    # v2 & v1 & unversioned aliases
    assert client.post("/v2/rerank", json={"query": "q", "documents": ["d1", "d2"]}).status_code == 200
    assert client.post("/v2/score", json={"text_1": "q", "text_2": ["d1", "d2"]}).status_code == 200
    assert client.post("/v2/classify", json={"input": "test"}).status_code == 200
    assert client.post("/v1/classify", json={"input": "test"}).status_code == 200
    assert client.post("/v2/pooling", json={"input": "test"}).status_code == 200
    assert client.post("/v1/pooling", json={"input": "test"}).status_code == 200
    assert client.post("/v2/embeddings", json={"input": "test"}).status_code == 200
    assert client.post("/embeddings", json={"input": "test"}).status_code == 200
    assert client.post("/v1/tokenize", json={"prompt": "hello"}).status_code == 200
    assert client.post("/v2/tokenize", json={"prompt": "hello"}).status_code == 200
    assert client.post("/v1/detokenize", json={"tokens": [100, 200]}).status_code == 200
    assert client.post("/v2/detokenize", json={"tokens": [100, 200]}).status_code == 200

    # vLLM profiler, cache, elastic EP, collective RPC, server_info, invocations
    assert client.post("/start_profile").status_code == 200
    assert client.post("/stop_profile").status_code == 200
    assert client.post("/reset_mm_cache").status_code == 200
    assert client.post("/reset_encoder_cache").status_code == 200
    assert client.post("/scale_elastic_ep", json={"new_data_parallel_size": 2}).status_code == 200
    assert client.post("/collective_rpc", json={"method": "health_check"}).status_code == 200
    assert client.get("/server_info").status_code == 200
    assert client.get("/v1/server_info").status_code == 200
    assert client.post("/invocations", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 200
    assert client.get("/v1/audio/voices").status_code == 200


def test_extended_openai_endpoints_and_realtime_ws() -> None:
    # Stored Chat Completions CRUD
    chat_res = client.post(
        "/v1/chat/completions",
        json={"model": Constants.MODEL_CLAUDE_FABLE, "messages": [{"role": "user", "content": "Store this completion"}]},
    ).json()
    cid = chat_res["id"]
    assert client.get("/v1/chat/completions").status_code == 200
    assert client.get(f"/v1/chat/completions/{cid}").status_code == 200
    assert client.get(f"/v1/chat/completions/{cid}/messages").status_code == 200
    assert client.post(f"/v1/chat/completions/{cid}", json={"metadata": {"env": "test"}}).status_code == 200
    assert client.delete(f"/v1/chat/completions/{cid}").status_code == 200

    # Fine-Tuning Jobs
    ft = client.post("/v1/fine_tuning/jobs", json={"training_file": Constants.SEED_FILE_RLVR_ID}).json()
    ft_id = ft["id"]
    assert client.get("/v1/fine_tuning/jobs").status_code == 200
    assert client.get(f"/v1/fine_tuning/jobs/{ft_id}").status_code == 200
    assert client.get(f"/v1/fine_tuning/jobs/{ft_id}/events").status_code == 200
    assert client.get(f"/v1/fine_tuning/jobs/{ft_id}/checkpoints").status_code == 200
    assert client.post(f"/v1/fine_tuning/jobs/{ft_id}/cancel").status_code == 200

    # Uploads
    up = client.post("/v1/uploads", json={"filename": "data.jsonl", "bytes": 128, "purpose": "batch"}).json()
    up_id = up["id"]
    part = client.post(
        f"/v1/uploads/{up_id}/parts",
        files={"data": ("part1", io.BytesIO(b"chunk"), "application/octet-stream")},
    ).json()
    assert client.post(f"/v1/uploads/{up_id}/complete", json={"part_ids": [part["id"]]}).status_code == 200

    # Vector Stores
    vs = client.post("/v1/vector_stores", json={"name": "test-vs"}).json()
    vs_id = vs["id"]
    assert client.get("/v1/vector_stores").status_code == 200
    assert client.get(f"/v1/vector_stores/{vs_id}").status_code == 200
    assert client.post(f"/v1/vector_stores/{vs_id}/search", json={"query": "architecture"}).status_code == 200
    assert client.delete(f"/v1/vector_stores/{vs_id}").status_code == 200

    # Assistants, Threads, Messages, Runs
    asst = client.post("/v1/assistants", json={"model": Constants.MODEL_CLAUDE_OPUS, "name": "Test Asst"}).json()
    asst_id = asst["id"]
    thread = client.post("/v1/threads", json={}).json()
    tid = thread["id"]
    assert client.post(f"/v1/threads/{tid}/messages", json={"role": "user", "content": "Hello"}).status_code == 200
    assert client.get(f"/v1/threads/{tid}/messages").status_code == 200
    run = client.post(f"/v1/threads/{tid}/runs", json={"assistant_id": asst_id}).json()
    assert client.get(f"/v1/threads/{tid}/runs/{run['id']}").status_code == 200
    assert client.delete(f"/v1/threads/{tid}").status_code == 200
    assert client.delete(f"/v1/assistants/{asst_id}").status_code == 200

    # Realtime WebSocket
    with client.websocket_connect(f"/v1/realtime?model={Constants.MODEL_GPT_SOL_REALTIME}") as ws:
        created_ev = ws.receive_json()
        assert created_ev["type"] == "session.created"
        assert created_ev["session"]["model"] == Constants.MODEL_GPT_SOL_REALTIME

        ws.send_text(json.dumps({"type": "session.update", "session": {"instructions": "Be concise"}}))
        updated_ev = ws.receive_json()
        assert updated_ev["type"] == "session.updated"
        assert updated_ev["session"]["instructions"] == "Be concise"

        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": "UklGRg=="}))
        speech_ev = ws.receive_json()
        assert speech_ev["type"] == "input_audio_buffer.speech_started"
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        committed_ev = ws.receive_json()
        assert committed_ev["type"] == "input_audio_buffer.committed"

        ws.send_text(json.dumps({"type": "response.create"}))
        resp_created = ws.receive_json()
        assert resp_created["type"] == "response.created"
        item_added = ws.receive_json()
        assert item_added["type"] == "response.output_item.added"
        delta_ev = ws.receive_json()
        assert delta_ev["type"] == "response.text.delta"
        text_done = ws.receive_json()
        assert text_done["type"] == "response.text.done"
        done_ev = ws.receive_json()
        assert done_ev["type"] == "response.done"


def test_bounded_stores_prevent_memory_leaks() -> None:
    # 1. Verify BoundedListStore FIFO eviction cap (count + byte budget)
    bounded_list = BoundedListStore([{"id": i} for i in range(5)], max_size=10, max_bytes=1024 * 1024)
    for i in range(5, 25):
        bounded_list.append({"id": i})
    assert len(bounded_list) == 10
    assert bounded_list[0]["id"] == 15
    assert bounded_list[-1]["id"] == 24

    # Verify byte cap eviction on BoundedListStore
    byte_capped_list = BoundedListStore(max_size=100, max_bytes=200)
    for i in range(10):
        byte_capped_list.append({"payload": "x" * 60, "idx": i})
    assert byte_capped_list.total_bytes <= 200
    assert len(byte_capped_list) <= 3

    # 2. Verify BoundedDictStore LRU/FIFO capacity eviction cap + byte budget
    bounded_dict = BoundedDictStore(max_size=10, max_bytes=1024 * 1024)
    for i in range(25):
        bounded_dict[f"key_{i}"] = {"val": i}
    assert len(bounded_dict) == 10
    assert "key_0" not in bounded_dict
    assert "key_24" in bounded_dict

    byte_capped_dict = BoundedDictStore(max_size=100, max_bytes=250)
    for i in range(10):
        byte_capped_dict[f"k_{i}"] = {"blob": "y" * 70}
    assert byte_capped_dict.total_bytes <= 250
    assert len(byte_capped_dict) <= 3


def test_stdout_json_and_text_logging_real_ip_and_request_id(monkeypatch: Any) -> None:
    import logging
    from main import StdoutStreamingFormatter, logger

    buf = io.StringIO()
    stream_handler = logging.StreamHandler(buf)
    stream_handler.setFormatter(StdoutStreamingFormatter())
    logger.addHandler(stream_handler)

    try:
        # 1. JSON log format + X-Forwarded-For real IP + custom X-Request-ID
        monkeypatch.setenv("MOCK_ENABLE_LOGGING", "true")
        monkeypatch.setenv("MOCK_LOG_FORMAT", "json")
        monkeypatch.setenv("MOCK_ENABLE_OBSERVABILITY", "true")

        res = client.get(
            "/version",
            headers={
                "X-Forwarded-For": "203.0.113.42, 10.0.0.1",
                "X-Request-ID": "req_custom_k8s_trace_01",
            },
        )
        assert res.status_code == 200
        assert res.headers[Constants.HEADER_REQUEST_ID] == "req_custom_k8s_trace_01"
        assert res.headers[Constants.HEADER_CLIENT_IP] == "203.0.113.42"
        assert float(res.headers[Constants.HEADER_PROCESS_TIME_MS]) >= 0.0
        assert res.headers[Constants.HEADER_RESPONSE_TIME].endswith("ms")

        json_log_line = buf.getvalue().strip().splitlines()[-1]
        parsed_log = json.loads(json_log_line)
        assert parsed_log["request_id"] == "req_custom_k8s_trace_01"
        assert parsed_log["client_ip"] == "203.0.113.42"
        assert parsed_log["path"] == "/version"
        assert parsed_log["status_code"] == 200
        assert parsed_log["vllm_version"] == Constants.VLLM_VERSION
        assert "response_time_ms" in parsed_log

        # 2. Text log format + CF-Connecting-IP + auto-generated X-Request-ID
        buf.seek(0)
        buf.truncate(0)
        monkeypatch.setenv("MOCK_LOG_FORMAT", "text")

        res_text = client.get("/health", headers={"CF-Connecting-IP": "198.51.100.77"})
        assert res_text.status_code == 200
        auto_req_id = res_text.headers[Constants.HEADER_REQUEST_ID]
        assert auto_req_id.startswith("req_")
        assert res_text.headers[Constants.HEADER_CLIENT_IP] == "198.51.100.77"

        text_log_line = buf.getvalue().strip().splitlines()[-1]
        assert f"request_id={auto_req_id}" in text_log_line
        assert "client_ip=198.51.100.77" in text_log_line
        assert "GET /health -> 200" in text_log_line

        # 3. Optional disable of observability headers & logging
        buf.seek(0)
        buf.truncate(0)
        monkeypatch.setenv("MOCK_ENABLE_LOGGING", "false")
        monkeypatch.setenv("MOCK_ENABLE_OBSERVABILITY", "false")

        res_disabled = client.get("/health")
        assert res_disabled.status_code == 200
        assert Constants.HEADER_REQUEST_ID not in res_disabled.headers
        assert Constants.HEADER_PROCESS_TIME_MS not in res_disabled.headers
        assert buf.getvalue() == ""
    finally:
        logger.removeHandler(stream_handler)


def test_p0_responses_api_streaming_tool_calls() -> None:
    """Verify P0-1: /v1/responses streaming with tools emits proper function_call SSE events."""
    tools = [
        {
            "type": "function",
            "name": "query_cluster_metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string"},
                    "window_minutes": {"type": "integer"},
                },
                "required": ["cluster_id", "window_minutes"],
            },
        },
        {
            "type": "function",
            "name": "scale_node_pool",
            "parameters": {
                "type": "object",
                "properties": {"replicas": {"type": "integer"}},
            },
        },
    ]
    res = client.post(
        "/v1/responses",
        json={
            "model": Constants.MODEL_GPT_SOL,
            "input": "Check cluster metrics and scale node pool",
            "tools": tools,
            "stream": True,
        },
    )
    assert res.status_code == 200
    raw_lines = [line for line in res.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
    events = [json.loads(line[len("data: ") :]) for line in raw_lines]
    event_types = [ev["type"] for ev in events]

    assert "response.created" in event_types
    assert "response.in_progress" in event_types
    assert "response.output_item.added" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types
    assert "response.output_item.done" in event_types
    assert "response.completed" in event_types
    # Must NOT emit output_text.delta when tool calls are streamed
    assert "response.output_text.delta" not in event_types


def test_p0_sqlite_shared_state_and_startup_warning(tmp_path: Any, monkeypatch: Any) -> None:
    """Verify P0-2: Optional SQLite cross-replica state sync and startup warning when disabled."""
    import logging
    from main import StdoutStreamingFormatter, logger

    # 1. Verify SQLite state sync across two separate BoundedDictStore instances (simulating 2 pods/workers)
    db_file = str(tmp_path / "shared_mock_state.db")
    monkeypatch.setenv("MOCK_STATE_DB_PATH", db_file)

    pod_a_store = BoundedDictStore(max_size=100, store_name="test_cross_pod")
    pod_b_store = BoundedDictStore(max_size=100, store_name="test_cross_pod")

    pod_a_store["resp_shared_1"] = {"id": "resp_shared_1", "status": "completed"}
    # Pod B should immediately retrieve it from SQLite even though it wasn't in Pod B's local memory
    fetched = pod_b_store.get("resp_shared_1")
    assert fetched is not None
    assert fetched["id"] == "resp_shared_1"

    # 2. Verify startup warning when MOCK_STATE_DB_PATH is empty
    monkeypatch.setenv("MOCK_STATE_DB_PATH", "")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(StdoutStreamingFormatter())
    logger.addHandler(handler)
    try:
        with TestClient(app):
            pass
        logs = buf.getvalue()
        assert "MOCK_STATE_DB_PATH is not configured" in logs
        assert "replicas=1" in logs
    finally:
        logger.removeHandler(handler)


def test_p0_deterministic_seed_and_deterministic_mode(monkeypatch: Any) -> None:
    """Verify P0-3: Request seed and MOCK_DETERMINISTIC produce identical outputs across runs."""
    r1 = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_CLAUDE_OPUS,
            "messages": [{"role": "user", "content": "Snapshot test prompt"}],
            "seed": 42,
        },
    ).json()
    r2 = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_CLAUDE_OPUS,
            "messages": [{"role": "user", "content": "Snapshot test prompt"}],
            "seed": 42,
        },
    ).json()
    assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]

    # Also test MOCK_DETERMINISTIC=true without explicit seed
    monkeypatch.setenv("MOCK_DETERMINISTIC", "true")
    d1 = client.post(
        "/v1/completions",
        json={"model": Constants.MODEL_GPT_SOL, "prompt": "Deterministic completion prompt"},
    ).json()
    d2 = client.post(
        "/v1/completions",
        json={"model": Constants.MODEL_GPT_SOL, "prompt": "Deterministic completion prompt"},
    ).json()
    assert d1["choices"][0]["text"] == d2["choices"][0]["text"]


def test_p1_json_schema_synthesis_and_multi_tool_streaming() -> None:
    """Verify P1-1 (JSON Schema recursive synthesis) and P1-2 (multi-tool streaming in Chat API)."""
    schema = {
        "type": "object",
        "properties": {
            "service_name": {"type": "string"},
            "replica_count": {"type": "integer"},
            "healthy": {"type": "boolean"},
            "environment": {"type": "string", "enum": ["prod", "staging"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }
    res = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_GPT_SOL,
            "messages": [{"role": "user", "content": "Return service status adhering to schema"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "service_status", "schema": schema},
            },
            "seed": 7,
        },
    )
    assert res.status_code == 200
    parsed = json.loads(res.json()["choices"][0]["message"]["content"])
    assert isinstance(parsed["service_name"], str)
    assert isinstance(parsed["replica_count"], int)
    assert isinstance(parsed["healthy"], bool)
    assert parsed["environment"] in ["prod", "staging"]
    assert isinstance(parsed["tags"], list)

    # Multi-tool streaming in /v1/chat/completions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_cpu_usage",
                "parameters": {"type": "object", "properties": {"host": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_memory_usage",
                "parameters": {"type": "object", "properties": {"host": {"type": "string"}}},
            },
        },
    ]
    stream_res = client.post(
        "/v1/chat/completions",
        json={
            "model": Constants.MODEL_CLAUDE_SONNET,
            "messages": [{"role": "user", "content": "Check CPU and memory usage"}],
            "tools": tools,
            "stream": True,
        },
    )
    assert stream_res.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in stream_res.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    streamed_indices = {
        tc["index"]
        for c in chunks
        for tc in c["choices"][0].get("delta", {}).get("tool_calls", [])
    }
    assert streamed_indices == {0, 1}


def test_p1_reversible_tokenizer_and_lexical_reranker() -> None:
    """Verify P1-3 (lossless tokenize/detokenize round-trip) and P1-6 (lexical rerank relevance ordering)."""
    original_text = "Hello, world! Testing reversible BPE-style mock tokenization 2026."
    tok_res = client.post(
        "/tokenize",
        json={"model": Constants.MODEL_GPT_SOL, "prompt": original_text, "add_special_tokens": True},
    )
    assert tok_res.status_code == 200
    tokens = tok_res.json()["tokens"]

    detok_res = client.post(
        "/detokenize",
        json={"model": Constants.MODEL_GPT_SOL, "tokens": tokens},
    )
    assert detok_res.status_code == 200
    assert detok_res.json()["prompt"] == original_text

    # Verify /v1/rerank ranks the matching document first
    rerank_res = client.post(
        "/v1/rerank",
        json={
            "model": Constants.MODEL_RERANKER,
            "query": "Kubernetes pod horizontal autoscaling CPU utilization",
            "documents": [
                "Baking sourdough bread requires flour, water, salt, and starter fermentation.",
                "Kubernetes HorizontalPodAutoscaler automatically scales pods based on CPU utilization metrics.",
                "Medieval castles featured stone curtain walls and defensive moats.",
            ],
        },
    )
    assert rerank_res.status_code == 200
    results = rerank_res.json()["results"]
    assert results[0]["index"] == 1
    assert results[0]["relevance_score"] > results[1]["relevance_score"]


def test_p1_chaos_error_simulation_and_413_file_limit(monkeypatch: Any) -> None:
    """Verify P1-4 (chaos headers 429/500/503 & malformed SSE) and P1-5 (413 file size enforcement)."""
    # 1. Deterministic status error injection via X-Mock-Error-Status
    for status_code in (429, 500, 503):
        err_res = client.post(
            "/v1/chat/completions",
            headers={Constants.HEADER_MOCK_ERROR_STATUS: str(status_code)},
            json={"model": Constants.MODEL_GPT_SOL, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert err_res.status_code == status_code
        assert "error" in err_res.json()

    # 2. Malformed SSE stream injection via X-Mock-Malformed-SSE
    malformed_res = client.post(
        "/v1/chat/completions",
        headers={Constants.HEADER_MOCK_MALFORMED_SSE: "true"},
        json={"model": Constants.MODEL_GPT_SOL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert malformed_res.status_code == 200
    assert "truncated_malformed_sse_frame" in malformed_res.text

    # 3. File upload exceeding MOCK_MAX_FILE_BYTES returns HTTP 413
    monkeypatch.setenv("MOCK_MAX_FILE_BYTES", "64")
    oversized_res = client.post(
        "/v1/files",
        files={"file": ("oversized.jsonl", b"x" * 128, "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert oversized_res.status_code == 413


