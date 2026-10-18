"""
Pydantic v2 request and response schemas for OpenAI & vLLM compatible endpoints.
Uses `extra="allow"` on request models so any vLLM or OpenAI extension parameter
(e.g., guided_json, top_k, min_p, repetition_penalty, chat_template_kwargs) is accepted.
All default model identifiers are sourced from `Constants`.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from constants import Constants


class FlexibleRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Common / Shared Usage Models
# ---------------------------------------------------------------------------
class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


# ---------------------------------------------------------------------------
# Chat Completions (/v1/chat/completions)
# ---------------------------------------------------------------------------
class ChatMessage(FlexibleRequestModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class StreamOptions(FlexibleRequestModel):
    include_usage: bool = False
    continuous_usage_stats: bool = False


class ChatCompletionRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_CHAT_MODEL
    messages: list[ChatMessage | dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = True
    reasoning_effort: str | None = None
    store: bool | None = False
    # vLLM specific extensions
    best_of: int | None = None
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float | None = 1.0
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    prompt_logprobs: int | None = None
    guided_json: str | dict[str, Any] | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
    guided_decoding_backend: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    include_reasoning: bool | None = None


# ---------------------------------------------------------------------------
# Legacy Completions (/v1/completions)
# ---------------------------------------------------------------------------
class CompletionRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_COMPLETION_MODEL
    prompt: str | list[str] | list[int] | list[list[int]] = ""
    suffix: str | None = None
    max_tokens: int | None = 64
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    logprobs: int | None = None
    echo: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    best_of: int | None = 1
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    # vLLM extensions
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    guided_json: str | dict[str, Any] | None = None
    guided_choice: list[str] | None = None


# ---------------------------------------------------------------------------
# OpenAI / vLLM Responses API (/v1/responses)
# ---------------------------------------------------------------------------
class ResponsesRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_RESPONSES_MODEL
    input: str | list[dict[str, Any]] = ""
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = True
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    previous_response_id: str | None = None
    seed: int | None = None
    store: bool = True


# ---------------------------------------------------------------------------
# Embeddings & Pooling (/v1/embeddings, /pooling)
# ---------------------------------------------------------------------------
class EmbeddingRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_EMBEDDING_MODEL
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None
    seed: int | None = None


class PoolingRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_POOLING_MODEL
    input: str | list[str] | list[int] | list[list[int]] = ""
    messages: list[dict[str, Any]] | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None


# ---------------------------------------------------------------------------
# vLLM Score, Rerank, Classify (/v1/score, /v1/rerank, /classify)
# ---------------------------------------------------------------------------
class ScoreRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_SCORE_MODEL
    text_1: str | list[str]
    text_2: str | list[str]
    truncate_prompt_tokens: int | None = None


class RerankRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_RERANK_MODEL
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = None
    return_documents: bool = True
    truncate_prompt_tokens: int | None = None


class ClassifyRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_CLASSIFY_MODEL
    input: str | list[str]
    truncate_prompt_tokens: int | None = None


# ---------------------------------------------------------------------------
# vLLM Tokenizer (/tokenize, /detokenize)
# ---------------------------------------------------------------------------
class TokenizeRequest(FlexibleRequestModel):
    model: str | None = Constants.DEFAULT_TOKENIZE_MODEL
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    add_special_tokens: bool = True
    return_token_strs: bool = False


class DetokenizeRequest(FlexibleRequestModel):
    model: str | None = Constants.DEFAULT_TOKENIZE_MODEL
    tokens: list[int]


# ---------------------------------------------------------------------------
# vLLM LoRA Adapter Management
# ---------------------------------------------------------------------------
class LoadLoraAdapterRequest(FlexibleRequestModel):
    lora_name: str
    lora_path: str


class UnloadLoraAdapterRequest(FlexibleRequestModel):
    lora_name: str
    lora_int_id: int | None = None


# ---------------------------------------------------------------------------
# Audio, Images, Moderations, Batches
# ---------------------------------------------------------------------------
class AudioSpeechRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_SPEECH_MODEL
    input: str
    voice: str = Constants.DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0


class ImageGenerationRequest(FlexibleRequestModel):
    prompt: str
    model: str | None = Constants.DEFAULT_IMAGE_MODEL
    n: int = 1
    quality: str = "hd"
    response_format: Literal["url", "b64_json"] = "url"
    size: str = "1024x1024"
    style: str = "vivid"
    user: str | None = None


class ModerationRequest(FlexibleRequestModel):
    input: str | list[str] | list[dict[str, Any]]
    model: str = Constants.DEFAULT_MODERATION_MODEL


class BatchCreateRequest(FlexibleRequestModel):
    input_file_id: str
    endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"
    metadata: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# vLLM Distributed / Profiler / Elastic EP / Invocations
# ---------------------------------------------------------------------------
class ScaleElasticEPRequest(FlexibleRequestModel):
    new_data_parallel_size: int = 1


class CollectiveRpcRequest(FlexibleRequestModel):
    method: str = "health_check"
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None


# ---------------------------------------------------------------------------
# OpenAI Fine-Tuning, Uploads, Vector Stores, Assistants & Threads
# ---------------------------------------------------------------------------
class FineTuningJobCreateRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_FINE_TUNE_MODEL
    training_file: str
    validation_file: str | None = None
    hyperparameters: dict[str, Any] | None = None
    suffix: str | None = None
    seed: int | None = 42


class UploadCreateRequest(FlexibleRequestModel):
    filename: str
    purpose: str = "batch"
    bytes: int
    mime_type: str = "application/jsonl"


class UploadCompleteRequest(FlexibleRequestModel):
    part_ids: list[str]
    md5: str | None = None


class VectorStoreCreateRequest(FlexibleRequestModel):
    name: str = "frontier-2026-knowledge-base"
    file_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class VectorStoreSearchRequest(FlexibleRequestModel):
    query: str | list[str]
    max_num_results: int = 10
    filters: dict[str, Any] | None = None
    rewrite_query: bool = False


class AssistantCreateRequest(FlexibleRequestModel):
    model: str = Constants.DEFAULT_ASSISTANT_MODEL
    name: str | None = "2026 Frontier Systems Architect Assistant"
    description: str | None = None
    instructions: str | None = "You are a helpful technical assistant."
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class ThreadCreateRequest(FlexibleRequestModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class ThreadMessageCreateRequest(FlexibleRequestModel):
    role: str = "user"
    content: str | list[dict[str, Any]] = ""
    metadata: dict[str, Any] | None = None


class ThreadRunCreateRequest(FlexibleRequestModel):
    assistant_id: str
    model: str | None = None
    instructions: str | None = None
    stream: bool = False
