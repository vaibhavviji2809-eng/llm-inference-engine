from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .dashboard import render_dashboard
from .metrics import metrics_store
from .runtime import DEFAULT_CHECKPOINT, get_runtime


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    use_kv_cache: bool = True


class ChatRequest(BaseModel):
    messages: list[str]
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    use_kv_cache: bool = True


class BenchmarkRequest(BaseModel):
    prompt: str
    sample_tokens: int = Field(default=24, ge=1, le=128)
    repeats: int = Field(default=10, ge=1, le=100)


class BatchGenerateRequest(BaseModel):
    prompts: list[str]
    max_new_tokens: int = Field(default=16, ge=1, le=128)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class BatchBenchmarkRequest(BaseModel):
    prompts: list[str]
    sample_tokens: int = Field(default=16, ge=1, le=128)
    repeats: int = Field(default=5, ge=1, le=100)


app = FastAPI(title="LLM Inference Engine")


@app.get("/health")
def health() -> dict:
    checkpoint_exists = DEFAULT_CHECKPOINT.exists()
    return {
        "status": "ok",
        "checkpoint_exists": checkpoint_exists,
        "runtime_ready": checkpoint_exists,
    }


@app.get("/model")
def model_info() -> dict:
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    return runtime.model_card()


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    try:
        result = runtime.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            use_cache=request.use_kv_cache,
        )
        metrics_store.record_request(result)
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    prompt = "\n".join(request.messages)
    try:
        result = runtime.generate_text(
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            use_cache=request.use_kv_cache,
        )
        metrics_store.record_request(result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "messages": request.messages,
        "response": result["completion"],
        "latency_seconds": result["latency_seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "use_kv_cache": result["use_kv_cache"],
    }


@app.post("/benchmark")
def benchmark(request: BenchmarkRequest) -> dict:
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    try:
        result = runtime.benchmark(
            prompt=request.prompt,
            max_new_tokens=request.sample_tokens,
            repeats=request.repeats,
        )
        metrics_store.record_benchmark(result)
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/batch_generate")
def batch_generate(request: BatchGenerateRequest) -> dict:
    if not request.prompts:
        raise HTTPException(status_code=400, detail="prompts must not be empty.")
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    try:
        result = runtime.batch_generate_text(
            prompts=request.prompts,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        metrics_store.record_batch_run(result)
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/benchmark_batching")
def benchmark_batching(request: BatchBenchmarkRequest) -> dict:
    if not request.prompts:
        raise HTTPException(status_code=400, detail="prompts must not be empty.")
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    try:
        result = runtime.benchmark_batching(
            prompts=request.prompts,
            max_new_tokens=request.sample_tokens,
            repeats=request.repeats,
        )
        metrics_store.record_benchmark(result)
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/stream")
def stream(request: GenerateRequest) -> StreamingResponse:
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=404,
            detail="Model checkpoint not found. Run scripts/train_tiny_transformer.py first.",
        )
    runtime = get_runtime()
    try:
        chunks = runtime.stream_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            use_cache=request.use_kv_cache,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    async def event_stream():
        for chunk in chunks:
            yield f"data: {chunk}\\n\\n"
        yield "data: [DONE]\\n\\n"

    metrics_store.record_request(
        {
            "prompt": request.prompt,
            "latency_seconds": None,
            "tokens_per_second": None,
            "use_kv_cache": request.use_kv_cache,
            "streaming": True,
        }
    )
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    return metrics_store.summary()


@app.get("/dashboard")
def dashboard():
    return render_dashboard()
