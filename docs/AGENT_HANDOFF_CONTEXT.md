# Agent Handoff Context

Use this file when switching between Claude Code, Codex, or another coding assistant after quota limits, context loss, or a long break. It is intentionally operational: it says what this repo is, what is implemented, how to verify it, and where the next assistant should look first.

## Project Snapshot

Repository: `Multimodal_Document_QA_Inference_Lab`

Purpose: a small FastAPI-based lab for multimodal document question answering and benchmarking. It is not a general chatbot. The intended use is to upload a document image or PDF, ask for structured fields, validate the model output against a schema, and benchmark latency plus accuracy.

Current implementation status:

- FastAPI app exists.
- In-memory document upload exists.
- Image and PDF ingestion exist.
- Mock backend exists and is used by default.
- Optional HuggingFace Transformers backend exists.
- JSON extraction, Pydantic validation, and retry logic exist.
- Benchmark runner and CLI exist.
- Synthetic invoice dataset generator exists.
- Tests exist for API, mock backend, schema parsing, metrics, field matching, and benchmark runner.

Not implemented yet:

- Routing layer.
- OCR-only path.
- vLLM backend.
- llama.cpp/GGUF backend.
- INT8/INT4 quantization.
- Persistent database-backed run history.
- Dashboard or frontend.
- Async benchmark queue.

## Important Files

Start here:

- `README.md`: quickstart and high-level architecture.
- `CLAUDE.md`: original architectural contracts and roadmap.
- `app/main.py`: FastAPI app factory and route registration.
- `app/api/routes_documents.py`: upload endpoint.
- `app/api/routes_qa.py`: QA endpoint.
- `app/api/routes_benchmark.py`: benchmark endpoint.
- `app/api/deps.py`: dependency providers, singleton document store, backend selection.
- `app/backends/base.py`: backend interface contract.
- `app/backends/mock_backend.py`: deterministic mock backend for tests and CI.
- `app/backends/transformers_backend.py`: optional real VLM backend.
- `app/ingestion/document_store.py`: in-memory document storage.
- `app/evals/schema_validation.py`: JSON extraction, schema validation, retry loop.
- `app/benchmarking/runner.py`: dataset benchmark loop and aggregate report.
- `scripts/generate_sample_data.py`: generates synthetic invoice examples.
- `scripts/benchmark.py`: benchmark CLI entry point.
- `tests/`: behavior currently protected by tests.

## Common Commands

Install dependencies:

```bash
uv sync --extra dev
```

Generate sample invoice data:

```bash
uv run python scripts/generate_sample_data.py
```

Run tests:

```bash
uv run pytest -q
```

Start API:

```bash
uv run uvicorn app.main:app --reload
```

Run mock benchmark:

```bash
uv run python scripts/benchmark.py --dataset data/sample_invoices --backend mock --output results/smoke.json
```

Run real Transformers backend, if ML dependencies and model download are available:

```bash
uv sync --extra dev --extra ml
uv run python scripts/benchmark.py --dataset data/sample_invoices --backend transformers --model Qwen/Qwen2-VL-2B-Instruct --quantization fp16 --output results/qwen_fp16.json
```

## Runtime Behavior

Upload flow:

1. `POST /documents/upload` receives an image or PDF.
2. `DocumentStore` converts it into one or more `PIL.Image` pages.
3. Images are resized so the longest edge is at most `MMI_MAX_IMAGE_EDGE`, default `1280`.
4. A short random `file_id` is returned.
5. The document only lives in memory.

QA flow:

1. `POST /qa` receives a `file_id`, question, output mode, schema, and optional backend settings.
2. The uploaded document is looked up in the in-memory store.
3. The first page is passed to the selected backend.
4. For JSON mode, the prompt asks for exact schema keys.
5. The raw backend answer is parsed as JSON.
6. Parsed JSON is validated with a Pydantic schema.
7. If parsing or validation fails, the backend is retried with a stricter prompt.
8. The API returns the parsed answer plus metrics.

Benchmark flow:

1. `run_benchmark()` reads `labels.jsonl`.
2. Each row points to an image file and expected structured fields.
3. Each image/question pair is sent through the same backend and schema-validation logic used by the API.
4. Predicted fields are compared to expected fields.
5. A report is returned and optionally written to JSON.

## Key Contracts

Every backend must subclass `ModelBackend` and implement:

```python
def generate(image, prompt, generation_config=None) -> BackendResponse: ...
def get_model_info() -> dict: ...
```

Every `BackendResponse` must contain:

```json
{
  "answer": "...",
  "tokens_generated": 128,
  "time_to_first_token_ms": 240.0,
  "total_latency_ms": 1220.0,
  "tokens_per_second": 42.1,
  "peak_memory_mb": 7420.0,
  "backend": "backend-name",
  "model": "model-name",
  "quantization": "fp16"
}
```

Benchmarking depends on this response shape. Do not change it casually.

## Environment Variables

Settings use the `MMI_` prefix through Pydantic settings.

Known settings:

- `MMI_DEFAULT_BACKEND`
- `MMI_DEFAULT_MODEL`
- `MMI_DEFAULT_QUANTIZATION`
- `MMI_TRANSFORMERS_MODEL_ID`
- `MMI_MAX_IMAGE_EDGE`

Defaults are in `app/config.py`.

## Current Risks And Limitations

- Uploaded documents are not persistent.
- `get_backend()` can raise `ValueError` for unknown backend names; some API paths do not convert that into a clean HTTP error yet.
- Only the first document page is used for QA.
- The mock backend returns generic valid invoice JSON when no canned answer is registered, so schema-valid rate can be high even when field accuracy is low.
- `TransformersBackend` estimates TTFT by doing a separate one-token generation before the full generation, so timings are useful but not equivalent to streaming TTFT.
- `app/benchmarking/profiler.py` exists but is not currently used by `run_benchmark()`.
- INT8 and INT4 intentionally raise `NotImplementedError`.

## Safe Next-Step Checklist

When resuming work:

1. Run `git status --short`.
2. Read this file, `README.md`, and the specific files related to the requested task.
3. Do not assume roadmap items in `CLAUDE.md` are implemented.
4. Prefer adding tests when changing API behavior, schema parsing, backend contracts, or benchmark output.
5. Run `uv run pytest -q` before committing when dependencies are installed.
6. Keep backend response shape stable.
7. Avoid persistent storage or routing refactors unless the user explicitly asks for that milestone.

## Suggested Commit Style

Use focused commit messages such as:

- `docs: add handoff context`
- `docs: explain project architecture`
- `feat: add backend registry error handling`
- `test: cover benchmark failure path`

## GitHub Push Notes

If no remote exists, add one before pushing:

```bash
git remote add origin <github-repo-url>
git push -u origin main
```

If a remote already exists:

```bash
git push
```
