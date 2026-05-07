# Multimodal Document QA Inference Lab

A benchmark and optimization platform for multimodal document QA inference — not a chatbot. It serves a FastAPI surface for document upload + structured QA, plus a benchmark CLI that sweeps backends/quantization configs over a labeled dataset and reports TTFT, p50/p95 latency, tokens/sec, peak memory, schema-valid rate, and field-level accuracy.

See `multimodal_document_qa_inference_lab_design_doc.md` for the full design and `CLAUDE.md` for the architectural contracts.

## Quickstart

Requires [`uv`](https://github.com/astral-sh/uv). Python 3.11 is auto-installed by uv.

```bash
# 1. Install deps
uv sync --extra dev

# 2. Generate a synthetic invoice dataset
uv run python scripts/generate_sample_data.py
# → data/sample_invoices/images/*.png + labels.jsonl

# 3. Run the test suite
uv run pytest -q

# 4. Start the API
uv run uvicorn app.main:app --reload
# → docs at http://localhost:8000/docs

# 5. Optional: start the Streamlit QA workbench
uv sync --extra dev --extra ui
uv run --extra ui streamlit run ui/streamlit_app.py
# → UI at http://localhost:8501

# 6. Run the benchmark CLI (mock backend — no model download)
uv run python scripts/benchmark.py \
  --dataset data/sample_invoices --backend mock \
  --output results/smoke.json
```

API smoke (against the running server):

```bash
curl -F "file=@data/sample_invoices/images/invoice_001.png" \
  http://localhost:8000/documents/upload
# → {"file_id":"...","num_pages":1,"document_type":"image"}

curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"file_id":"<file_id>","question":"Extract invoice fields",
       "output_mode":"json","schema_name":"invoice_extraction","backend":"mock"}'
```

## Streamlit UI

The UI is a separate Streamlit app that talks to the FastAPI backend over HTTP.
Start the API first, then launch the workbench:

```bash
uv run uvicorn app.main:app --reload
uv run --extra ui streamlit run ui/streamlit_app.py
```

The workbench supports document upload, local image/PDF first-page preview,
structured QA, backend/schema controls, API health status, and inference metrics.
The default API URL is `http://localhost:8000` and can be changed in the sidebar.
The default `mock` backend uses canned answers for generated sample invoices; use
the `transformers` backend for real document reading.

## Real VLM (opt-in)

The `transformers` backend loads `Qwen/Qwen2-VL-2B-Instruct` (~4.4GB) via HuggingFace. It picks `cuda` → `mps` → `cpu` automatically; uses fp16 on GPU/MPS and fp32 on CPU. INT8/INT4 quantization is intentionally raised as `NotImplementedError` — wiring `bitsandbytes` is milestone 4.

```bash
# install ML deps
uv sync --extra dev --extra ui --extra ml

# benchmark with the real model
uv run python scripts/benchmark.py \
  --dataset data/sample_invoices --backend transformers \
  --model Qwen/Qwen2-VL-2B-Instruct --quantization fp16 \
  --output results/qwen_fp16.json
```

## Architecture

Six layers behind a common backend interface (see `CLAUDE.md`):

```
Document Ingestion (PDF→PNG, resize) → Inference Router → Model Backend
                                                              ↓
                       Structured Output (JSON parse + retry + Pydantic)
                                                              ↓
                                    Benchmarking (latency, memory, accuracy)
```

Every backend implements `app.backends.base.ModelBackend` and returns a fixed
`BackendResponse` dict — that contract is what the benchmark layer keys off.

## Adding a backend

1. Subclass `ModelBackend` in `app/backends/<name>_backend.py`
2. Implement `generate(image, prompt, generation_config) -> BackendResponse`
3. Implement `get_model_info() -> dict`
4. Register it in `app/api/deps.py::_build_backend`
5. Add to the `--backend` choices in `scripts/benchmark.py`

The `MockBackend` (`app/backends/mock_backend.py`) is the simplest reference.

## Layout

```
app/
  api/          # FastAPI routes (documents, qa, benchmark)
  backends/     # ModelBackend ABC + mock + transformers
  ingestion/    # PDF → image, resize, in-memory document store
  schemas/      # Pydantic request + output models
  evals/        # JSON parse-with-retry, field match, percentile/mean helpers
  benchmarking/ # profiler context manager + dataset runner
ui/             # Streamlit QA workbench + API/preview helpers
configs/        # benchmark_matrix.yaml
scripts/        # generate_sample_data.py, benchmark.py
tests/          # tests covering all of the above
```

## What the MVP does *not* do (next milestones)

- Routing layer (OCR / small VLM / large VLM cost-aware path selection)
- vLLM / llama.cpp backends
- INT8 / INT4 / GGUF quantization (CUDA-only path stubbed with a clean error)
- Full benchmark dashboard
- Persistent SQLite-backed run history (file-based JSON results suffice for now)
