# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a **multimodal document QA inference benchmarking platform** — not a chatbot. The framing is: a production-style lab for benchmarking VLM serving backends, quantization levels, batching strategies, and structured output reliability. See `multimodal_document_qa_inference_lab_design_doc.md` for the full design.

## Planned Commands

Once implemented, the primary commands will be:

```bash
# Start the API server
uvicorn app.main:app --reload

# Run a benchmark sweep
python benchmark.py --dataset data/invoices --model qwen-vl-small --backend transformers --quantization int4 --output results/run.json

# Run against a benchmark matrix config
python benchmark.py --dataset data/forms --compare-configs configs/benchmark_matrix.yaml

# Run tests
pytest tests/
```

## Planned Architecture

The system has six distinct layers:

1. **Document Ingestion** (`app/ingestion/`) — PDF→image conversion, resizing, OCR, metadata extraction
2. **Inference Router** (`app/routing/`) — selects OCR-only, small VLM, or large VLM path based on query/document complexity
3. **Model Serving Layer** (`app/backends/`) — pluggable backends behind a common `ModelBackend` interface:
   - `TransformersBackend` — HuggingFace Transformers, supports FP16/INT8/INT4 via bitsandbytes
   - `VLLMBackend` — vLLM serving
   - `LlamaCppBackend` — GGUF quantized models
   - `MockBackend` — testing only
4. **Structured Output Layer** (`app/evals/schema_validation.py`) — Pydantic schema enforcement, retry-on-invalid-JSON, field normalization
5. **Benchmarking Layer** (`app/benchmarking/`) — latency/memory/throughput profiling; results written to SQLite or DuckDB
6. **FastAPI Backend** (`app/main.py`) — routes: `POST /documents/upload`, `POST /qa`, `POST /benchmarks/run`

## Key Design Constraints

**Backend interface** — every backend must implement:
```python
class ModelBackend:
    def generate(self, image, prompt, generation_config): ...
    def get_model_info(self): ...
```

**Backend response shape** — every inference call must return these fields (used by benchmarking layer):
```json
{
  "answer": "...",
  "tokens_generated": 128,
  "time_to_first_token_ms": 240,
  "total_latency_ms": 1220,
  "tokens_per_second": 42.1,
  "peak_memory_mb": 7420,
  "backend": "transformers",
  "model": "model-name",
  "quantization": "fp16"
}
```

**Routing logic** starts rule-based (OCR confidence + question type) and should be designed to swap in a classifier later. Route decisions must be logged for accuracy/latency tradeoff analysis.

**Structured output** — the system must ask the model for JSON, parse, validate with Pydantic, and retry on failure. Track `schema_valid_rate`, `retry_count`, and `field_missing_rate` per benchmark run.

## Dataset Format

Labels are stored as JSONL:
```json
{"id": "invoice_001", "file": "images/invoice_001.png", "question": "...", "expected": {...}, "schema": "invoice_extraction"}
```

## Tech Stack

Core: Python, FastAPI, PyTorch, Transformers, Pydantic, Pandas, SQLite/DuckDB, Docker

Document processing: PyMuPDF or pdf2image, Pillow, pytesseract or easyocr

Quantization: bitsandbytes (INT8/INT4), AutoAWQ or GPTQModel, llama.cpp (GGUF)

Target VLMs: Qwen2-VL / Qwen2.5-VL (primary), LLaVA, InternVL, Phi-3.5 Vision, MiniCPM-V

## Implementation Milestones

1. Basic Document QA — FastAPI + one Transformers VLM backend
2. Structured Output — Pydantic schemas, JSON retry loop, schema metrics
3. Benchmark Harness — JSONL loader, latency/memory profiler, CSV/JSON results
4. Quantization Comparison — FP16 vs INT8 vs INT4 (vs GGUF optional)
5. Routing Layer — OCR → small VLM → large VLM with routing logs
6. Frontend/Dashboard — Streamlit (fast) or React (polished); show latency/memory/accuracy charts per config
