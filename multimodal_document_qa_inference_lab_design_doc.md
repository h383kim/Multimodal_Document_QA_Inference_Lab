# Project 1 Design Doc: Multimodal Document QA Inference Lab

## 1. Project Summary

Build a multimodal document question-answering inference platform that allows users to upload documents/images and ask questions about them. The main goal is not just to answer questions, but to **benchmark and optimize inference performance** across different serving backends, quantization levels, batching strategies, and output formats.

The system should support document/image inputs such as:

- receipts
- invoices
- screenshots
- forms
- PDF pages
- charts/tables
- scanned documents

The core output should be either:

- natural language answer
- structured JSON answer
- extracted fields

Example:

```text
Input: invoice.pdf
Question: What is the invoice number, total amount, and due date?

Output:
{
  "invoice_number": "INV-10293",
  "total_amount": "$1,248.50",
  "due_date": "2026-05-15"
}
```

---

## 2. Main Goal

The main goal is to build a **production-style inference lab** for multimodal models.

This should demonstrate:

- multimodal model serving
- inference backend comparison
- quantization trade-offs
- latency profiling
- throughput benchmarking
- structured output generation
- document QA evaluation
- cache-aware and cost-aware routing
- API and frontend deployment

The project should be framed as:

> A benchmark and optimization platform for multimodal document QA inference.

Not simply:

> A chatbot over documents.

---

## 3. Target Resume Framing

Possible resume title:

> Built a multimodal document QA inference platform benchmarking VLM serving backends, quantization levels, batching strategies, and structured output reliability.

Possible resume bullets:

```text
Built a multimodal document QA inference lab supporting PDF/image ingestion, VLM-based question answering, structured JSON extraction, and benchmarked inference performance across latency, throughput, memory, and answer quality.

Implemented an evaluation harness measuring TTFT, tokens/sec, p50/p95 latency, GPU memory usage, field-level extraction accuracy, and schema-valid output rate across FP16, INT8, INT4/GGUF, and backend configurations.

Designed a cost-aware routing layer that selects OCR-only, small VLM, or large VLM inference paths based on query/document complexity, reducing average latency while preserving answer accuracy.
```

---

## 4. System Architecture

High-level architecture:

```text
Frontend / CLI
    |
    v
FastAPI Backend
    |
    +--> Document Ingestion Module
    |       - PDF to image conversion
    |       - image resizing
    |       - OCR extraction
    |       - metadata extraction
    |
    +--> Inference Router
    |       - OCR-only route
    |       - small VLM route
    |       - large VLM route
    |       - backend selector
    |
    +--> Model Serving Layer
    |       - Hugging Face Transformers backend
    |       - vLLM backend, if supported
    |       - llama.cpp / GGUF backend, optional
    |
    +--> Structured Output Layer
    |       - JSON schema validation
    |       - retry-on-invalid-output
    |       - field normalization
    |
    +--> Benchmarking Layer
    |       - latency measurement
    |       - memory measurement
    |       - throughput measurement
    |       - quality metrics
    |
    +--> Evaluation Store
            - benchmark results
            - model config
            - quantization config
            - outputs
            - correctness labels
```

---

## 5. Suggested Tech Stack

Core:

```text
Python
FastAPI
PyTorch
Transformers
Pydantic
Pandas
NumPy
SQLite or DuckDB
Docker
```

Optional but valuable:

```text
vLLM
llama.cpp
GGUF models
bitsandbytes
AutoAWQ or GPTQModel
Streamlit or React frontend
Prometheus/Grafana
Locust or custom load tester
```

Document processing:

```text
PyMuPDF or pdf2image
Pillow
pytesseract or easyocr
opencv-python
```

Evaluation:

```text
pytest
jsonschema or Pydantic
scikit-learn metrics
matplotlib
```

---

## 6. Model Options

Choose models based on available hardware.

Possible VLMs:

```text
Qwen2-VL / Qwen2.5-VL
LLaVA
InternVL
Phi-3.5 Vision
MiniCPM-V
```

For lower-resource local experiments:

```text
small VLM model through Transformers
quantized model through bitsandbytes
GGUF-compatible model through llama.cpp, if available
```

The system should be model-agnostic through an adapter interface.

Example:

```python
class ModelBackend:
    def generate(self, image, prompt, generation_config):
        raise NotImplementedError

    def get_model_info(self):
        raise NotImplementedError
```

Backends:

```text
TransformersBackend
VLLMBackend
LlamaCppBackend
MockBackend for testing
```

---

## 7. Core Features

### Feature 1: Document/Image Upload

Support:

```text
PDF
PNG
JPG
JPEG
```

For PDFs:

```text
Convert first N pages to images
Allow selecting page range
Store page-level metadata
```

For images:

```text
Normalize size
Preserve aspect ratio
Optionally downsample for speed
```

---

### Feature 2: Question Answering

User provides:

```text
document/image
question
desired output mode
```

Output modes:

```text
natural_language
json
field_extraction
```

Example request:

```json
{
  "file_id": "invoice_001",
  "question": "Extract invoice number, vendor name, due date, and total amount.",
  "output_schema": {
    "invoice_number": "string",
    "vendor_name": "string",
    "due_date": "date",
    "total_amount": "string"
  }
}
```

---

### Feature 3: Structured JSON Output

Use Pydantic schemas.

Example:

```python
class InvoiceExtraction(BaseModel):
    invoice_number: str | None
    vendor_name: str | None
    due_date: str | None
    total_amount: str | None
```

The system should:

```text
ask model for JSON
parse output
validate schema
retry if invalid
record schema success/failure
```

Metrics:

```text
schema_valid_rate
retry_count
final_parse_success
field_missing_rate
```

---

### Feature 4: Inference Backend Comparison

Support multiple backends through a common interface.

Each backend should return:

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

---

### Feature 5: Quantization Benchmarking

Compare:

```text
FP16/BF16 baseline
INT8
INT4
GGUF quantization, if supported
```

For each config, measure:

```text
model loading time
peak memory
TTFT
total latency
tokens/sec
answer quality
schema valid rate
```

Example benchmark matrix:

```text
Model: Qwen2-VL small
Backend: Transformers
Precision: FP16, INT8, INT4

Model: GGUF-compatible model
Backend: llama.cpp
Precision: Q4_K_M, Q5_K_M, Q8_0
```

---

### Feature 6: Routing Layer

Implement a simple cost-aware router.

Possible routes:

```text
OCR-only route:
    Use OCR + regex/rules for simple extraction.

Small VLM route:
    Use small/quantized model for simple visual QA.

Large VLM route:
    Use larger model for harder visual reasoning.
```

Routing logic can start simple:

```python
if question asks for simple field and OCR confidence is high:
    use OCR route
elif image has clear text and question is extraction-heavy:
    use small VLM
else:
    use large VLM
```

Later, add classifier-based routing.

Routing metrics:

```text
route_selected
route_latency
route_accuracy
route_cost_estimate
fallback_count
```

---

### Feature 7: Benchmarking CLI

Example commands:

```bash
python benchmark.py   --dataset data/invoices   --model qwen-vl-small   --backend transformers   --quantization int4   --batch-size 1   --output results/qwen_int4.json
```

```bash
python benchmark.py   --dataset data/forms   --compare-configs configs/benchmark_matrix.yaml
```

Benchmark config example:

```yaml
dataset: data/invoices
tasks:
  - invoice_extraction
  - receipt_total
  - document_question_answering

configs:
  - name: transformers_fp16
    backend: transformers
    model: qwen-vl-small
    quantization: fp16

  - name: transformers_int4
    backend: transformers
    model: qwen-vl-small
    quantization: int4

  - name: llama_cpp_gguf_q4
    backend: llama_cpp
    model_path: models/model-q4.gguf
    quantization: q4
```

---

## 8. Dataset Design

Start with a small custom dataset.

Example structure:

```text
data/
  invoices/
    images/
      invoice_001.png
      invoice_002.png
    labels.jsonl

  receipts/
    images/
      receipt_001.png
      receipt_002.png
    labels.jsonl

  screenshots/
    images/
      screenshot_001.png
    labels.jsonl
```

Label example:

```json
{
  "id": "invoice_001",
  "file": "images/invoice_001.png",
  "question": "Extract invoice number, due date, and total amount.",
  "expected": {
    "invoice_number": "INV-1002",
    "due_date": "2026-05-20",
    "total_amount": "$392.12"
  },
  "schema": "invoice_extraction"
}
```

Can use:

```text
synthetic invoices
public receipt datasets
hand-created screenshots
forms generated with Python
```

Important: the dataset does not need to be huge. The benchmark system matters more.

---

## 9. Evaluation Metrics

### Inference Performance

```text
TTFT: time to first token
total latency
tokens/sec
p50 latency
p95 latency
throughput requests/sec
peak CPU memory
peak GPU memory
model load time
```

### Output Quality

```text
exact match
normalized exact match
field-level accuracy
JSON schema valid rate
missing field rate
hallucinated field rate
semantic similarity, optional
```

### Production Metrics

```text
retry rate
fallback rate
route distribution
cost estimate per request
batch utilization
cold start latency
warm latency
```

---

## 10. API Design

### Upload document

```http
POST /documents/upload
```

Response:

```json
{
  "file_id": "doc_123",
  "num_pages": 2,
  "document_type": "pdf"
}
```

### Ask question

```http
POST /qa
```

Request:

```json
{
  "file_id": "doc_123",
  "question": "What is the total amount?",
  "output_mode": "json",
  "schema_name": "invoice_extraction",
  "backend": "transformers",
  "model": "qwen-vl-small",
  "quantization": "int4"
}
```

Response:

```json
{
  "answer": {
    "total_amount": "$392.12"
  },
  "metrics": {
    "ttft_ms": 220,
    "total_latency_ms": 1050,
    "tokens_per_second": 38.5,
    "peak_memory_mb": 6200
  },
  "backend": "transformers",
  "quantization": "int4"
}
```

### Run benchmark

```http
POST /benchmarks/run
```

---

## 11. Repository Structure

```text
multimodal-inference-lab/
  README.md
  pyproject.toml
  Dockerfile

  app/
    main.py
    api/
      routes_documents.py
      routes_qa.py
      routes_benchmark.py

    ingestion/
      pdf_loader.py
      image_preprocessor.py
      ocr.py

    backends/
      base.py
      transformers_backend.py
      vllm_backend.py
      llama_cpp_backend.py
      mock_backend.py

    routing/
      router.py
      complexity.py

    schemas/
      outputs.py
      requests.py

    evals/
      metrics.py
      field_match.py
      schema_validation.py

    benchmarking/
      runner.py
      profiler.py
      load_test.py

    storage/
      db.py
      models.py

  configs/
    benchmark_matrix.yaml
    models.yaml

  data/
    sample_invoices/
    sample_receipts/
    labels.jsonl

  notebooks/
    analysis_latency.ipynb
    analysis_quantization.ipynb

  tests/
    test_schema_validation.py
    test_metrics.py
    test_router.py
```

---

## 12. Implementation Milestones

### Milestone 1: Basic Document QA

Goal:

```text
Upload image/PDF
Ask question
Get VLM answer
```

Tasks:

```text
FastAPI app
PDF/image loader
one Transformers VLM backend
basic prompt template
simple response
```

---

### Milestone 2: Structured Output

Goal:

```text
Return validated JSON answers
```

Tasks:

```text
Pydantic schemas
JSON extraction
retry on invalid JSON
field normalization
schema success metric
```

---

### Milestone 3: Benchmark Harness

Goal:

```text
Run batch benchmarks over dataset
```

Tasks:

```text
JSONL dataset loader
benchmark runner
latency profiler
memory profiler
CSV/JSON result writer
basic plots
```

---

### Milestone 4: Quantization Comparison

Goal:

```text
Compare FP16, INT8, INT4, optionally GGUF
```

Tasks:

```text
bitsandbytes loading
quantization config support
memory/latency comparison
quality comparison table
```

---

### Milestone 5: Routing Layer

Goal:

```text
Route easy cases to cheaper inference path
```

Tasks:

```text
OCR route
small VLM route
large VLM route
routing logs
accuracy/latency trade-off evaluation
```

---

### Milestone 6: Frontend / Dashboard

Goal:

```text
Show upload, answer, and benchmark dashboard
```

Can use:

```text
Streamlit for speed
or React for polished UI
```

Dashboard should show:

```text
latency by backend
memory by quantization
accuracy by config
schema-valid rate
example outputs
```

---

## 13. Stretch Goals

```text
Continuous batching simulation
Streaming responses
Canary deployment between model versions
Prompt prefix caching for repeated document types
OpenTelemetry tracing
Docker Compose deployment
GPU utilization profiling with nvidia-smi
Automatic model fallback on OOM
Cost-per-1K-document estimator
```

---

## 14. What Makes This Resume-Strong

This project should emphasize:

```text
Inference engineering, not just model usage
Latency/throughput/memory benchmarking
Quantization trade-offs
Production-style API design
Evaluation harness
Structured output reliability
Routing and fallback strategies
```

The final README should include charts like:

```text
Latency vs quantization
Memory vs quantization
Accuracy vs quantization
TTFT vs backend
p95 latency under load
Route accuracy/latency tradeoff
```
