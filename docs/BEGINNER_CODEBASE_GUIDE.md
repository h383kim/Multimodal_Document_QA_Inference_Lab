# Beginner Codebase Guide

This guide explains the project for someone who is new to the code and to many of the concepts behind it. It starts with the plain-language idea, then explains the moving parts and how a request travels through the system.

## What This Project Is

This project is a document question-answering and benchmarking lab.

In simple terms, it tries to answer questions about documents such as invoices and receipts. For example:

> Here is an invoice image. What is the invoice number, vendor name, due date, and total amount?

The project is also a benchmark tool. That means it can run the same document questions many times and measure:

- How fast the model responds.
- Whether the model returns valid JSON.
- Whether the extracted fields are correct.
- How many retries were needed.
- Rough memory usage.

The project is not meant to be a chat app. Its focus is structured document understanding.

## Important Concepts

### FastAPI

FastAPI is the web framework used by this project. It lets Python code expose HTTP endpoints.

An endpoint is a URL that another program can call. This project has endpoints such as:

- `GET /health`
- `POST /documents/upload`
- `POST /qa`
- `POST /benchmarks/run`

The FastAPI app is created in `app/main.py`.

### API Request And Response

An API request is input sent to the server.

An API response is output returned by the server.

For example, a QA request contains:

- The uploaded document ID.
- The question.
- The output mode.
- The schema name.
- Optional backend/model settings.

The response contains:

- The answer.
- Runtime metrics.
- Backend/model information.

The request and response shapes are defined with Pydantic models in `app/schemas/requests.py`.

### Pydantic

Pydantic is a Python library for validating data.

This project uses it to say, "An invoice extraction answer should have these fields":

- `invoice_number`
- `vendor_name`
- `due_date`
- `total_amount`

Those output schemas live in `app/schemas/outputs.py`.

If the model returns malformed data or the wrong shape, Pydantic helps detect that.

### Multimodal Model

A multimodal model can understand more than one kind of input. In this project, the important inputs are:

- Image input: the document page.
- Text input: the question or instruction.

The optional real backend uses a vision-language model from HuggingFace Transformers. The default backend is a mock backend so tests and benchmarks can run without downloading a large model.

### Backend

A backend is the component that actually generates an answer.

The project currently has:

- `MockBackend`: fake deterministic backend for tests and local smoke runs.
- `TransformersBackend`: real HuggingFace model backend, if ML dependencies are installed.

Both must follow the same interface from `app/backends/base.py`.

This is important because the API and benchmark code should not care which backend is being used. They only care that the backend has a `generate()` method and returns the expected fields.

### Structured Output

Structured output means the answer must follow a predictable format.

Instead of returning:

```text
The invoice number is INV-10293 and the total is $1,248.50.
```

the project wants:

```json
{
  "invoice_number": "INV-10293",
  "vendor_name": "Northwind Supplies",
  "due_date": "2026-05-15",
  "total_amount": "$1,248.50"
}
```

Structured output is easier to test, store, compare, and use in other software.

### Benchmark

A benchmark is a repeatable test that measures performance.

This project's benchmark runner loads a dataset, asks the backend questions, compares answers against known expected values, and calculates metrics.

The main benchmark code is in `app/benchmarking/runner.py`.

## Repository Layout

The most important directories are:

```text
app/
  api/           FastAPI route handlers and dependencies
  backends/      model backend interface and implementations
  ingestion/     image/PDF loading and preprocessing
  schemas/       Pydantic request and output schemas
  evals/         JSON parsing, schema validation, accuracy metrics
  benchmarking/  benchmark runner and profiling helpers

scripts/
  generate_sample_data.py  creates synthetic invoice images
  benchmark.py             command-line benchmark runner

tests/
  automated tests

configs/
  benchmark configuration examples

docs/
  extra documentation
```

## How The App Starts

The entry point is `app/main.py`.

It defines `create_app()`, which:

1. Creates a FastAPI application.
2. Adds the document routes.
3. Adds the QA route.
4. Adds the benchmark route.
5. Adds a simple `/health` route.

At the bottom of the file:

```python
app = create_app()
```

This gives Uvicorn something to run when you start the server with:

```bash
uv run uvicorn app.main:app --reload
```

## Uploading A Document

The upload endpoint is in `app/api/routes_documents.py`.

The endpoint is:

```text
POST /documents/upload
```

It accepts a file. The code checks whether the file is a PDF by looking at:

- The filename.
- The content type.

If it is a PDF, the code calls:

```python
store.add_pdf(raw)
```

If it is not a PDF, the code calls:

```python
store.add_image(raw)
```

The `store` is a `DocumentStore`.

## DocumentStore

`DocumentStore` is in `app/ingestion/document_store.py`.

It is an in-memory dictionary. That means documents are stored only while the Python server process is running.

Each stored document has:

- `file_id`
- `document_type`
- `pages`

The pages are PIL images.

When an image is added, the store:

1. Opens the image bytes with Pillow.
2. Resizes it if needed.
3. Converts it to RGB.
4. Generates a short random `file_id`.
5. Saves it in memory.

When a PDF is added, the store:

1. Renders each PDF page to an image using PyMuPDF.
2. Resizes each image if needed.
3. Stores the pages in memory.

## Asking A Question

The QA endpoint is in `app/api/routes_qa.py`.

The endpoint is:

```text
POST /qa
```

The request includes fields like:

```json
{
  "file_id": "abc123",
  "question": "Extract invoice fields",
  "output_mode": "json",
  "schema_name": "invoice_extraction",
  "backend": "mock"
}
```

The flow is:

1. Look up the document using `file_id`.
2. Make sure the document exists.
3. Select a backend.
4. Build a prompt.
5. Send the first page image and prompt to the backend.
6. If JSON mode is used, parse and validate the backend answer.
7. Return the answer and metrics.

Currently, only the first page of a document is used for QA.

## Prompt Building

The `_build_prompt()` function in `app/api/routes_qa.py` creates the instruction sent to the backend.

For JSON mode, it asks the backend to return exactly the schema fields.

For invoice extraction, the prompt tells the backend to return keys like:

```text
invoice_number, vendor_name, due_date, total_amount
```

This makes the model more likely to return a predictable answer.

## Backend Selection

Backend selection happens in `app/api/deps.py`.

The default backend is `mock`.

If the request says:

```json
"backend": "mock"
```

then the app uses `MockBackend`.

If the request says:

```json
"backend": "transformers"
```

then the app tries to use `TransformersBackend`.

The Transformers backend requires extra ML dependencies and may download a large model.

## MockBackend

`MockBackend` is in `app/backends/mock_backend.py`.

It is useful because it lets the rest of the system be tested without a real AI model.

It can:

- Return canned answers for known images.
- Return a generic fallback JSON answer.
- Simulate latency.
- Simulate a bad first response to test retry behavior.

The mock backend returns valid metrics so the benchmark system can still calculate performance numbers.

## TransformersBackend

`TransformersBackend` is in `app/backends/transformers_backend.py`.

It uses HuggingFace Transformers to load a real vision-language model.

Default model:

```text
Qwen/Qwen2-VL-2B-Instruct
```

It chooses a device automatically:

1. CUDA GPU, if available.
2. Apple MPS, if available.
3. CPU, otherwise.

Supported quantization values in the current MVP:

- `fp16`
- `bf16`
- `fp32`

`int8` and `int4` intentionally raise `NotImplementedError`.

## JSON Parsing And Retry

The JSON parsing logic is in `app/evals/schema_validation.py`.

AI models often return extra text. For example:

```text
Here is the JSON:
{"invoice_number": "INV-1"}
```

The parser tries to extract just the JSON part.

It handles:

- Bare JSON.
- JSON inside code fences.
- JSON embedded inside prose.

After extracting JSON, it validates the result with the requested Pydantic schema.

If parsing or validation fails, the code retries with a stricter prompt:

```text
Reply with ONLY a single valid JSON object...
```

The number of retries is tracked in the response metrics.

## Accuracy Measurement

Accuracy helpers are in `app/evals/field_match.py`.

The project compares predicted fields with expected fields.

Before comparing, it normalizes values by:

- Lowercasing.
- Removing non-alphanumeric characters.

This means these can match:

```text
INV-001
inv001
```

Field accuracy is the fraction of fields that match.

If 3 out of 4 fields match, field accuracy is `0.75`.

## Benchmark Runner

The benchmark runner is in `app/benchmarking/runner.py`.

It expects a dataset directory like:

```text
data/sample_invoices/
  images/
    invoice_001.png
    invoice_002.png
  labels.jsonl
```

Each line in `labels.jsonl` is one example:

```json
{
  "id": "invoice_001",
  "file": "images/invoice_001.png",
  "question": "Extract the invoice number, vendor name, due date, and total amount.",
  "expected": {
    "invoice_number": "INV-10293",
    "vendor_name": "Northwind Supplies",
    "due_date": "2026-05-15",
    "total_amount": "$1,248.50"
  },
  "schema": "invoice_extraction"
}
```

For each row, the benchmark runner:

1. Loads the image.
2. Builds a prompt.
3. Calls the backend.
4. Parses and validates JSON.
5. Compares predicted fields to expected fields.
6. Records latency and accuracy metrics.

At the end, it calculates aggregate metrics.

## Benchmark Metrics

Important benchmark metrics include:

- `n`: number of examples.
- `p50_latency_ms`: median-ish latency.
- `p95_latency_ms`: high-percentile latency.
- `mean_latency_ms`: average latency.
- `mean_ttft_ms`: average time to first token.
- `mean_tokens_per_second`: average generation speed.
- `schema_valid_rate`: fraction of outputs that passed schema validation.
- `field_accuracy_mean`: average field-level accuracy.
- `retry_rate`: fraction of examples that needed at least one retry.

## Sample Data Generator

`scripts/generate_sample_data.py` creates fake invoice images.

It writes:

```text
data/sample_invoices/images/*.png
data/sample_invoices/labels.jsonl
```

This is useful for smoke testing the benchmark pipeline.

Run it with:

```bash
uv run python scripts/generate_sample_data.py
```

## Tests

Tests are in the `tests/` directory.

They check things like:

- The health endpoint works.
- Uploading and then asking QA works.
- Unknown `file_id` returns a 404.
- The mock backend returns the required response shape.
- JSON can be extracted from different text formats.
- Retry behavior works.
- Benchmark aggregation works.
- Field matching works.

Run tests with:

```bash
uv run pytest -q
```

## What To Understand First

If you are new to the project, understand these in order:

1. `app/main.py` starts the API.
2. `app/api/routes_documents.py` uploads documents.
3. `app/ingestion/document_store.py` stores document pages in memory.
4. `app/api/routes_qa.py` handles questions.
5. `app/backends/base.py` defines what all backends must return.
6. `app/backends/mock_backend.py` gives fake model answers.
7. `app/evals/schema_validation.py` turns raw model text into validated JSON.
8. `app/benchmarking/runner.py` runs many examples and calculates metrics.

## Current Limitations

This project is a working MVP, not a full production system.

Current limitations:

- Uploaded files disappear when the server restarts.
- Only the first page is used for QA.
- There is no frontend.
- There is no OCR routing yet.
- There is no database for benchmark history.
- Real model inference requires optional ML dependencies.
- Real model inference may require downloading a large model.
- INT8 and INT4 quantization are not implemented.

## Mental Model

The simplest way to think about the system is:

```text
Document file
  -> image pages
  -> prompt + first page
  -> backend generates text
  -> JSON is extracted
  -> Pydantic validates fields
  -> metrics and accuracy are calculated
```

Once that flow makes sense, the rest of the codebase becomes much easier to follow.
