"""Streamlit QA workbench for the multimodal document QA lab."""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.api_client import ApiClientError, LabApiClient, UploadedDocument
from ui.document_preview import PreviewError, build_preview


DEFAULT_API_BASE_URL = "http://localhost:8000"
SCHEMA_OPTIONS = ["invoice_extraction", "receipt_extraction"]
OUTPUT_MODES = ["json", "natural_language", "field_extraction"]
BACKENDS = ["mock", "transformers"]
QUANTIZATIONS = ["fp16", "bf16", "fp32", "int8", "int4"]


def main() -> None:
    st.set_page_config(
        page_title="Document QA Workbench",
        page_icon="",
        layout="wide",
    )
    _ensure_state()

    controls = _render_sidebar()
    client = LabApiClient(controls["api_base_url"])

    st.title("Document QA Workbench")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        uploaded = _render_upload(client)
        question = st.text_area(
            "Question",
            value="Extract the invoice number, vendor name, due date, and total amount.",
            height=110,
        )
        run_disabled = uploaded is None or not question.strip()
        if st.button("Run QA", type="primary", disabled=run_disabled, use_container_width=True):
            _run_qa(client, uploaded, question, controls)

    with right:
        _render_answer()
        _render_metrics()


def _ensure_state() -> None:
    st.session_state.setdefault("api_base_url", DEFAULT_API_BASE_URL)
    st.session_state.setdefault("uploaded_document", None)
    st.session_state.setdefault("uploaded_file_signature", None)
    st.session_state.setdefault("qa_response", None)


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("Connection")
        api_base_url = st.text_input("API base URL", key="api_base_url")
        _render_health(api_base_url)

        st.header("Inference")
        backend = st.selectbox("Backend", BACKENDS, index=0)
        if backend == "mock":
            st.caption(
                "Mock uses canned answers for generated sample invoices. "
                "Use transformers for real document reading."
            )
        model = st.text_input("Model override", value="")
        quantization = st.selectbox("Quantization", QUANTIZATIONS, index=0)
        output_mode = st.selectbox("Output mode", OUTPUT_MODES, index=0)
        schema_name = st.selectbox("Schema", SCHEMA_OPTIONS, index=0)
        max_retries = st.number_input("Max retries", min_value=0, max_value=10, value=2, step=1)

    return {
        "api_base_url": api_base_url,
        "backend": backend,
        "model": model.strip() or None,
        "quantization": quantization,
        "output_mode": output_mode,
        "schema_name": schema_name if output_mode == "json" else None,
        "max_retries": int(max_retries),
    }


def _render_health(api_base_url: str) -> None:
    client = LabApiClient(api_base_url)
    try:
        health = client.health()
    except ApiClientError as exc:
        st.error(str(exc))
        st.code("uv run uvicorn app.main:app --reload", language="bash")
        return
    st.success(f"{health.get('status', 'ok')} - {health.get('version', 'unknown')}")


def _render_upload(client: LabApiClient) -> UploadedDocument | None:
    uploaded_file = st.file_uploader(
        "Document",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        st.session_state.uploaded_document = None
        st.session_state.uploaded_file_signature = None
        st.info("Upload a document to start.")
        return None

    file_bytes = uploaded_file.getvalue()
    signature = (uploaded_file.name, uploaded_file.size)
    if signature != st.session_state.uploaded_file_signature:
        st.session_state.uploaded_document = None
        st.session_state.uploaded_file_signature = signature
        st.session_state.qa_response = None

    try:
        preview = build_preview(file_bytes, uploaded_file.name, uploaded_file.type or "")
        st.image(preview.image, caption=preview.label, use_container_width=True)
    except PreviewError as exc:
        st.warning(str(exc))

    if st.button("Upload", use_container_width=True):
        try:
            st.session_state.uploaded_document = client.upload_document(
                filename=uploaded_file.name,
                content=file_bytes,
                content_type=uploaded_file.type or "application/octet-stream",
            )
        except ApiClientError as exc:
            _render_api_error("Upload failed", exc)

    uploaded = st.session_state.uploaded_document
    if uploaded is not None:
        cols = st.columns(3)
        cols[0].metric("Pages", uploaded.num_pages)
        cols[1].metric("Type", uploaded.document_type)
        cols[2].metric("File ID", uploaded.file_id)
    return uploaded


def _run_qa(
    client: LabApiClient,
    uploaded: UploadedDocument | None,
    question: str,
    controls: dict[str, Any],
) -> None:
    if uploaded is None:
        return
    try:
        st.session_state.qa_response = client.ask_qa(
            file_id=uploaded.file_id,
            question=question.strip(),
            output_mode=controls["output_mode"],
            schema_name=controls["schema_name"],
            backend=controls["backend"],
            model=controls["model"],
            quantization=controls["quantization"],
            max_retries=controls["max_retries"],
        )
    except ApiClientError as exc:
        _render_api_error("QA failed", exc)


def _render_answer() -> None:
    st.subheader("Answer")
    response = st.session_state.qa_response
    if response is None:
        st.info("Run QA to see the answer.")
        return

    answer = response.get("answer")
    if isinstance(answer, dict):
        rows = [{"field": key, "value": value} for key, value in answer.items()]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        with st.expander("JSON"):
            st.json(answer)
    else:
        st.write(answer)

    with st.expander("Response"):
        st.json(response)


def _render_metrics() -> None:
    st.subheader("Metrics")
    response = st.session_state.qa_response
    if response is None:
        st.info("Run QA to see metrics.")
        return

    metrics = response.get("metrics", {})
    first = st.columns(3)
    first[0].metric("TTFT", f"{metrics.get('ttft_ms', 0):.1f} ms")
    first[1].metric("Latency", f"{metrics.get('total_latency_ms', 0):.1f} ms")
    first[2].metric("Tokens/sec", f"{metrics.get('tokens_per_second', 0):.1f}")

    second = st.columns(3)
    second[0].metric("Peak memory", f"{metrics.get('peak_memory_mb', 0):.1f} MB")
    second[1].metric("Retries", metrics.get("retry_count", 0))
    second[2].metric("Schema valid", str(metrics.get("schema_valid", False)))

    backend = response.get("backend", "unknown")
    model = response.get("model", "unknown")
    quantization = response.get("quantization", "unknown")
    st.caption(f"{backend} / {model} / {quantization}")


def _render_api_error(title: str, exc: ApiClientError) -> None:
    prefix = f"{title}: "
    if exc.status_code is not None:
        prefix += f"HTTP {exc.status_code} - "
    st.error(prefix + str(exc))


if __name__ == "__main__":
    main()
