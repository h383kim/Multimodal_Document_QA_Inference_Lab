"""HuggingFace Transformers backend (opt-in; requires `pip install -e .[ml]`).

Default model: Qwen/Qwen2-VL-2B-Instruct (~4.4GB). On Apple Silicon, runs on MPS in
fp16; on Linux GPU it can run in bf16. INT8/INT4 quantization requires CUDA +
bitsandbytes and is intentionally raised as NotImplementedError so milestone 4 has
a clean extension point.
"""

from __future__ import annotations

import time
from typing import Any

import psutil
from PIL.Image import Image

from app.backends.base import BackendResponse, ModelBackend


class TransformersBackend(ModelBackend):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        quantization: str = "fp16",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        if quantization in {"int8", "int4"}:
            raise NotImplementedError(
                f"quantization='{quantization}' requires CUDA + bitsandbytes; not wired up in MVP"
            )
        if quantization not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"unsupported quantization '{quantization}'")

        # Lazy imports — keeps the module importable when torch/transformers aren't installed.
        import torch
        from transformers import AutoProcessor

        self._torch = torch
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[quantization]
        # CPU + fp16 is slow and unstable; bump to fp32.
        if device == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = self._load_model(model_id, dtype, device)
        self.model.eval()

    def _load_model(self, model_id: str, dtype: Any, device: str) -> Any:
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except (ImportError, ValueError):
            from transformers import Qwen2VLForConditionalGeneration

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        model.to(device)
        return model

    def generate(
        self,
        image: Image,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
    ) -> BackendResponse:
        torch = self._torch
        gen_cfg = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        if generation_config:
            gen_cfg.update(generation_config)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        proc = psutil.Process()
        rss_before = proc.memory_info().rss / (1024 * 1024)

        # Approximate TTFT by timing a 1-token generate, then full generate.
        start = time.perf_counter_ns()
        with torch.inference_mode():
            ttft_out = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        ttft_ns = time.perf_counter_ns() - start
        del ttft_out

        gen_start = time.perf_counter_ns()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_cfg)
        total_ns = time.perf_counter_ns() - gen_start

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        tokens_generated = int(new_tokens.shape[1])
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        rss_after = proc.memory_info().rss / (1024 * 1024)
        peak_rss = max(rss_before, rss_after)

        total_ms = total_ns / 1_000_000.0
        ttft_ms = ttft_ns / 1_000_000.0
        tps = (tokens_generated / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        return BackendResponse(
            answer=decoded,
            tokens_generated=tokens_generated,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
            peak_memory_mb=peak_rss,
            backend="transformers",
            model=self.model_id,
            quantization=self.quantization,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model": self.model_id,
            "quantization": self.quantization,
            "device": self.device,
        }
