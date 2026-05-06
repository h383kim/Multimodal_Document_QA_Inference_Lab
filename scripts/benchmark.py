"""CLI entry point for the benchmark runner.

    python scripts/benchmark.py --dataset data/sample_invoices --backend mock --output results/smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.deps import get_backend  # noqa: E402
from app.benchmarking.runner import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Directory containing labels.jsonl + images/")
    parser.add_argument("--backend", default="mock", choices=["mock", "transformers"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--quantization", default="fp16")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    backend = get_backend(args.backend, args.model, args.quantization)
    report = run_benchmark(
        dataset_dir=args.dataset,
        backend=backend,
        output_path=args.output,
        max_retries=args.max_retries,
    )
    summary = {
        "n": report["aggregate"]["n"],
        "p50_latency_ms": round(report["aggregate"]["p50_latency_ms"], 2),
        "p95_latency_ms": round(report["aggregate"]["p95_latency_ms"], 2),
        "mean_tokens_per_second": round(report["aggregate"]["mean_tokens_per_second"], 2),
        "schema_valid_rate": round(report["aggregate"]["schema_valid_rate"], 3),
        "field_accuracy_mean": round(report["aggregate"]["field_accuracy_mean"], 3),
        "retry_rate": round(report["aggregate"]["retry_rate"], 3),
        "backend": report["backend"],
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        print(f"\nfull report → {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
