"""Download a public document-QA dataset and convert it to the lab's JSONL row format.

Requires the ``datasets`` library (HuggingFace) — install on demand::

    uv pip install datasets

Examples::

    python scripts/download_datasets.py --dataset cord --split test --limit 50 \\
        --out data/datasets/cord_test

    python scripts/download_datasets.py --dataset sroie --split test --limit 50 \\
        --out data/datasets/sroie_test

    python scripts/download_datasets.py --dataset docvqa --split validation \\
        --limit 100 --out data/datasets/docvqa_val

The output directory will contain ``images/`` plus ``labels.jsonl``, which is the
shape expected by ``app.benchmarking.runner.run_benchmark``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from app.ingestion.dataset_adapters import JsonlRow, cord_row, docvqa_row, sroie_row

DATASET_SOURCES: dict[str, str] = {
    "cord": "naver-clova-ix/cord-v2",
    "sroie": "darentang/sroie",
    "docvqa": "lmms-lab/DocVQA",
}


def _save_image(image: Any, path: Path) -> None:
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path, format="PNG")
        return
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            opened.convert("RGB").save(path, format="PNG")
        return
    raise TypeError(f"unsupported image type for save: {type(image)}")


def _iter_cord(rows: Iterator[dict[str, Any]], image_dir: Path) -> Iterator[JsonlRow]:
    for index, row in enumerate(rows):
        ground_truth_raw = row.get("ground_truth")
        if isinstance(ground_truth_raw, str):
            ground_truth = json.loads(ground_truth_raw)
        else:
            ground_truth = ground_truth_raw or {}
        item_id = f"cord_{index:05d}"
        file_rel = f"images/{item_id}.png"
        _save_image(row["image"], image_dir.parent / file_rel)
        yield cord_row(item_id=item_id, file=file_rel, ground_truth=ground_truth)


def _iter_sroie(rows: Iterator[dict[str, Any]], image_dir: Path) -> Iterator[JsonlRow]:
    for index, row in enumerate(rows):
        item_id = f"sroie_{index:05d}"
        file_rel = f"images/{item_id}.png"
        _save_image(row["image"], image_dir.parent / file_rel)
        yield sroie_row(item_id=item_id, file=file_rel, fields=row)


def _iter_docvqa(rows: Iterator[dict[str, Any]], image_dir: Path) -> Iterator[JsonlRow]:
    for index, row in enumerate(rows):
        question_id = row.get("questionId") or row.get("question_id") or index
        item_id = f"docvqa_{question_id}"
        file_rel = f"images/{item_id}.png"
        _save_image(row["image"], image_dir.parent / file_rel)
        yield docvqa_row(
            item_id=item_id,
            file=file_rel,
            question=row.get("question", ""),
            answers=row.get("answers") or row.get("answer") or [],
        )


_ADAPTERS: dict[str, Callable[[Iterator[dict[str, Any]], Path], Iterator[JsonlRow]]] = {
    "cord": _iter_cord,
    "sroie": _iter_sroie,
    "docvqa": _iter_docvqa,
}


def download_and_convert(
    dataset: str,
    split: str,
    limit: int | None,
    output_dir: Path,
) -> int:
    if dataset not in DATASET_SOURCES:
        raise ValueError(f"unknown dataset '{dataset}'. Known: {sorted(DATASET_SOURCES)}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "the `datasets` library is required for this script.\n"
            "Install with: uv pip install datasets"
        ) from exc

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"

    source = DATASET_SOURCES[dataset]
    hf_dataset = load_dataset(source, split=split, streaming=False)
    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    adapter = _ADAPTERS[dataset]
    written = 0
    with labels_path.open("w", encoding="utf-8") as out:
        for row in adapter(iter(hf_dataset), image_dir):
            out.write(json.dumps(row) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SOURCES))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory; will contain images/ and labels.jsonl",
    )
    args = parser.parse_args()

    output_dir: Path = args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    written = download_and_convert(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        output_dir=output_dir,
    )
    print(f"Wrote {written} rows to {output_dir / 'labels.jsonl'}")


if __name__ == "__main__":
    main()
