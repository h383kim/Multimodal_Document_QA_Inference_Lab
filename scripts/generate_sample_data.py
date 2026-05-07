"""Render a tiny synthetic invoice dataset with known ground truth.

Output:
    data/sample_invoices/images/invoice_001.png ... invoice_005.png
    data/sample_invoices/labels.jsonl

Run from the repo root:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INVOICES = [
    {
        "id": "invoice_001",
        "invoice_number": "INV-10293",
        "vendor_name": "Northwind Supplies",
        "due_date": "2026-05-15",
        "total_amount": "$1,248.50",
    },
    {
        "id": "invoice_002",
        "invoice_number": "INV-44120",
        "vendor_name": "Acme Logistics",
        "due_date": "2026-06-01",
        "total_amount": "$392.12",
    },
    {
        "id": "invoice_003",
        "invoice_number": "INV-77001",
        "vendor_name": "Globex Corp",
        "due_date": "2026-04-30",
        "total_amount": "$10,500.00",
    },
    {
        "id": "invoice_004",
        "invoice_number": "INV-00892",
        "vendor_name": "Initech Services",
        "due_date": "2026-07-10",
        "total_amount": "$67.45",
    },
    {
        "id": "invoice_005",
        "invoice_number": "INV-55234",
        "vendor_name": "Soylent Foods",
        "due_date": "2026-05-22",
        "total_amount": "$2,310.99",
    },
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try a few common system fonts; fall back to PIL's default.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_invoice(meta: dict[str, str], path: Path) -> None:
    img = Image.new("RGB", (1024, 720), color="white")
    draw = ImageDraw.Draw(img)

    title_font = _font(40)
    label_font = _font(22)
    value_font = _font(26)
    small_font = _font(16)

    draw.text((40, 40), "INVOICE", fill="black", font=title_font)
    draw.line([(40, 100), (984, 100)], fill="black", width=2)

    rows = [
        ("Invoice Number", meta["invoice_number"]),
        ("Vendor", meta["vendor_name"]),
        ("Due Date", meta["due_date"]),
        ("Total Amount", meta["total_amount"]),
    ]
    y = 160
    for label, value in rows:
        draw.text((60, y), label, fill="#666", font=label_font)
        draw.text((360, y - 4), value, fill="black", font=value_font)
        y += 70

    draw.text((40, 660), "Generated for benchmarking purposes only.", fill="#888", font=small_font)
    img.save(path, format="PNG")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / "sample_invoices"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as f:
        for meta in INVOICES:
            image_rel = f"images/{meta['id']}.png"
            render_invoice(meta, images_dir / f"{meta['id']}.png")
            row = {
                "id": meta["id"],
                "file": image_rel,
                "question": "Extract the invoice number, vendor name, due date, and total amount.",
                "expected": {
                    "invoice_number": meta["invoice_number"],
                    "vendor_name": meta["vendor_name"],
                    "due_date": meta["due_date"],
                    "total_amount": meta["total_amount"],
                },
                "schema": "invoice_extraction",
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(INVOICES)} invoices to {out_dir}")


if __name__ == "__main__":
    main()
