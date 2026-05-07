"""Append-only JSONL log of routing decisions.

Each line records what the router chose, why, and enough request context
(``request_id``, ``schema_name``, ``timestamp``) for later analysis of
accuracy/latency tradeoffs across paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.routing.types import RouteDecision


def append_decision(
    log_path: Path,
    decision: RouteDecision,
    *,
    request_id: str | None = None,
    schema_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "schema_name": schema_name,
        **decision.to_dict(),
    }
    if extra:
        record.update(extra)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
