from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prov_entry(
    provider: str,
    source_type: str,
    method: str,
    inputs: List[str],
    fetched_at: Optional[str] = None,
    note: Optional[str] = None,
    validated_by: Optional[List[str]] = None,
) -> dict:
    return {
        "source_type": source_type,
        "provider": provider,
        "method": method,
        "inputs": inputs,
        "validated_by": validated_by or [],
        "fetched_at": fetched_at or now_iso(),
        "note": note,
    }


def conflict_entry(
    provider: str,
    method: str,
    inputs: List[str],
    note: str,
    validators: Optional[List[str]] = None,
) -> dict:
    return prov_entry(
        provider=provider,
        source_type="conflict",
        method=method,
        inputs=inputs,
        note=note,
        validated_by=validators or [],
    )


def missing_entry(
    provider: str,
    method: str,
    inputs: List[str],
    note: Optional[str] = None,
) -> dict:
    return prov_entry(
        provider=provider,
        source_type="missing",
        method=method,
        inputs=inputs,
        note=note or "not deterministically computable",
    )

