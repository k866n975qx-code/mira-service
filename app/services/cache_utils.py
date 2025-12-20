from __future__ import annotations

import hashlib
import json
import os
import time
from tempfile import NamedTemporaryFile
from typing import Any, Optional


# Simple disk-backed TTL cache shared across workers. Defaults to .snapshot_cache/ttl.
def _base_dir() -> str:
    default = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".snapshot_cache", "ttl"))
    return os.getenv("MIRA_TTL_CACHE_DIR", default)


def _safe_namespace(ns: str) -> str:
    return "".join(ch for ch in (ns or "default") if ch.isalnum() or ch in ("_", "-")).strip() or "default"


def _cache_path(namespace: str, key: str) -> str:
    ns = _safe_namespace(namespace)
    hashed = hashlib.sha1(key.encode("utf-8")).hexdigest()
    base = os.path.join(_base_dir(), ns)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{hashed}.json")


def load_ttl_cache(namespace: str, key: str) -> Optional[Any]:
    """
    Return cached data if present and unexpired; otherwise None.
    """
    path = _cache_path(namespace, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        expires_at = float(payload.get("expires_at", 0))
    except Exception:
        return None
    if expires_at <= time.time():
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    return payload.get("data")


def store_ttl_cache(namespace: str, key: str, data: Any, ttl_seconds: int) -> None:
    """
    Persist data with a TTL. Overwrites existing entry atomically.
    """
    path = _cache_path(namespace, key)
    payload = {"data": data, "expires_at": time.time() + ttl_seconds}
    tmp: Optional[NamedTemporaryFile] = None
    try:
        tmp = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path))
        json.dump(payload, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    finally:
        if tmp and not tmp.closed:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
