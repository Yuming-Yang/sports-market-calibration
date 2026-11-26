"""Minimal HTTP access with an on-disk cache.

All network access in the package goes through `get_json` and `get_bytes`, so tests can
run with no network by pre-filling the cache or by monkeypatching `_fetch`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "sports-market-calibration/0.1"
# Polymarket documents limits of hundreds of requests per 10 seconds. One request every
# 0.2 seconds stays far below them.
MIN_INTERVAL_S = 0.2

_last_request = 0.0


def _fetch(url: str) -> bytes:
    global _last_request
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    _last_request = time.monotonic()
    return body


def cache_path(cache_dir: Path, url: str, suffix: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    return cache_dir / f"{key}{suffix}"


def get_bytes(url: str, cache_dir: Path, suffix: str = ".bin") -> bytes:
    """Return the body for `url`, reading it from `cache_dir` if it was fetched before."""
    path = cache_path(cache_dir, url, suffix)
    if path.exists():
        return path.read_bytes()
    body = _fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    meta = {"url": url, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta))
    return body


def get_json(url: str, cache_dir: Path) -> Any:
    return json.loads(get_bytes(url, cache_dir, suffix=".json"))
