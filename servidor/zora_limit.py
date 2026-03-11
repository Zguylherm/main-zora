from __future__ import annotations

from threading import Lock
from time import time
from typing import Any

from fastapi import HTTPException, Request, status

MAX_REQUESTS = 10
WINDOW_SECONDS = 2 * 60 * 60  # 2 horas

_lock = Lock()

_ip_store: dict[str, dict[str, float]] = {}
_fp_store: dict[str, dict[str, float]] = {}
_browser_store: dict[str, dict[str, float]] = {}


def _normalize(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _clean_expired(store: dict[str, dict[str, float]], now: float) -> None:
    expired_keys = [key for key, item in store.items() if now >= item["reset_at"]]
    for key in expired_keys:
        del store[key]


def _prepare_bucket(
    store: dict[str, dict[str, float]],
    key: str,
    now: float,
) -> dict[str, float]:
    item = store.get(key)

    if item is None or now >= item["reset_at"]:
        item = {
            "count": 0,
            "reset_at": now + WINDOW_SECONDS,
        }
        store[key] = item

    return item


def _remaining(item: dict[str, float]) -> int:
    return max(0, MAX_REQUESTS - int(item["count"]))


def _retry_after(item: dict[str, float], now: float) -> int:
    return max(1, int(item["reset_at"] - now))


def get_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def check_chat_limit(
    request: Request,
    fingerprint: str | None = None,
    browser_id: str | None = None,
) -> dict[str, Any]:
    now = time()

    ip = _normalize(get_client_ip(request))
    fp = _normalize(fingerprint)
    bid = _normalize(browser_id)

    with _lock:
        _clean_expired(_ip_store, now)
        _clean_expired(_fp_store, now)
        _clean_expired(_browser_store, now)

        targets: list[tuple[str, dict[str, dict[str, float]], str]] = []

        if ip:
            targets.append(("ip", _ip_store, ip))
        if fp:
            targets.append(("fingerprint", _fp_store, fp))
        if bid:
            targets.append(("browser_id", _browser_store, bid))

        if not targets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Não foi possível identificar o cliente.",
            )

        prepared: list[tuple[str, dict[str, float]]] = []
        blocked_by: list[str] = []

        for name, store, key in targets:
            item = _prepare_bucket(store, key, now)
            prepared.append((name, item))

            if item["count"] >= MAX_REQUESTS:
                blocked_by.append(name)

        if blocked_by:
            retry_after = max(_retry_after(item, now) for _, item in prepared)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Limite atingido. Tente novamente em {retry_after} segundos.",
            )

        for _, item in prepared:
            item["count"] += 1

        remaining = min(_remaining(item) for _, item in prepared)
        reset_in = max(_retry_after(item, now) for _, item in prepared)

        return {
            "remaining": remaining,
            "reset_in": reset_in,
            "max_requests": MAX_REQUESTS,
            "window_seconds": WINDOW_SECONDS,
            "ip": ip,
            "has_fingerprint": fp is not None,
            "has_browser_id": bid is not None,
        }