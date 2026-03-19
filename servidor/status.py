import os
import time
from threading import Lock
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()

status_lock = Lock()

runtime_state = {
    "in_flight": 0,
    "current_request_started_at": 0.0,
    "last_success_at": 0.0,
    "last_response_time": 0.0,
    "last_error_at": 0.0,
    "last_error_kind": None,   # transient | severe | config
    "last_error_message": None,
}


class StatusResponse(BaseModel):
    status: str
    detail: Optional[str] = None


def has_any_provider_configured() -> bool:
    return any(
        [
            bool(os.getenv("GROQ_API_KEY_1")),
            bool(os.getenv("GROQ_API_KEY_2")),
            bool(os.getenv("GEMINI_API_KEY_1")),
            bool(os.getenv("GEMINI_API_KEY_2")),
        ]
    )


def begin_request() -> None:
    with status_lock:
        runtime_state["in_flight"] += 1
        if runtime_state["current_request_started_at"] <= 0:
            runtime_state["current_request_started_at"] = time.time()


def end_request() -> None:
    with status_lock:
        runtime_state["in_flight"] = max(0, runtime_state["in_flight"] - 1)
        if runtime_state["in_flight"] == 0:
            runtime_state["current_request_started_at"] = 0.0


def mark_success(elapsed_seconds: float) -> None:
    with status_lock:
        runtime_state["last_success_at"] = time.time()
        runtime_state["last_response_time"] = elapsed_seconds
        runtime_state["last_error_at"] = 0.0
        runtime_state["last_error_kind"] = None
        runtime_state["last_error_message"] = None


def mark_error(kind: str, message: str) -> None:
    with status_lock:
        runtime_state["last_error_at"] = time.time()
        runtime_state["last_error_kind"] = kind
        runtime_state["last_error_message"] = str(message)[:300]


def classify_exception(exc: Exception) -> tuple[str, int, str]:
    message = str(exc).lower()
    status_code = getattr(exc, "status_code", None)

    if status_code == 429 or any(
        x in message for x in [
            "rate limit",
            "quota",
            "resource exhausted",
            "too many requests",
        ]
    ):
        return ("severe", 429, "Limite da API atingido.")

    if status_code in (401, 403) or any(
        x in message for x in [
            "invalid api key",
            "api key not valid",
            "permission denied",
            "unauthorized",
            "forbidden",
        ]
    ):
        return ("severe", 503, "Erro de autenticação na API.")

    if status_code in (500, 502, 503, 504) or any(
        x in message for x in [
            "timeout",
            "timed out",
            "deadline exceeded",
            "connection",
            "temporarily unavailable",
            "service unavailable",
            "internal server error",
            "bad gateway",
            "gateway",
        ]
    ):
        return ("transient", 503, "Provider indisponível no momento.")

    return ("transient", 500, f"Erro interno: {str(exc)}")


def get_system_status() -> tuple[str, Optional[str]]:
    if not has_any_provider_configured():
        return ("offline", "Nenhuma API configurada.")

    now = time.time()

    with status_lock:
        in_flight = runtime_state["in_flight"]
        current_request_started_at = runtime_state["current_request_started_at"]
        last_error_at = runtime_state["last_error_at"]
        last_error_kind = runtime_state["last_error_kind"]
        last_error_message = runtime_state["last_error_message"]
        last_response_time = runtime_state["last_response_time"]

    # Online / Ausente durante processamento
    if in_flight > 0 and current_request_started_at > 0:
        running_for = now - current_request_started_at
        if running_for >= 2.5:
            return ("away", "Processando lentamente.")
        return ("online", "Processando.")

    # Último erro recente
    if last_error_at > 0 and (now - last_error_at) <= 90:
        if last_error_kind == "severe":
            return ("offline", last_error_message or "Falha grave no provider.")
        return ("away", last_error_message or "Instabilidade temporária.")

    # Resposta lenta recente
    if last_response_time >= 10:
        return ("away", "Resposta lenta recentemente.")

    return ("online", "Sistema operacional.")


@router.api_route("/status", methods=["GET", "HEAD"], response_model=StatusResponse)
@router.api_route("/api/status", methods=["GET", "HEAD"], response_model=StatusResponse)
def status():
    current_status, detail = get_system_status()
    return StatusResponse(status=current_status, detail=detail)


@router.api_route("/health", methods=["GET", "HEAD"])
@router.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    return {
        "ok": True,
        "groq_1_key": bool(os.getenv("GROQ_API_KEY_1")),
        "groq_2_key": bool(os.getenv("GROQ_API_KEY_2")),
        "gemini_1_key": bool(os.getenv("GEMINI_API_KEY_1")),
        "gemini_2_key": bool(os.getenv("GEMINI_API_KEY_2")),
        "groq_1_model": os.getenv("GROQ_MODEL_1", "llama-3.3-70b-versatile"),
        "groq_2_model": os.getenv("GROQ_MODEL_2", "llama-3.3-70b-versatile"),
        "gemini_1_model": os.getenv("GEMINI_MODEL_1", "gemini-2.5-flash"),
        "gemini_2_model": os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash"),
        "system_status": get_system_status()[0],
    }