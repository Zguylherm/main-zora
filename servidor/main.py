import os
import time
import logging
import traceback
from enum import Enum
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from groq import Groq


logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Zora AI Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois troque pelo seu domínio
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Provider(str, Enum):
    groq = "groq"
    gemini = "gemini"
    openai = "openai"
    custom = "custom"


class ChatRequest(BaseModel):
    message: str
    provider: Provider


class ChatResponse(BaseModel):
    reply: str
    provider: Provider


class StatusResponse(BaseModel):
    status: str
    detail: Optional[str] = None


def same_language_rule() -> str:
    return (
        "Always reply in the same language as the user's latest message. "
        "If the user writes in Portuguese, answer in Portuguese. "
        "If the user writes in English, answer in English. "
        "If the user writes in another language, answer in that same language. "
        "Do not translate unless the user asks."
    )


GROQ_PROFILES = {
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY_1", ""),
        "model": os.getenv("GROQ_MODEL_1", "llama-3.3-70b-versatile"),
        "system": os.getenv(
            "GROQ_SYSTEM_PROMPT_1",
            f"{same_language_rule()} Be fast, direct, and practical."
        ),
    },
    "openai": {
        "api_key": os.getenv("GROQ_API_KEY_2", ""),
        "model": os.getenv("GROQ_MODEL_2", "llama-3.3-70b-versatile"),
        "system": os.getenv(
            "GROQ_SYSTEM_PROMPT_2",
            f"{same_language_rule()} Be smart, concise, and clear."
        ),
    },
}

GEMINI_PROFILES = {
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY_1", ""),
        "model": os.getenv("GEMINI_MODEL_1", "gemini-2.5-flash"),
        "system": os.getenv(
            "GEMINI_SYSTEM_PROMPT_1",
            f"{same_language_rule()} Be balanced, clear, and helpful."
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_1", "0.8")),
    },
    "custom": {
        "api_key": os.getenv("GEMINI_API_KEY_2", ""),
        "model": os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash"),
        "system": os.getenv(
            "ZORA_SYSTEM_PROMPT",
            (
                f"{same_language_rule()} "
                "You are Zora AI. Be modern, elegant, useful, natural, and helpful."
            ),
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_2", "0.85")),
    },
}


status_lock = Lock()

runtime_state = {
    "in_flight": 0,
    "current_request_started_at": 0.0,
    "last_success_at": 0.0,
    "last_response_time": 0.0,
    "last_error_at": 0.0,
    "last_error_kind": None,
    "last_error_message": None,
}


def has_any_provider_configured() -> bool:
    return any(
        [
            bool(GROQ_PROFILES["groq"]["api_key"]),
            bool(GROQ_PROFILES["openai"]["api_key"]),
            bool(GEMINI_PROFILES["gemini"]["api_key"]),
            bool(GEMINI_PROFILES["custom"]["api_key"]),
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
        runtime_state["last_error_message"] = message[:300]


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

    if in_flight > 0 and current_request_started_at > 0:
        running_for = now - current_request_started_at
        if running_for >= 2.5:
            return ("away", "Processando lentamente.")
        return ("online", "Processando.")

    if last_error_at > 0 and (now - last_error_at) <= 90:
        if last_error_kind == "severe":
            return ("offline", last_error_message or "Falha grave no provider.")
        return ("away", last_error_message or "Instabilidade temporária.")

    if last_response_time >= 10:
        return ("away", "Resposta lenta recentemente.")

    return ("online", "Sistema operacional.")


def ask_groq_profile(user_text: str, provider_key: str) -> str:
    profile = GROQ_PROFILES[provider_key]

    if not profile["api_key"]:
        raise HTTPException(
            status_code=500,
            detail=f"GROQ API key do perfil {provider_key} não configurada."
        )

    client = Groq(api_key=profile["api_key"])

    response = client.chat.completions.create(
        model=profile["model"],
        messages=[
            {"role": "system", "content": profile["system"]},
            {"role": "user", "content": user_text},
        ],
    )

    content = response.choices[0].message.content
    return content or f"Sem resposta do perfil {provider_key}."


def ask_gemini_profile(user_text: str, provider_key: str) -> str:
    profile = GEMINI_PROFILES[provider_key]

    if not profile["api_key"]:
        raise HTTPException(
            status_code=500,
            detail=f"GEMINI API key do perfil {provider_key} não configurada."
        )

    client = genai.Client(api_key=profile["api_key"])

    response = client.models.generate_content(
        model=profile["model"],
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=profile["system"],
            temperature=profile["temperature"],
        ),
    )

    return response.text or f"Sem resposta do perfil {provider_key}."


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"ok": True, "name": "Zora AI Router"}


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
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


@app.api_route("/status", methods=["GET", "HEAD"], response_model=StatusResponse)
@app.api_route("/api/status", methods=["GET", "HEAD"], response_model=StatusResponse)
def status():
    current_status, detail = get_system_status()
    return StatusResponse(status=current_status, detail=detail)


@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    text = payload.message.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    begin_request()
    started_at = time.time()

    try:
        logger.info(f"Provider recebido: {payload.provider}")

        if payload.provider == Provider.groq:
            reply = ask_groq_profile(text, "groq")
        elif payload.provider == Provider.gemini:
            reply = ask_gemini_profile(text, "gemini")
        elif payload.provider == Provider.openai:
            reply = ask_groq_profile(text, "openai")
        elif payload.provider == Provider.custom:
            reply = ask_gemini_profile(text, "custom")
        else:
            raise HTTPException(status_code=400, detail="Provider inválido.")

        elapsed = time.time() - started_at
        mark_success(elapsed)

        logger.info(f"Resposta OK para provider {payload.provider} em {elapsed:.2f}s")
        return ChatResponse(reply=reply, provider=payload.provider)

    except HTTPException as e:
        mark_error("config" if e.status_code < 500 else "transient", str(e.detail))
        logger.error(f"HTTPException no provider {payload.provider}: {e.detail}")
        raise

    except Exception as e:
        kind, status_code, public_message = classify_exception(e)
        mark_error(kind, public_message)

        logger.error(f"Erro no provider {payload.provider}: {str(e)}")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=status_code,
            detail=public_message
        )

    finally:
        end_request()
