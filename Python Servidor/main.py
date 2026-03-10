import os
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types


app = FastAPI(title="Zora AI Gemini Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois troque pelo seu domínio
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Provider(str, Enum):
    groq = "groq"      # perfil Gemini 1
    gemini = "gemini"  # perfil Gemini 2
    openai = "openai"  # perfil Gemini 3
    custom = "custom"  # perfil Gemini 4 = Zora AI


class ChatRequest(BaseModel):
    message: str
    provider: Provider


class ChatResponse(BaseModel):
    reply: str
    provider: Provider


def make_client(env_name: str):
    key = os.getenv(env_name) or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


clients = {
    "groq": make_client("GEMINI_API_KEY_1"),
    "gemini": make_client("GEMINI_API_KEY_2"),
    "openai": make_client("GEMINI_API_KEY_3"),
    "custom": make_client("GEMINI_API_KEY_4"),
}


COMMON_LANGUAGE_RULE = (
    "Always reply in the same language as the user's latest message. "
    "If the user writes in Portuguese, answer in Portuguese. "
    "If the user writes in English, answer in English. "
    "If the user writes in another language, answer in that same language. "
    "Do not translate unless the user asks."
)

PROFILES = {
    "groq": {
        "model": os.getenv("GEMINI_MODEL_1", "gemini-2.5-flash"),
        "system_instruction": (
            f"{COMMON_LANGUAGE_RULE} "
            "Be fast, direct, and practical."
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_1", "0.7")),
    },
    "gemini": {
        "model": os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash"),
        "system_instruction": (
            f"{COMMON_LANGUAGE_RULE} "
            "Be balanced, clear, and helpful."
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_2", "0.8")),
    },
    "openai": {
        "model": os.getenv("GEMINI_MODEL_3", "gemini-2.5-flash"),
        "system_instruction": (
            f"{COMMON_LANGUAGE_RULE} "
            "Be smart, concise, and natural."
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_3", "0.75")),
    },
    "custom": {
        "model": os.getenv("GEMINI_MODEL_4", "gemini-2.5-flash"),
        "system_instruction": os.getenv(
            "ZORA_SYSTEM_PROMPT",
            (
                f"{COMMON_LANGUAGE_RULE} "
                "You are Zora AI. Respond clearly, elegantly, usefully, and naturally. "
                "Keep a modern tone and be genuinely helpful."
            ),
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_4", "0.85")),
    },
}


def ask_gemini_profile(user_text: str, provider: Provider) -> str:
    provider_key = provider.value
    client = clients.get(provider_key)
    profile = PROFILES[provider_key]

    if client is None:
        raise HTTPException(
            status_code=500,
            detail=f"API key do perfil {provider_key} não configurada."
        )

    response = client.models.generate_content(
        model=profile["model"],
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=profile["system_instruction"],
            temperature=profile["temperature"],
        ),
    )

    return response.text or f"Sem resposta do perfil {provider_key}."


@app.get("/")
def root():
    return {"ok": True, "name": "Zora AI Gemini Router"}


@app.get("/health")
def health():
    return {
        "gemini_api_key_1": bool(os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY")),
        "gemini_api_key_2": bool(os.getenv("GEMINI_API_KEY_2") or os.getenv("GEMINI_API_KEY")),
        "gemini_api_key_3": bool(os.getenv("GEMINI_API_KEY_3") or os.getenv("GEMINI_API_KEY")),
        "gemini_api_key_4": bool(os.getenv("GEMINI_API_KEY_4") or os.getenv("GEMINI_API_KEY")),
        "model_1": PROFILES["groq"]["model"],
        "model_2": PROFILES["gemini"]["model"],
        "model_3": PROFILES["openai"]["model"],
        "model_4": PROFILES["custom"]["model"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    text = payload.message.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    try:
        reply = ask_gemini_profile(text, payload.provider)
        return ChatResponse(reply=reply, provider=payload.provider)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no provider {payload.provider}: {str(e)}"
        )
