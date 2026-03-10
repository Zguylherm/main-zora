import os
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from groq import Groq


app = FastAPI(title="Zora AI Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://profound-rabanadas-f9e503.netlify.app/"],  # depois troque pelo seu domínio
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Provider(str, Enum):
    groq = "groq"       # Groq perfil 1
    gemini = "gemini"   # Gemini perfil 1
    openai = "openai"   # Groq perfil 2
    custom = "custom"   # Gemini perfil 2 = Zora AI


class ChatRequest(BaseModel):
    message: str
    provider: Provider


class ChatResponse(BaseModel):
    reply: str
    provider: Provider


def _lang_rule() -> str:
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
            f"{_lang_rule()} Be fast, direct, and practical."
        ),
    },
    "openai": {
        "api_key": os.getenv("GROQ_API_KEY_2", ""),
        "model": os.getenv("GROQ_MODEL_2", "llama-3.3-70b-versatile"),
        "system": os.getenv(
            "GROQ_SYSTEM_PROMPT_2",
            f"{_lang_rule()} Be smart, concise, and clear."
        ),
    },
}

GEMINI_PROFILES = {
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY_1", ""),
        "model": os.getenv("GEMINI_MODEL_1", "gemini-2.5-flash"),
        "system": os.getenv(
            "GEMINI_SYSTEM_PROMPT_1",
            f"{_lang_rule()} Be balanced, clear, and helpful."
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_1", "0.8")),
    },
    "custom": {
        "api_key": os.getenv("GEMINI_API_KEY_2", ""),
        "model": os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash"),
        "system": os.getenv(
            "ZORA_SYSTEM_PROMPT",
            f"{_lang_rule()} You are Zora AI. Be modern, elegant, useful, and natural."
        ),
        "temperature": float(os.getenv("GEMINI_TEMP_2", "0.85")),
    },
}


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


@app.get("/")
def root():
    return {"ok": True, "name": "Zora AI Router"}


@app.get("/health")
def health():
    return {
        "groq_1_key": bool(os.getenv("GROQ_API_KEY_1")),
        "groq_2_key": bool(os.getenv("GROQ_API_KEY_2")),
        "gemini_1_key": bool(os.getenv("GEMINI_API_KEY_1")),
        "gemini_2_key": bool(os.getenv("GEMINI_API_KEY_2")),
        "groq_1_model": os.getenv("GROQ_MODEL_1", "llama-3.3-70b-versatile"),
        "groq_2_model": os.getenv("GROQ_MODEL_2", "llama-3.3-70b-versatile"),
        "gemini_1_model": os.getenv("GEMINI_MODEL_1", "gemini-2.5-flash"),
        "gemini_2_model": os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    text = payload.message.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    try:
        if payload.provider == Provider.groq:
            reply = ask_groq_profile(text, "groq")
        elif payload.provider == Provider.openai:
            reply = ask_groq_profile(text, "openai")
        elif payload.provider == Provider.gemini:
            reply = ask_gemini_profile(text, "gemini")
        elif payload.provider == Provider.custom:
            reply = ask_gemini_profile(text, "custom")
        else:
            raise HTTPException(status_code=400, detail="Provider inválido.")

        return ChatResponse(reply=reply, provider=payload.provider)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no provider {payload.provider}: {str(e)}"
        )


