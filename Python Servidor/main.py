import os
import logging
import traceback
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from google import genai
from groq import Groq

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Zora AI Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Provider(str, Enum):
    groq = "groq"
    gemini = "gemini"
    openai = "openai"
    custom = "custom"  # custom = Zora AI


class ChatRequest(BaseModel):
    message: str
    provider: Provider


class ChatResponse(BaseModel):
    reply: str
    provider: Provider


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
zora_openai_client = OpenAI(api_key=os.getenv("ZORA_OPENAI_API_KEY")) if os.getenv("ZORA_OPENAI_API_KEY") else None
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None


def ask_openai_normal(user_text: str) -> str:
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada.")

    response = openai_client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        input=user_text,
    )
    return response.output_text or "Sem resposta da OpenAI."


def ask_zora_ai(user_text: str) -> str:
    if zora_openai_client is None:
        raise HTTPException(status_code=500, detail="ZORA_OPENAI_API_KEY não configurada.")

    response = zora_openai_client.responses.create(
        model=os.getenv("ZORA_OPENAI_MODEL", "gpt-5-mini"),
        instructions=os.getenv(
            "ZORA_SYSTEM_PROMPT",
            "Você é a Zora AI. Responda em português, de forma clara e útil."
        ),
        input=user_text,
    )
    return response.output_text or "Sem resposta da Zora AI."


def ask_gemini(user_text: str) -> str:
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada.")

    response = gemini_client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=user_text,
    )
    return response.text or "Sem resposta do Gemini."


def ask_groq(user_text: str) -> str:
    if groq_client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada.")

    response = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "user", "content": user_text}
        ],
    )
    return response.choices[0].message.content or "Sem resposta da Groq."


@app.get("/")
def root():
    return {"ok": True, "name": "Zora AI Router"}


@app.get("/health")
def health():
    return {
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "zora_openai_key": bool(os.getenv("ZORA_OPENAI_API_KEY")),
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "groq_key": bool(os.getenv("GROQ_API_KEY")),
        "openai_model": os.getenv("OPENAI_MODEL"),
        "zora_openai_model": os.getenv("ZORA_OPENAI_MODEL"),
        "gemini_model": os.getenv("GEMINI_MODEL"),
        "groq_model": os.getenv("GROQ_MODEL"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    text = payload.message.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    try:
        logger.info(f"Provider recebido: {payload.provider}")

        if payload.provider == Provider.openai:
            reply = ask_openai_normal(text)
        elif payload.provider == Provider.gemini:
            reply = ask_gemini(text)
        elif payload.provider == Provider.groq:
            reply = ask_groq(text)
        elif payload.provider == Provider.custom:
            reply = ask_zora_ai(text)
        else:
            raise HTTPException(status_code=400, detail="Provider inválido.")

        return ChatResponse(reply=reply, provider=payload.provider)

    except HTTPException as e:
        logger.error(f"HTTPException no provider {payload.provider}: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Erro no provider {payload.provider}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erro no provider {payload.provider}: {str(e)}"
        )
