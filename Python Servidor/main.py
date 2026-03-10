import os
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from google import genai
from groq import Groq


app = FastAPI(title="Zora AI Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois troca pelo seu domínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Provider(str, Enum):
    groq = "groq"
    gemini = "gemini"
    openai = "openai"
    custom = "custom"   # custom = Zora AI


class ChatRequest(BaseModel):
    message: str
    provider: Provider


class ChatResponse(BaseModel):
    reply: str
    provider: Provider


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
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
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada.")

    zora_prompt = os.getenv(
        "ZORA_SYSTEM_PROMPT",
        "Você é a Zora AI. Responda em português, de forma clara, moderna, útil e inteligente."
    )

    response = openai_client.responses.create(
        model=os.getenv("ZORA_OPENAI_MODEL", "gpt-5"),
        instructions=zora_prompt,
        input=user_text,
    )
    return response.output_text or "Sem resposta da Zora AI."


def ask_gemini(user_text: str) -> str:
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada.")

    prompt = os.getenv(
        "GEMINI_SYSTEM_PROMPT",
        "Responda em português, de forma clara e objetiva."
    )

    response = gemini_client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=f"{prompt}\n\nUsuário: {user_text}",
    )
    return response.text or "Sem resposta do Gemini."


def ask_groq(user_text: str) -> str:
    if groq_client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada.")

    prompt = os.getenv(
        "GROQ_SYSTEM_PROMPT",
        "Responda em português, de forma rápida e direta."
    )

    response = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text}
        ],
    )
    content = response.choices[0].message.content
    return content or "Sem resposta da Groq."


@app.get("/")
def root():
    return {"ok": True, "name": "Zora AI Router"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    text = payload.message.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    try:
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))