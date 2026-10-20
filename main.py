
import os
import sqlite3
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types

MODEL = os.getenv("MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.6-flash")
DB_PATH = os.getenv("DB_PATH", "chuck.db")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from environment variables.")

client = genai.Client(api_key=api_key)
app = FastAPI(title="C.H.U.C.K.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute(
    "CREATE TABLE IF NOT EXISTS history "
    "(id INTEGER PRIMARY KEY, role TEXT, content TEXT, ts REAL)"
)
db.execute(
    "CREATE TABLE IF NOT EXISTS memory "
    "(key TEXT PRIMARY KEY, value TEXT)"
)
db.commit()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "C.H.U.C.K. backend is running."
    }


def ask_gemini(message: str) -> str:
    models = list(dict.fromkeys([MODEL, FALLBACK_MODEL]))
    last_error = None

    for model in models:
        for attempt in range(2):
            try:
                result = client.models.generate_content(
                    model=model,
                    contents=message,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="low"
                        )
                    ),
                )

                reply = result.text
                if reply and reply.strip():
                    return reply.strip()

                raise RuntimeError("Gemini returned an empty response.")

            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()

                is_temporary = any(
                    term in error_text
                    for term in [
                        "503", "unavailable", "overloaded",
                        "high demand", "429", "resource_exhausted",
                        "rate limit", "timeout"
                    ]
                )

                print(
                    f"Gemini error: model={model}, "
                    f"attempt={attempt + 1}, error={exc}"
                )

                if is_temporary and attempt == 0:
                    time.sleep(2)
                    continue

                break

    error_text = str(last_error).lower() if last_error else ""

    if any(term in error_text for term in [
        "503", "unavailable", "overloaded", "high demand",
        "429", "resource_exhausted", "rate limit", "timeout"
    ]):
        raise HTTPException(
            status_code=503,
            detail="Gemini is temporarily busy. Please try again shortly."
        )

    raise HTTPException(
        status_code=502,
        detail="C.H.U.C.K. could not get a response from Gemini."
    )


async def process_chat(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail='Invalid JSON. Send {"message": "Hello"}'
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail='Request body must be a JSON object containing "message".'
        )

    message = payload.get("message")

    if not isinstance(message, str):
        raise HTTPException(
            status_code=422,
            detail='Missing or invalid "message". Send {"message": "Hello"}.'
        )

    message = message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    reply = ask_gemini(message)
    return {"response": reply}


@app.post("/ask")
async def ask_endpoint(request: Request):
    return await process_chat(request)


@app.post("/chat")
async def chat_endpoint(request: Request):
    return await process_chat(request)
