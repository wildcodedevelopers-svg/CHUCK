
import os
import sqlite3
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Models are configurable through Render environment variables.
MODEL = os.getenv("MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.6-flash")
DB_PATH = os.getenv("DB_PATH", "chuck.db")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from environment variables.")

client = genai.Client(api_key=api_key)
app = FastAPI(title="C.H.U.C.K.")

# Allows your website to connect to the API.
# No credentials are enabled for wildcard origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Keep your existing SQLite storage.
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
        # Retry temporary overloads twice before trying fallback.
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
                if reply:
                    return reply

                raise RuntimeError("Gemini returned an empty response.")

            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()

                is_temporary = any(
                    term in error_text
                    for term in [
                        "503",
                        "unavailable",
                        "overloaded",
                        "high demand",
                        "429",
                        "resource_exhausted",
                        "rate limit",
                        "timeout",
                    ]
                )

                print(
                    f"Gemini error: model={model}, "
                    f"attempt={attempt + 1}, error={exc}"
                )

                if is_temporary and attempt == 0:
                    time.sleep(2)
                    continue

                # Try the fallback model after the primary fails.
                break

    error_text = str(last_error).lower() if last_error else ""

    if any(term in error_text for term in [
        "503", "unavailable", "overloaded", "high demand",
        "429", "resource_exhausted", "rate limit", "timeout"
    ]):
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily busy. "
                "Please try again shortly."
            ),
        )

    print(f"All Gemini models failed: {last_error}")
    raise HTTPException(
        status_code=502,
        detail="C.H.U.C.K. could not get a response from Gemini."
    )


# Support both the old route and the website's route.
@app.post("/ask")
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    message = req.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    reply = ask_gemini(message)
    return {"response": reply}
