import os, sqlite3, time
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
TOKEN = os.getenv("CHUCK_TOKEN", "")
DB_PATH = os.getenv("DB_PATH", "chuck.db")

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
app = FastAPI(title="Chuck")

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("create table if not exists history(id integer primary key, role text, content text, ts real)")
db.execute("create table if not exists memory(key text primary key, value text)")
db.commit()

TOOLS = [
    {
        "name": "remember",
        "description": "Save a lasting fact about the user (name, preferences, people, routines).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a saved memory by key.",
        "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    },
    {
        "name": "open_url",
        "description": "Open an app or website on the user's phone using a URL or deep link, "
        "e.g. https://youtube.com, spotify:, whatsapp://send, tel:+15551234567, "
        "intent://#Intent;package=com.whatsapp;end",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "label": {"type": "string"}},
            "required": ["url"],
        },
    },
]


def memories():
