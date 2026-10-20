import os, sqlite3, time
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

MODEL = os.getenv("MODEL", "gemini-2.5-flash")
TOKEN = os.getenv("CHUCK_TOKEN", "")
DB_PATH = os.getenv("DB_PATH", "chuck.db")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = FastAPI(title="Chuck")

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("create table if not exists history(id integer primary key, role text, content text, ts real)")
db.execute("create table if not exists memory(key text primary key, value text)")
db.commit()

TOOLS = types.Tool(function_declarations=[
    {
        "name": "remember",
        "description": "Save a lasting fact about the user (name, preferences, people, routines).",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a saved memory by key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "open_url",
        "description": "Open an app or website on the user's phone using a URL or deep link, "
        "e.g. https://youtube.com, spotify:, tel:+15551234567, "
