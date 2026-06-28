import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import json
import requests
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY in your .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# TABLE = "blot_results"
TABLE = "western_blot_records"

app = FastAPI(title="Western Blot Miner API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.getenv("OPENAI_KEY")

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY not found")

client = OpenAI(api_key=OPENAI_KEY)

class SearchRequest(BaseModel):
    query: str
    limit: int = 100

PARSE_PROMPT = """Extract search filters from the user's question about western
blot data. Return ONLY JSON with any of these keys that apply (omit the rest):
  "protein"   - protein/gene name
  "cell_line" - cell line or tissue
  "condition" - treatment/drug/genotype
Return {} if nothing maps. No commentary.
Question: """

SQL_PROMPT = """
You are an expert PostgreSQL query generator.

Convert the user's natural language question into a SINGLE PostgreSQL SELECT query.

Database schema:

CREATE TABLE western_blot_records (
   id BIGSERIAL PRIMARY KEY,
   paper_id TEXT NOT NULL,
   page INTEGER,
   western_blot_type TEXT,
   sample TEXT,
   organism TEXT,
   treatment_context TEXT,
   figure_label TEXT,
   target TEXT,
   condition TEXT,
   band_detected BOOLEAN,
   confidence TEXT
);

Rules:
1. ONLY generate a SELECT query. Never INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE.
2. Use only the table western_blot_records.
3. Return all columns using SELECT *.
4. For text matching use ILIKE with '%term%'.
5. A drug, treatment, or condition term may appear in EITHER treatment_context OR condition. Match against both with OR, e.g.:
   (treatment_context ILIKE '%term%' OR condition ILIKE '%term%')
6. A sample, cell line, or tissue term may appear in EITHER sample OR organism. Match against both with OR.
7. Use only the core root of a term, not the full phrase. For "Nutlin-3" use '%nutlin%'. For "HeLa cells" use '%hela%'. Drop suffixes like numbers, "cells", "treated".
8. Combine the distinct concepts (target, sample, treatment) with AND, but each concept's column-matching uses OR as above.
9. Do not explain anything. Return ONLY SQL. Never wrap in markdown.

User Question:
"""


@app.get("/")
def index():
    try:
        resp = supabase.table(TABLE).select("*").limit(5).execute()
        return {"connected": True, "sample_rows": resp.data}
    except Exception as e:
        raise HTTPException(500, f"Supabase query failed: {e}")


@app.get("/health")
def health():
    """Liveness check (use this for uptime pings / hosting health checks)."""
    return {"status": "ok"}


@app.get("/proteins")
def search_protein(
    name: str = Query(..., description="Protein name to search, e.g. p53"),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
):

    if not name.strip():
        raise HTTPException(400, "name query param cannot be empty")
    try:
        resp = (
            supabase.table(TABLE)
            .select("*")
            .ilike("target", f"%{name}%")
            .limit(limit)
            .execute()
        )
    except Exception as e:
        raise HTTPException(500, f"Supabase query failed: {e}")

    return {"protein": name, "count": len(resp.data), "results": resp.data}


def parse_query(question: str) -> dict:
    """NL question -> {protein, cell_line, condition} via the LLM."""
    resp = requests.post(
        os.environ["QWEN_URL"],
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [{"role": "user", "content": PARSE_PROMPT + question}],
            "temperature": 0,
        },
        timeout=30,
    )
    raw = resp.json()["choices"][0]["message"]["content"]
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

def generate_sql(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # or gpt-4.1 / gpt-5 if available to your account
        messages=[
            {
                "role": "system",
                "content": SQL_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0
    )

    sql = response.choices[0].message.content.strip()

    sql = (
        sql.replace("```sql", "")
           .replace("```", "")
           .strip()
    )

    return sql

def validate_sql(sql: str):

    sql = sql.strip()

    if not sql.lower().startswith("select"):
        raise HTTPException(400, "Only SELECT queries are allowed")

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate"
    ]

    lower = sql.lower()

    for keyword in forbidden:
        if keyword in lower:
            raise HTTPException(400, f"{keyword.upper()} is not allowed")

    if "western_blot_records" not in lower:
        raise HTTPException(
            400,
            "Query must use western_blot_records"
        )

    return sql


@app.post("/search")
def search(request: SearchRequest):

    sql = generate_sql(request.query)
    print("Generated SQL:", sql)
    sql = validate_sql(sql)

    result = supabase.rpc(
        "execute_sql",
        {"query": sql}
    ).execute()

    return {
        "question": request.query,
        "generated_sql": sql,
        "count": len(result.data),
        "results": result.data
    }
