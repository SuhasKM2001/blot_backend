import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import json
import requests
from pydantic import BaseModel

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY in your .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE = "blot_results"

app = FastAPI(title="Western Blot Miner API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            .ilike("protein_target_name", f"%{name}%")
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

@app.post("/search")
def natural_search(req: SearchRequest):
    """Natural-language search: parse to filters, then query safely."""
    if not req.query.strip():
        raise HTTPException(400, "query cannot be empty")

    filters = parse_query(req.query)
    if not filters:
        raise HTTPException(422, "Could not extract any filters from the query")

    query = supabase.table(TABLE).select("*")
    if filters.get("protein"):
        query = query.ilike("protein_target_name", f"%{filters['protein']}%")
    if filters.get("cell_line"):
        query = query.ilike("cell_line_tissue", f"%{filters['cell_line']}%")
    if filters.get("condition"):
        query = query.ilike("treatment_condition", f"%{filters['condition']}%")

    resp = query.limit(req.limit).execute()
    return {"query": req.query, "parsed_filters": filters,
            "count": len(resp.data), "results": resp.data}