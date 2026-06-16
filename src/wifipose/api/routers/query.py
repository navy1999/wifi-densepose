"""Natural-language analytics query endpoint (text-to-SQL over pose_events)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from wifipose.api.deps import get_llm, get_repository
from wifipose.api.metrics import LLM_QUERIES
from wifipose.llm.nl_to_sql import SCHEMA_DESCRIPTION, generate_sql

router = APIRouter(prefix="/api/v1", tags=["llm-query"])


class QueryRequest(BaseModel):
    question: str
    execute: bool = True


@router.get("/query/schema")
async def query_schema() -> dict:
    """Expose the queryable schema (handy for the frontend + transparency)."""
    return {"schema": SCHEMA_DESCRIPTION}


@router.post("/query")
async def natural_language_query(
    req: QueryRequest, repo=Depends(get_repository), llm=Depends(get_llm)
) -> dict:
    """Convert a question to a safe SELECT, optionally execute it, return rows."""
    gen = await generate_sql(req.question, llm)
    LLM_QUERIES.labels(provider=gen.provider, valid=str(gen.valid).lower()).inc()

    response: dict = {
        "question": gen.question,
        "sql": gen.sql,
        "provider": gen.provider,
        "valid": gen.valid,
    }
    if not gen.valid:
        response["error"] = gen.error
        return response

    if req.execute:
        try:
            response["rows"] = await repo.run_safe_select(gen.sql)
            response["row_count"] = len(response["rows"])
        except Exception as e:  # noqa: BLE001
            response["error"] = f"execution failed: {e}"
    return response
