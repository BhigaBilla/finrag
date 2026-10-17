"""FastAPI service around the agent and the search index.

    uvicorn src.api:app --port 8000

POST /ask     {"question": ..., "approval": "never|multi|always"}
              -> {"status": "answered", "answer", "citations", ...}
              or {"status": "needs_approval", "thread_id", "plan"}
POST /resume  {"thread_id", "action": "approve|reject|edit", "search_query"?, "companies"?, "fy"?}
GET  /search  ?q=...&company=...&fy=...&k=8   (retrieval only, no LLM)
GET  /health
"""
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import COMPANIES, YEARS

STATE = {}


def load(app_state):
    """Build the retriever and the agents once (slow: loads indexes and models)."""
    from dotenv import load_dotenv
    from langgraph.checkpoint.memory import InMemorySaver

    from .agent import GeminiBrain, build_graph
    from .retrieve import Retriever

    load_dotenv()
    retriever, brain, saver = Retriever(rerank=None), GeminiBrain(), InMemorySaver()
    app_state["retriever"] = retriever
    app_state["agents"] = {p: build_graph(retriever, brain, approval=p, checkpointer=saver)
                           for p in ("never", "multi", "always")}
    app_state["threads"] = {}


@asynccontextmanager
async def lifespan(app):
    if not STATE:
        load(STATE)
    yield


app = FastAPI(title="FinRAG", lifespan=lifespan)


class Ask(BaseModel):
    question: str
    approval: Literal["never", "multi", "always"] = "never"


class Resume(BaseModel):
    thread_id: str
    action: Literal["approve", "reject", "edit"]
    search_query: Optional[str] = None
    companies: Optional[list[str]] = None
    fy: Optional[str] = None


def respond(out, thread_id):
    if "__interrupt__" in out:
        return {"status": "needs_approval", "thread_id": thread_id,
                "plan": out["__interrupt__"][0].value}
    return {"status": "answered", "thread_id": thread_id, "answer": out["answer"],
            "companies": out.get("companies", []), "fy": out.get("fy"),
            "queries": out.get("tried_queries", []),
            "citations": [{"company": p["company"], "fy": p["fy"], "page": p["page"]}
                          for p in out.get("relevant", [])]}


@app.get("/health")
def health():
    return {"ok": True, "companies": list(COMPANIES), "financial_years": YEARS}


@app.get("/search")
def search(q: str, company: Optional[str] = None, fy: Optional[str] = None, k: int = 8):
    if company and company not in COMPANIES:
        raise HTTPException(400, f"unknown company {company}")
    if fy and fy not in YEARS:
        raise HTTPException(400, f"unknown fy {fy}")
    r = STATE["retriever"]
    return [{key: r.by_id[c][key] for key in ("company", "fy", "page", "text")}
            for c in r.hybrid_search(q, k=max(1, min(k, 25)), company=company, fy=fy)]


@app.post("/ask")
def ask(req: Ask):
    thread_id = str(uuid.uuid4())
    STATE["threads"][thread_id] = req.approval
    out = STATE["agents"][req.approval].invoke({"question": req.question},
                                               {"configurable": {"thread_id": thread_id}})
    return respond(out, thread_id)


@app.post("/resume")
def resume(req: Resume):
    from langgraph.types import Command

    policy = STATE["threads"].get(req.thread_id)
    if policy is None:
        raise HTTPException(404, "unknown thread_id")
    decision = req.model_dump(exclude_none=True, exclude={"thread_id"})
    out = STATE["agents"][policy].invoke(Command(resume=decision),
                                         {"configurable": {"thread_id": req.thread_id}})
    return respond(out, req.thread_id)
