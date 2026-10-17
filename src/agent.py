"""LangGraph agent: route -> [human approval] -> retrieve -> answer -> (rewrite -> retrieve -> answer)?

Built to run on Gemini's free tier (about 20 requests per model per day), so
LLM calls are spent only where they matter:
- route (no LLM): company and financial year come from name/alias matching.
  Step 3 showed this metadata filter matters more than any retriever choice.
  A question naming none of the 10 companies is answered with a fixed
  out-of-scope message, at zero cost.
- answer (1 LLM call): one structured call both picks the passages that
  contain the asked fact and writes the cited answer, or reports "not found".
- rewrite (1 more call, at most once): if nothing relevant came back, the
  LLM rewrites the search query and the agent tries again.
A typical question costs 1 call; the worst case is 3.

    python -m src.agent "What was TCS's attrition rate in FY2026?"
"""
import argparse
import os
import re
from typing import Optional, TypedDict

from pydantic import BaseModel, Field

from .config import COMPANIES, YEARS

MAX_REWRITES = 1
K_PER_COMPANY = 10
NOT_FOUND = "I couldn't find this in the annual reports I have (FY2025 and FY2026)."
OUT_OF_SCOPE = ("I only answer questions about the FY2025 and FY2026 annual reports of these "
                "companies: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, ITC, L&T, Bharti Airtel, "
                "Maruti Suzuki and Sun Pharma. Please name the company in your question.")

ALIASES = {
    "RELIANCE": [r"reliance", r"\bril\b", r"\bjio\b"],
    "TCS": [r"\btcs\b", r"tata consultancy"],
    "INFY": [r"infosys", r"\binfy\b"],
    "HDFCBANK": [r"\bhdfc\b"],
    "ICICIBANK": [r"\bicici\b"],
    "ITC": [r"\bitc\b"],
    "LT": [r"\bl\s?&\s?t\b", r"larsen", r"\blt\b"],
    "BHARTIARTL": [r"airtel", r"\bbharti\b"],
    "MARUTI": [r"maruti", r"suzuki"],
    "SUNPHARMA": [r"sun pharma"],
}
YEAR_PATTERNS = {
    "FY2026": [r"fy\s?'?(20)?26\b", r"2025\s?[-–/]\s?(20)?26\b", r"march\s+(31,?\s+)?2026"],
    "FY2025": [r"fy\s?'?(20)?25\b", r"2024\s?[-–/]\s?(20)?25\b", r"march\s+(31,?\s+)?2025"],
}


def route(question):
    """Companies and financial year named in the question (no LLM)."""
    q = question.lower()
    companies = [c for c, pats in ALIASES.items() if any(re.search(p, q) for p in pats)]
    years = [fy for fy, pats in YEAR_PATTERNS.items() if any(re.search(p, q) for p in pats)]
    return companies, (years[0] if len(years) == 1 else None)


class Answer(BaseModel):
    relevant: list[int] = Field(description="Numbers of the passages that contain the asked facts")
    found: bool = Field(description="True only if the passages actually answer the question")
    answer: str = Field(description="The answer with a citation tag after every fact, or empty if not found")


class Rewrite(BaseModel):
    search_query: str


class State(TypedDict, total=False):
    question: str
    companies: list
    fy: Optional[str]
    search_query: str
    tried_queries: list
    passages: list
    relevant: list
    found: bool
    rewrites: int
    answer: str


class GeminiBrain:
    """The two LLM calls. Any LangChain chat model with structured output works."""

    def __init__(self, model=None, rpm=None):
        from langchain.chat_models import init_chat_model
        from langchain_core.rate_limiters import InMemoryRateLimiter

        rpm = float(rpm or os.getenv("LLM_RPM", "8"))  # stay under the free-tier per-minute cap
        limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=1)
        self.llm = init_chat_model(model or os.getenv("LLM_MODEL", "google_genai:gemini-3.8-flash"),
                                   temperature=0, rate_limiter=limiter)

    def answer(self, question, passages):
        listing = "\n\n".join(f"[{i}] [{p['company']} {p['fy']} p.{p['page']}] {p['text']}"
                              for i, p in enumerate(passages))
        return self.llm.with_structured_output(Answer).invoke(
            "You answer questions from Indian company annual reports using ONLY the numbered "
            "passages below. First decide which passages contain the asked fact (a passage on "
            "the general topic without the fact does not count). If they answer the question, "
            "write a concise answer and put the passage's tag after every fact, e.g. "
            "[TCS FY2026 p.56]. If standalone and consolidated figures differ, give both. If the "
            "passages don't answer it, set found=false and leave answer empty. Never guess.\n\n"
            f"Question: {question}\n\nPassages:\n{listing}")

    def rewrite(self, question, tried):
        return self.llm.with_structured_output(Rewrite).invoke(
            "A search over annual reports found nothing relevant. Write a different search "
            "query using the terms an annual report would use (table headings, standard "
            "financial terms, synonyms). Plain keywords only: no quotes, OR/AND or other "
            f"search operators.\nQuestion: {question}\nAlready tried: {tried}").search_query


def needs_approval(state, policy):
    """Human-approval policy. 'multi': pause when the search fans out over
    several companies, where a wrong plan wastes the most."""
    if policy == "always":
        return True
    if policy == "multi":
        return len(state["companies"]) > 1
    return False


def build_graph(retriever, brain, approval="never", checkpointer=None):
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    def understand(s):
        companies, fy = route(s["question"])
        if not companies:
            return {"answer": OUT_OF_SCOPE, "companies": [], "fy": fy, "tried_queries": [],
                    "relevant": [], "found": False, "rewrites": 0}
        return {"companies": companies, "fy": fy, "search_query": s["question"],
                "tried_queries": [], "rewrites": 0}

    def approve(s):
        """Pause for a human. Resume with {"action": "approve"}, {"action": "reject"},
        or {"action": "edit", "companies": [...], "fy": ..., "search_query": ...}."""
        if not needs_approval(s, approval):
            return {}
        decision = interrupt({"companies": s["companies"], "fy": s["fy"],
                              "search_query": s["search_query"]})
        action = decision.get("action")
        if action == "reject":
            return {"answer": "Stopped: the search plan was rejected.", "relevant": [], "found": False}
        if action == "edit":
            return {k: decision[k] for k in ("companies", "fy", "search_query") if k in decision}
        return {}

    def retrieve(s):
        targets = s["companies"]
        k = K_PER_COMPANY if len(targets) == 1 else max(4, K_PER_COMPANY // len(targets) + 1)
        passages = []
        for company in targets:
            for cid in retriever.hybrid_search(s["search_query"], k=k, company=company, fy=s["fy"]):
                c = retriever.by_id[cid]
                passages.append({"id": cid, "company": c["company"], "fy": c["fy"],
                                 "page": c["page"], "text": c["text"]})
        return {"passages": passages, "tried_queries": s["tried_queries"] + [s["search_query"]]}

    def answer(s):
        res = brain.answer(s["question"], s["passages"]) if s["passages"] else None
        if res is None or not res.found:
            return {"found": False, "relevant": [], "answer": NOT_FOUND}
        keep = set(res.relevant)
        return {"found": True, "answer": res.answer,
                "relevant": [p for i, p in enumerate(s["passages"]) if i in keep]}

    def rewrite(s):
        return {"search_query": brain.rewrite(s["question"], s["tried_queries"]),
                "rewrites": s["rewrites"] + 1}

    def after_answer(s):
        return "end" if s["found"] or s["rewrites"] >= MAX_REWRITES else "rewrite"

    def stop_if_answered(next_node):
        return lambda s: "end" if s.get("answer") else next_node

    g = StateGraph(State)
    for name, fn in [("understand", understand), ("approve", approve), ("retrieve", retrieve),
                     ("answer", answer), ("rewrite", rewrite)]:
        g.add_node(name, fn)
    g.set_entry_point("understand")
    g.add_conditional_edges("understand", stop_if_answered("approve"), {"end": END, "approve": "approve"})
    g.add_conditional_edges("approve", stop_if_answered("retrieve"), {"end": END, "retrieve": "retrieve"})
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges("answer", after_answer, {"end": END, "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    return g.compile(checkpointer=checkpointer)


def make_agent(approval="never"):
    from dotenv import load_dotenv
    from langgraph.checkpoint.memory import InMemorySaver

    from .retrieve import Retriever

    load_dotenv()
    return build_graph(Retriever(rerank=None), GeminiBrain(), approval=approval,
                       checkpointer=InMemorySaver())


def ask_human(plan):
    print(f"\nPlan: companies={plan['companies']} fy={plan['fy']} query={plan['search_query']!r}")
    choice = input("Approve [y], reject [n], or type a new search query: ").strip().lstrip("\ufeff")  # PowerShell pipes add a BOM
    if choice.lower() in ("", "y", "yes"):
        return {"action": "approve"}
    if choice.lower() in ("n", "no"):
        return {"action": "reject"}
    return {"action": "edit", "search_query": choice}


def main():
    import uuid

    from langgraph.types import Command

    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--approval", choices=["never", "multi", "always"], default="multi")
    args = ap.parse_args()
    agent = make_agent(args.approval)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    out = agent.invoke({"question": args.question}, config)
    while "__interrupt__" in out:
        out = agent.invoke(Command(resume=ask_human(out["__interrupt__"][0].value)), config)
    print(out["answer"])
    print(f"\n-- companies={out['companies']} fy={out['fy']} queries={out['tried_queries']} "
          f"relevant={[p['id'] for p in out['relevant']]}")


if __name__ == "__main__":
    main()
