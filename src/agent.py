"""LangGraph agent: understand -> retrieve -> grade -> (rewrite -> retrieve)* -> answer.

The graph is deliberately small. The LLM pulls the company and financial
year out of the question (step 3 showed that metadata filtering matters more
than any retriever choice), grades which retrieved chunks actually help, and
rewrites the search query when nothing relevant came back. Answers must cite
pages as [COMPANY FY p.N] and say so when the filings don't contain the
answer.

    python -m src.agent "What was TCS's attrition rate in FY2026?"
"""
import argparse
import os
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from .config import COMPANIES, YEARS

MAX_REWRITES = 2
K_PER_COMPANY = 10
Company = Literal[tuple(COMPANIES)]
FY = Literal[tuple(YEARS)]

COMPANY_HINTS = ("RELIANCE (Reliance Industries, Jio, Reliance Retail), TCS (Tata Consultancy "
                 "Services), INFY (Infosys), HDFCBANK, ICICIBANK, ITC, LT (Larsen & Toubro), "
                 "BHARTIARTL (Bharti Airtel), MARUTI (Maruti Suzuki), SUNPHARMA (Sun Pharma)")


OUT_OF_SCOPE = ("I only answer questions about the FY2025 and FY2026 annual reports of these "
                "companies: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, ITC, L&T, Bharti Airtel, "
                "Maruti Suzuki and Sun Pharma.")


class Plan(BaseModel):
    in_scope: bool = Field(description="False if the question is not about these companies' annual reports")
    companies: list[Company] = Field(description="Companies the question is about; empty if none named")
    fy: Optional[FY] = Field(description="Financial year asked about, or null if not specified")
    search_query: str = Field(description="Short keyword-rich search query for the annual report")


class Grade(BaseModel):
    relevant: list[int] = Field(description="Numbers of the passages that help answer the question")


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
    rewrites: int
    answer: str


class GeminiBrain:
    """All LLM calls. Swappable: any LangChain chat model with structured output works."""

    def __init__(self, model=None, rpm=None):
        from langchain.chat_models import init_chat_model
        from langchain_core.rate_limiters import InMemoryRateLimiter

        rpm = float(rpm or os.getenv("LLM_RPM", "8"))  # stay under the free-tier per-minute cap
        limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=1)
        self.llm = init_chat_model(model or os.getenv("LLM_MODEL", "google_genai:gemini-3.8-flash"),
                                   temperature=0, rate_limiter=limiter)

    def plan(self, question):
        return self.llm.with_structured_output(Plan).invoke(
            "You route questions about Indian company annual reports (FY2025 and FY2026; "
            f"FY2026 = year ended March 31, 2026). Companies: {COMPANY_HINTS}.\n"
            f"Question: {question}")

    def grade(self, question, passages):
        listing = "\n\n".join(f"[{i}] ({p['company']} {p['fy']} p.{p['page']}) {p['text']}"
                              for i, p in enumerate(passages))
        return self.llm.with_structured_output(Grade).invoke(
            "Which passages contain information that directly helps answer the question? "
            "Be strict: a passage on the general topic without the asked fact is not relevant.\n"
            f"Question: {question}\n\nPassages:\n{listing}").relevant

    def rewrite(self, question, tried):
        return self.llm.with_structured_output(Rewrite).invoke(
            "A search over annual reports found nothing relevant. Write a different search "
            "query using the terms an annual report would use (e.g. table headings, standard "
            "financial terms, synonyms). Plain keywords only: no quotes, OR/AND or other "
            f"search operators.\nQuestion: {question}\nAlready tried: {tried}").search_query

    def answer(self, question, passages):
        if not passages:
            return "I couldn't find this in the annual reports I have (FY2025 and FY2026)."
        listing = "\n\n".join(f"[{p['company']} {p['fy']} p.{p['page']}] {p['text']}" for p in passages)
        return self.llm.invoke(
            "Answer the question using only the passages below. Cite every fact with its tag, "
            "e.g. [TCS FY2026 p.56]. If the passages don't contain the answer, say so plainly "
            "instead of guessing. If standalone and consolidated figures differ, give both.\n\n"
            f"Question: {question}\n\nPassages:\n{listing}").text


def build_graph(retriever, brain):
    from langgraph.graph import END, StateGraph

    def understand(s):
        plan = brain.plan(s["question"])
        if not plan.in_scope:
            return {"answer": OUT_OF_SCOPE, "companies": [], "fy": None, "tried_queries": [],
                    "relevant": [], "rewrites": 0}
        return {"companies": list(plan.companies), "fy": plan.fy,
                "search_query": plan.search_query, "tried_queries": [], "rewrites": 0}

    def retrieve(s):
        targets = s["companies"] or [None]
        k = max(3, K_PER_COMPANY // len(targets) + 1) if len(targets) > 1 else K_PER_COMPANY
        passages = []
        for company in targets:
            for cid in retriever.hybrid_search(s["search_query"], k=k, company=company, fy=s["fy"]):
                c = retriever.by_id[cid]
                passages.append({"id": cid, "company": c["company"], "fy": c["fy"],
                                 "page": c["page"], "text": c["text"]})
        return {"passages": passages, "tried_queries": s["tried_queries"] + [s["search_query"]]}

    def grade(s):
        keep = set(brain.grade(s["question"], s["passages"])) if s["passages"] else set()
        return {"relevant": [p for i, p in enumerate(s["passages"]) if i in keep]}

    def rewrite(s):
        return {"search_query": brain.rewrite(s["question"], s["tried_queries"]),
                "rewrites": s["rewrites"] + 1}

    def answer(s):
        return {"answer": brain.answer(s["question"], s["relevant"])}

    def after_grade(s):
        if s["relevant"] or s["rewrites"] >= MAX_REWRITES:
            return "answer"
        return "rewrite"

    g = StateGraph(State)
    for name, fn in [("understand", understand), ("retrieve", retrieve), ("grade", grade),
                     ("rewrite", rewrite), ("answer", answer)]:
        g.add_node(name, fn)
    g.set_entry_point("understand")
    g.add_conditional_edges("understand", lambda s: "end" if s.get("answer") else "retrieve",
                            {"end": END, "retrieve": "retrieve"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"answer": "answer", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", END)
    return g.compile()


def make_agent():
    from dotenv import load_dotenv

    from .retrieve import Retriever

    load_dotenv()
    return build_graph(Retriever(rerank=None), GeminiBrain())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    args = ap.parse_args()
    out = make_agent().invoke({"question": args.question})
    print(out["answer"])
    print(f"\n-- companies={out['companies']} fy={out['fy']} queries={out['tried_queries']} "
          f"relevant={[p['id'] for p in out['relevant']]}")


if __name__ == "__main__":
    main()
