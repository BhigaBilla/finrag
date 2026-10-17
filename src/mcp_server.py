"""MCP server exposing the filings search as tools, for any MCP client
(Claude Desktop, Claude Code, or the agent itself).

    python -m src.mcp_server            # stdio transport

Tools are read-only: list the reports, hybrid-search them (optionally
filtered by company and year), and fetch a full page for checking a citation.
"""
import json
from functools import lru_cache
from typing import Optional

from mcp.server.mcpserver import MCPServer

from .config import COMPANIES, PAGES, YEARS

server = MCPServer(
    name="finrag",
    instructions="Search FY2025/FY2026 annual reports of 10 Nifty 50 companies. Cite results "
                 "as [COMPANY FY p.N]; use get_page to read a whole page before quoting a number.",
)


@lru_cache(maxsize=1)
def retriever():
    from .retrieve import Retriever
    return Retriever(rerank=None)


@lru_cache(maxsize=1)
def pages():
    out = {}
    with open(PAGES, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            out[(p["company"], p["fy"], p["page"])] = p["text"]
    return out


@server.tool()
def list_reports() -> dict:
    """List the companies (by symbol) and financial years available."""
    return {"companies": list(COMPANIES), "financial_years": YEARS}


@server.tool()
def search_filings(query: str, company: Optional[str] = None, fy: Optional[str] = None,
                   k: int = 8) -> list[dict]:
    """Hybrid (BM25 + embedding) search over annual-report passages.

    company: symbol from list_reports (e.g. "TCS"); fy: "FY2025" or "FY2026".
    Returns passages with company, fy, page and text.
    """
    if company and company not in COMPANIES:
        raise ValueError(f"unknown company {company!r}; use one of {list(COMPANIES)}")
    if fy and fy not in YEARS:
        raise ValueError(f"unknown fy {fy!r}; use one of {YEARS}")
    r = retriever()
    hits = r.hybrid_search(query, k=max(1, min(k, 25)), company=company, fy=fy)
    return [{k_: r.by_id[c][k_] for k_ in ("company", "fy", "page", "text")} for c in hits]


@server.tool()
def get_page(company: str, fy: str, page: int) -> str:
    """Full extracted text of one page of an annual report."""
    text = pages().get((company, fy, page))
    if text is None:
        raise ValueError(f"no text for {company} {fy} page {page}")
    return text


if __name__ == "__main__":
    server.run("stdio")
