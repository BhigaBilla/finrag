import asyncio

from src import mcp_server


class FakeRetriever:
    by_id = {"TCS|FY2026|56|0": {"company": "TCS", "fy": "FY2026", "page": 56, "text": "attrition 13.7%"}}

    def hybrid_search(self, query, k, company, fy):
        self.last = (query, k, company, fy)
        return list(self.by_id)


def test_tools_listed_and_search_filters(monkeypatch):
    fake = FakeRetriever()
    monkeypatch.setattr(mcp_server, "retriever", lambda: fake)
    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {"list_reports", "search_filings", "get_page"}
    out = mcp_server.search_filings("attrition", company="TCS", fy="FY2026", k=100)
    assert out[0]["page"] == 56 and fake.last == ("attrition", 25, "TCS", "FY2026")


def test_search_rejects_unknown_company():
    import pytest
    with pytest.raises(ValueError):
        mcp_server.search_filings("x", company="APPLE")
