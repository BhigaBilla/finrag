"""Graph routing with a scripted brain and a fake retriever (no LLM, no index)."""
from types import SimpleNamespace

from src.agent import MAX_REWRITES, build_graph


class FakeRetriever:
    def __init__(self):
        self.by_id = {f"TCS|FY2026|{p}|0": {"company": "TCS", "fy": "FY2026", "page": p, "text": f"page {p}"}
                      for p in (1, 2)}
        self.calls = []

    def hybrid_search(self, query, k, company, fy):
        self.calls.append((query, company, fy))
        return list(self.by_id)


class ScriptedBrain:
    def __init__(self, grades):
        self.grades = list(grades)
        self.answered_with = None

    def plan(self, q):
        return SimpleNamespace(companies=["TCS"], fy="FY2026", search_query="attrition")

    def grade(self, q, passages):
        return self.grades.pop(0)

    def rewrite(self, q, tried):
        return f"query{len(tried)}"

    def answer(self, q, passages):
        self.answered_with = passages
        return "ok"


def test_answers_directly_when_first_retrieval_is_relevant():
    r, b = FakeRetriever(), ScriptedBrain([[1]])
    out = build_graph(r, b).invoke({"question": "q"})
    assert out["answer"] == "ok" and out["rewrites"] == 0
    assert [p["page"] for p in b.answered_with] == [2]
    assert r.calls == [("attrition", "TCS", "FY2026")]  # filter passed through


def test_rewrites_then_gives_up_with_no_passages():
    r, b = FakeRetriever(), ScriptedBrain([[]] * (MAX_REWRITES + 1))
    out = build_graph(r, b).invoke({"question": "q"})
    assert out["rewrites"] == MAX_REWRITES
    assert len(r.calls) == MAX_REWRITES + 1
    assert out["tried_queries"] == ["attrition", "query1", "query2"]
    assert b.answered_with == []
