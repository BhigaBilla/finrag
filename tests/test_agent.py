"""Graph routing with a scripted brain and a fake retriever (no LLM, no index)."""
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agent import MAX_REWRITES, NOT_FOUND, OUT_OF_SCOPE, build_graph, route


class FakeRetriever:
    def __init__(self):
        self.by_id = {f"TCS|FY2026|{p}|0": {"company": "TCS", "fy": "FY2026", "page": p, "text": f"page {p}"}
                      for p in (1, 2)}
        self.calls = []

    def hybrid_search(self, query, k, company, fy):
        self.calls.append((query, company, fy))
        return list(self.by_id)


class ScriptedBrain:
    """answers: list of relevant-index lists; [] means 'not found'."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.llm_calls = 0

    def answer(self, q, passages):
        self.llm_calls += 1
        rel = self.answers.pop(0)
        return SimpleNamespace(relevant=rel, found=bool(rel), answer="ok" if rel else "")

    def rewrite(self, q, tried):
        self.llm_calls += 1
        return f"query{len(tried)}"


Q_TCS = "What was TCS's attrition in FY2026?"
Q_TWO = "Compare TCS and Infosys attrition in FY2026"


def test_route_companies_and_years():
    assert route(Q_TCS) == (["TCS"], "FY2026")
    assert route("Jio subscribers in 2025-26") == (["RELIANCE"], "FY2026")
    assert route("L&T order book at March 31, 2025") == (["LT"], "FY2025")
    assert route("HDFC Bank vs ICICI NPA") == (["HDFCBANK", "ICICIBANK"], None)
    assert route("Sun Pharma FY25 and FY26 R&D") == (["SUNPHARMA"], None)
    assert route("What is the capital of France?") == ([], None)


def test_answers_in_one_llm_call():
    r, b = FakeRetriever(), ScriptedBrain([[1]])
    out = build_graph(r, b).invoke({"question": Q_TCS})
    assert out["answer"] == "ok" and b.llm_calls == 1
    assert [p["page"] for p in out["relevant"]] == [2]
    assert r.calls == [(Q_TCS, "TCS", "FY2026")]


def test_rewrites_once_then_reports_not_found():
    r, b = FakeRetriever(), ScriptedBrain([[]] * (MAX_REWRITES + 1))
    out = build_graph(r, b).invoke({"question": Q_TCS})
    assert out["answer"] == NOT_FOUND and out["relevant"] == []
    assert out["tried_queries"] == [Q_TCS, "query1"]
    assert b.llm_calls == 2 * MAX_REWRITES + 1


def test_out_of_scope_costs_no_llm_call():
    r, b = FakeRetriever(), ScriptedBrain([])
    out = build_graph(r, b).invoke({"question": "What is the capital of France?"})
    assert out["answer"] == OUT_OF_SCOPE and r.calls == [] and b.llm_calls == 0


def test_approval_pauses_multi_company_and_resumes_with_edit():
    r, b = FakeRetriever(), ScriptedBrain([[0]])
    g = build_graph(r, b, approval="multi", checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    out = g.invoke({"question": Q_TWO}, cfg)
    assert out["__interrupt__"][0].value["companies"] == ["TCS", "INFY"] and r.calls == []
    out = g.invoke(Command(resume={"action": "edit", "companies": ["TCS"], "search_query": "attrition rate"}), cfg)
    assert out["answer"] == "ok" and r.calls == [("attrition rate", "TCS", "FY2026")]


def test_approval_reject_stops_without_search():
    r, b = FakeRetriever(), ScriptedBrain([])
    g = build_graph(r, b, approval="multi", checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    g.invoke({"question": Q_TWO}, cfg)
    out = g.invoke(Command(resume={"action": "reject"}), cfg)
    assert out["answer"].startswith("Stopped") and r.calls == [] and b.llm_calls == 0


def test_single_company_not_paused_under_multi_policy():
    r, b = FakeRetriever(), ScriptedBrain([[0]])
    g = build_graph(r, b, approval="multi", checkpointer=InMemorySaver())
    out = g.invoke({"question": Q_TCS}, {"configurable": {"thread_id": "t3"}})
    assert out["answer"] == "ok" and "__interrupt__" not in out


def test_answer_eval_helpers():
    from src.answer_eval import citation_check, eval_question
    q = {"question": "Who are TCS's statutory auditors?", "fy": "FY2026", "company": "TCS", "gold_pages": [2]}
    assert eval_question(q).endswith("(FY2026)")
    a = {"answer": "B S R & Co. [TCS FY2026 p.2] [TCS FY2026 p.9]",
         "cited": [{"company": "TCS", "fy": "FY2026", "page": 2}]}
    assert citation_check(a, q) == {"tags": 2, "tags_not_retrieved": 1, "cites_gold": True}


def test_quota_handling_waits_on_per_minute_and_stops_on_per_day(monkeypatch):
    import time

    import pytest

    from src.answer_eval import DailyQuota, call
    monkeypatch.setattr(time, "sleep", lambda s: None)
    errs = [Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, retryDelay: 30s")]

    def flaky():
        if errs:
            raise errs.pop()
        return "ok"
    assert call(flaky) == "ok"

    def daily():
        raise Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    with pytest.raises(DailyQuota):
        call(daily)


def test_overloaded_model_is_retried(monkeypatch):
    import time

    from src.answer_eval import call
    monkeypatch.setattr(time, "sleep", lambda s: None)
    errs = [Exception("503 UNAVAILABLE This model is currently experiencing high demand")]

    def flaky():
        if errs:
            raise errs.pop()
        return "ok"
    assert call(flaky) == "ok"
