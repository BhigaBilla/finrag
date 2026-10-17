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
    def __init__(self, grades, in_scope=True):
        self.in_scope = in_scope
        self.grades = list(grades)
        self.answered_with = None

    def plan(self, q):
        return SimpleNamespace(in_scope=self.in_scope, companies=["TCS"], fy="FY2026", search_query="attrition")

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


def test_out_of_scope_skips_retrieval():
    from src.agent import OUT_OF_SCOPE
    r, b = FakeRetriever(), ScriptedBrain([], in_scope=False)
    out = build_graph(r, b).invoke({"question": "What is the capital of France?"})
    assert out["answer"] == OUT_OF_SCOPE and r.calls == [] and b.answered_with is None


def _multi_brain(grades):
    b = ScriptedBrain(grades)
    b.plan = lambda q: SimpleNamespace(in_scope=True, companies=["TCS", "INFY"], fy="FY2026",
                                       search_query="attrition")
    return b


def test_approval_pauses_multi_company_and_resumes_with_edit():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    r, b = FakeRetriever(), _multi_brain([[0]])
    g = build_graph(r, b, approval="multi", checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    out = g.invoke({"question": "compare"}, cfg)
    assert out["__interrupt__"][0].value["companies"] == ["TCS", "INFY"] and r.calls == []
    out = g.invoke(Command(resume={"action": "edit", "companies": ["TCS"], "search_query": "attrition rate"}), cfg)
    assert out["answer"] == "ok" and r.calls == [("attrition rate", "TCS", "FY2026")]


def test_approval_reject_stops_without_search():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    r, b = FakeRetriever(), _multi_brain([])
    g = build_graph(r, b, approval="multi", checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    g.invoke({"question": "compare"}, cfg)
    out = g.invoke(Command(resume={"action": "reject"}), cfg)
    assert out["answer"].startswith("Stopped") and r.calls == []


def test_single_company_not_paused_under_multi_policy():
    from langgraph.checkpoint.memory import InMemorySaver

    r, b = FakeRetriever(), ScriptedBrain([[0]])
    g = build_graph(r, b, approval="multi", checkpointer=InMemorySaver())
    out = g.invoke({"question": "q"}, {"configurable": {"thread_id": "t3"}})
    assert out["answer"] == "ok" and "__interrupt__" not in out
