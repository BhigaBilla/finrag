from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from src import api
from src.agent import build_graph
from tests.test_agent import FakeRetriever, ScriptedBrain, _multi_brain


def client(brain):
    r, saver = FakeRetriever(), InMemorySaver()
    api.STATE.clear()
    api.STATE.update(retriever=r, threads={},
                     agents={p: build_graph(r, brain, approval=p, checkpointer=saver)
                             for p in ("never", "multi", "always")})
    return TestClient(api.app), r


def test_ask_answers_with_citations():
    c, _ = client(ScriptedBrain([[1]]))
    body = c.post("/ask", json={"question": "q"}).json()
    assert body["status"] == "answered" and body["answer"] == "ok"
    assert body["citations"] == [{"company": "TCS", "fy": "FY2026", "page": 2}]


def test_approval_round_trip():
    c, r = client(_multi_brain([[0]]))
    body = c.post("/ask", json={"question": "compare", "approval": "multi"}).json()
    assert body["status"] == "needs_approval" and r.calls == []
    body = c.post("/resume", json={"thread_id": body["thread_id"], "action": "approve"}).json()
    assert body["status"] == "answered" and len(r.calls) == 2


def test_search_validates_company():
    c, _ = client(ScriptedBrain([]))
    assert c.get("/search", params={"q": "x", "company": "APPLE"}).status_code == 400
    assert c.get("/search", params={"q": "x", "company": "TCS"}).json()[0]["page"] in (1, 2)
