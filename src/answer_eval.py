"""Step 6: answer evaluation, paced to fit Gemini's free tier.

    python -m src.answer_eval run      # agent answers each question -> eval/answers.jsonl
    python -m src.answer_eval judge    # LLM judge grades each answer -> eval/judgements.jsonl
    python -m src.answer_eval summary  # prints and writes eval/answer_summary.json

`run` and `judge` skip questions already done and stop cleanly at the daily
quota (HTTP 429), so rerun them on later days until everything is done. The
judge should use a different model from the agent (JUDGE_MODEL) so the two
daily quotas don't collide and the model doesn't grade its own answers.

Questions with no year get "(FYxxxx)" appended, as a user asking about one
report would say: the gold answer belongs to one year's report.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

from pydantic import BaseModel, Field

from .agent import NOT_FOUND, build_graph, route
from .config import ROOT

EVAL = ROOT / "eval"
QUESTIONS, ANSWERS, JUDGEMENTS = EVAL / "questions.jsonl", EVAL / "answers.jsonl", EVAL / "judgements.jsonl"
NOT_IN_REPORT = "NOT IN REPORT"
TAG = re.compile(r"\[([A-Z]+) (FY\d{4}) p\.(\d+)\]")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_quota_error(e):
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


class DailyQuota(Exception):
    pass


def call(fn, retries=2):
    """Run fn; wait out a per-minute 429 (up to `retries` times), raise DailyQuota on a per-day one.
    The model is built with max_retries=0, so this is the only retrying."""
    import time

    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if is_overloaded(e) and attempt < retries:
                print(f"  model overloaded (503), waiting 60 s")
                time.sleep(60)
                continue
            if not is_quota_error(e):
                raise
            if "PerDay" in str(e) or attempt == retries:
                raise DailyQuota(quota_detail(e)) from e
            delay = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", str(e))
            wait = min(int(delay.group(1)) if delay else 60, 120) + 2
            print(f"  per-minute limit ({quota_detail(e)}), waiting {wait} s")
            time.sleep(wait)


def is_overloaded(e):
    return "503" in str(e) or "UNAVAILABLE" in str(e)


def quota_detail(e):
    """Which quota was hit (per-minute vs per-day) and when to retry, from the 429 text."""
    found = re.findall(r"(quotaId|retryDelay)['\"]?\s*[:=]\s*['\"]?([\w.-]+)", str(e))
    return ", ".join(f"{k}={v}" for k, v in dict(found).items()) or str(e)[:200]


def eval_question(q):
    text = q["question"]
    if route(text)[1] is None:
        text += f" ({q['fy']})"
    return text


class CountingBrain:
    def __init__(self, brain):
        self.brain, self.calls = brain, 0

    def answer(self, *a):
        self.calls += 1
        return self.brain.answer(*a)

    def rewrite(self, *a):
        self.calls += 1
        return self.brain.rewrite(*a)


def run(limit):
    from dotenv import load_dotenv

    from .agent import GeminiBrain
    from .retrieve import Retriever

    load_dotenv()
    done = {r["id"] for r in load(ANSWERS)}
    todo = [q for q in load(QUESTIONS) if q["id"] not in done][:limit]
    model = os.getenv("LLM_MODEL", "google_genai:gemini-3.8-flash")
    retriever, brain = Retriever(rerank=None), CountingBrain(GeminiBrain(model=model))
    graph = build_graph(retriever, brain)
    for q in todo:
        before = brain.calls
        try:
            out = call(lambda: graph.invoke({"question": eval_question(q)}))
        except DailyQuota as e:
            print(f"Daily quota reached at {q['id']} ({e}); rerun after the reset.")
            break
        append(ANSWERS, {"id": q["id"], "model": model, "question": eval_question(q), "answer": out["answer"],
                         "found": out.get("found", False), "llm_calls": brain.calls - before,
                         "queries": out.get("tried_queries", []),
                         "cited": [{"id": p["id"], "company": p["company"], "fy": p["fy"],
                                    "page": p["page"]} for p in out.get("relevant", [])]})
        print(q["id"], brain.calls - before, "calls", out["answer"][:100].replace("\n", " "))
    print(f"{len(load(ANSWERS))}/{len(load(QUESTIONS))} answered")


class Verdict(BaseModel):
    correctness: str = Field(description="correct, partial, wrong, or refused")
    faithful: bool = Field(description="True if every claim in the answer is supported by the cited passages")
    reason: str = Field(description="One sentence")


def judge_prompt(q, a, passages):
    cited = "\n\n".join(f"[{p['company']} {p['fy']} p.{p['page']}] {p['text']}" for p in passages) or "(none)"
    return (
        "You grade an answer from a question-answering system over company annual reports.\n"
        "correctness: compare with the reference answer. 'correct' if it states the reference's key "
        "facts (same numbers, small rounding is fine); 'partial' if some key facts are missing or one "
        "of several is wrong; 'wrong' if the main fact is wrong or it answers a different question; "
        "'refused' if it says the information wasn't found. If the reference is NOT IN REPORT, a "
        "refusal is the right behaviour and you still label it 'refused'.\n"
        "faithful: judge ONLY against the cited passages, not the reference or your own knowledge. "
        "True if every factual claim in the answer appears in the cited passages. A refusal is faithful.\n\n"
        f"Question: {q['question']}\nReference answer: {q['answer']}\nSystem answer: {a['answer']}\n\n"
        f"Cited passages:\n{cited}")


def judge(limit):
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model

    from .agent import GeminiBrain
    from .retrieve import Retriever

    load_dotenv()
    model = os.getenv("JUDGE_MODEL")
    if not model:
        sys.exit("Set JUDGE_MODEL to a different model from LLM_MODEL (another free model from your AI Studio rate-limit page)")
    llm = GeminiBrain(model=model).llm.with_structured_output(Verdict)
    by_id = Retriever(rerank=None).by_id
    qs = {q["id"]: q for q in load(QUESTIONS)}
    done = {r["id"] for r in load(JUDGEMENTS)}
    # never let a model grade its own answers
    todo = [a for a in load(ANSWERS) if a["id"] not in done and a.get("model") != model][:limit]
    for a in todo:
        prompt = judge_prompt(qs[a["id"]], a, [by_id[c["id"]] for c in a["cited"]])
        try:
            v = call(lambda: llm.invoke(prompt))
        except DailyQuota as e:
            print(f"Daily quota reached at {a['id']} ({e}); rerun after the reset.")
            break
        append(JUDGEMENTS, {"id": a["id"], "answered_by": a.get("model"), "judge": model, **v.model_dump()})
        print(a["id"], v.correctness, "faithful" if v.faithful else "UNFAITHFUL", v.reason[:90])
    print(f"{len(load(JUDGEMENTS))}/{len(load(ANSWERS))} judged (answers from {model} are left for another judge)")


def citation_check(a, q):
    """No LLM: do the answer's citation tags point to retrieved passages, and to gold pages?"""
    tags = {(c, fy, int(p)) for c, fy, p in TAG.findall(a["answer"])}
    retrieved = {(c["company"], c["fy"], c["page"]) for c in a["cited"]}
    gold = {(q["company"], q["fy"], p) for p in q["gold_pages"]}
    return {"tags": len(tags), "tags_not_retrieved": len(tags - retrieved),
            "cites_gold": bool(tags & gold)}


def summary():
    qs = {q["id"]: q for q in load(QUESTIONS)}
    answers = {a["id"]: a for a in load(ANSWERS)}
    verdicts = {v["id"]: v for v in load(JUDGEMENTS)}
    ans = [i for i in answers if qs[i]["answer"] != NOT_IN_REPORT]
    neg = [i for i in answers if qs[i]["answer"] == NOT_IN_REPORT]
    judged = [i for i in ans if i in verdicts]
    found = [i for i in ans if answers[i]["found"]]
    checks = {i: citation_check(answers[i], qs[i]) for i in found}
    by_type = {}
    for i in judged:
        by_type.setdefault(qs[i]["type"], Counter())[verdicts[i]["correctness"]] += 1
    out = {
        "answered": len(answers), "judged": len(judged),
        "correctness": dict(Counter(verdicts[i]["correctness"] for i in judged)),
        "correctness_by_type": {t: dict(c) for t, c in by_type.items()},
        "faithful_rate_of_non_refusals": _rate([verdicts[i]["faithful"] for i in judged
                                                 if answers[i]["found"]]),
        "answers_citing_a_gold_page": _rate([c["cites_gold"] for c in checks.values()]),
        "answers_with_a_citation_not_retrieved": sum(c["tags_not_retrieved"] > 0 for c in checks.values()),
        "answers_without_citation_tag": sum(c["tags"] == 0 for c in checks.values()),
        "not_in_report_refused": f"{sum(answers[i]['answer'] == NOT_FOUND for i in neg)}/{len(neg)}",
        "llm_calls": dict(Counter(a["llm_calls"] for a in answers.values())),
        "mean_llm_calls": round(sum(a["llm_calls"] for a in answers.values()) / max(len(answers), 1), 2),
    }
    (EVAL / "answer_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _rate(xs):
    return f"{sum(xs)}/{len(xs)}" if xs else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "judge", "summary"])
    ap.add_argument("--limit", type=int, default=1000, help="max questions this run")
    args = ap.parse_args()
    {"run": lambda: run(args.limit), "judge": lambda: judge(args.limit), "summary": summary}[args.cmd]()


if __name__ == "__main__":
    main()
