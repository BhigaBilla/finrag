"""Build judging pools for the eval questions (TREC-style pooling).

For each question, take the union of the top pages from every method and
write them, with full page text, to eval/pool/<qid>.md. A judge then marks
which pages answer the question in eval/questions.jsonl ("gold_pages", plus
the answer and where it sits). Pooling from all methods keeps the labels
from favouring any one retriever; the judge should also search pages.jsonl
directly for answers no method found.
"""
import json

from .config import PAGES, ROOT
from .retrieve import Retriever

QUESTIONS = ROOT / "eval" / "questions.jsonl"
POOL_DIR = ROOT / "eval" / "pool"
DEPTH = 10  # chunks per method


def page_key(chunk_id):
    company, fy, page, _ = chunk_id.split("|")
    return company, fy, int(page)


def main():
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pages = {}
    with open(PAGES, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pages[(p["company"], p["fy"], p["page"])] = p["text"]
    r = Retriever(rerank="minilm")
    for line in open(QUESTIONS, encoding="utf-8"):
        q = json.loads(line)
        found = {}
        for m in ["bm25", "dense", "hybrid", "rerank"]:
            for cid in r.search(q["question"], m, k=DEPTH, company=q["company"], fy=q["fy"]):
                found.setdefault(page_key(cid), set()).add(m)
        with open(POOL_DIR / f"{q['id']}.md", "w", encoding="utf-8") as out:
            out.write(f"# {q['id']}: {q['question']}\n({q['company']} {q['fy']})\n\n")
            for key in sorted(found):
                out.write(f"## {key[0]} {key[1]} page {key[2]}  [found by: {', '.join(sorted(found[key]))}]\n\n")
                out.write(pages.get(key, "(page text missing)") + "\n\n")
        print(q["id"], len(found), "pages pooled")


if __name__ == "__main__":
    main()
