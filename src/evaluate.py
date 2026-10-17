"""Page-level retrieval metrics on the labelled eval set.

A retrieved chunk counts as a hit if its page is one of the question's gold
pages. Ranked chunk lists are collapsed to ranked unique pages first, so
hit@k means "a gold page is among the first k distinct pages".
"""
import argparse
import json
import time
from collections import defaultdict

from .config import ROOT
from .pool import QUESTIONS, page_key
from .retrieve import Retriever

KS = (1, 3, 5, 10)
METHODS = ["bm25", "dense", "hybrid", "rerank"]


def ranked_pages(chunk_ids):
    seen, out = set(), []
    for cid in chunk_ids:
        key = page_key(cid)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def score(pages, gold):
    """Return (hits at each k, reciprocal rank) for one question."""
    rank = next((i for i, p in enumerate(pages, 1) if p in gold), None)
    hits = {k: int(rank is not None and rank <= k) for k in KS}
    return hits, (1.0 / rank if rank else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", default="minilm")
    ap.add_argument("--no-filter", action="store_true", help="search all companies/years")
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
    qs = [q for q in qs if q["gold_pages"]]
    r = Retriever(rerank=args.reranker)
    rows, per_q = [], []
    for m in METHODS:
        agg = defaultdict(list)
        for q in qs:
            gold = {(q["company"], q["fy"], p) for p in q["gold_pages"]}
            kw = {} if args.no_filter else {"company": q["company"], "fy": q["fy"]}
            t0 = time.perf_counter()
            chunks = r.search(q["question"], m, k=30, **kw)
            dt = time.perf_counter() - t0
            hits, rr = score(ranked_pages(chunks), gold)
            for bucket in ("all", q["type"]):
                for k in KS:
                    agg[(bucket, f"hit@{k}")].append(hits[k])
                agg[(bucket, "mrr")].append(rr)
            agg[("all", "sec")].append(dt)
            per_q.append({"id": q["id"], "method": m, "rr": round(rr, 3)})
        for bucket in ["all", "narrative", "numeric", "table"]:
            n = len(agg[(bucket, "mrr")])
            if not n:
                continue
            row = {"method": m, "subset": bucket, "n": n}
            for k in KS:
                row[f"hit@{k}"] = round(sum(agg[(bucket, f"hit@{k}")]) / n, 3)
            row["mrr"] = round(sum(agg[(bucket, "mrr")]) / n, 3)
            if bucket == "all":
                row["sec/query"] = round(sum(agg[("all", "sec")]) / n, 2)
            rows.append(row)

    tag = f"{args.reranker}{'_nofilter' if args.no_filter else ''}"
    out = ROOT / "eval" / f"retrieval_{tag}.json"
    out.write_text(json.dumps({"summary": rows, "per_question": per_q}, indent=1))
    print(f"{len(qs)} labelled questions, reranker={args.reranker}, filter={not args.no_filter}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
