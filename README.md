# FinRAG

Agentic RAG over Indian company annual reports (10 Nifty 50 companies, FY2025
and FY2026), with hybrid retrieval, page-level citations and an evaluation of
retrieval hit-rate and answer faithfulness.

Work in progress: see [PLAN.md](PLAN.md) for the build steps. Results are
added here as each step is finished.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in the keys
pytest
```

## Step 1: data

```bash
python -m src.download --list   # inspect what BSE returns
python -m src.download          # PDFs -> data/raw/<SYMBOL>/<FY>.pdf
python -m src.parse             # page text -> data/pages.jsonl
```

Result (run 2026-10-16): all 20 reports downloaded (371 MB), 8,194 pages, of
which 48 near-empty pages (covers, full-page images) are skipped. That leaves
8,146 pages and 28.2M characters of text. Per report: 174 pages (Reliance,
dense) to 756 (L&T). Where BSE has both an original and a "Revised" filing
(INFY FY2026, HDFCBANK FY2025) the original is used.

Limitation: PyMuPDF plain-text extraction flattens tables, so numbers in
financial statements lose their row/column structure. Scanned pages without
a text layer are not OCR'd.

## Step 2: retrieval

```bash
python -m src.chunk                  # pages -> 250-word chunks (50 overlap), page-bounded
python -m src.retrieve --build       # BM25 index + bge-small embeddings
python -m src.retrieve --query "attrition rate" --company TCS
```

## Step 3: retrieval evaluation

60 hand-written questions (6 per company), each tagged narrative / numeric
(number stated in prose) / table (number only in a table). Gold pages are
labelled by pooling the top-10 results of every method and judging each
pooled page, plus a direct text search for answers no method found.

```bash
python -m src.pool                         # eval/pool/<qid>.md for judging
python -m src.evaluate --reranker minilm   # hit@k and MRR per method
python -m src.evaluate --reranker bge
```
