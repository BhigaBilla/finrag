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
