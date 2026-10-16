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
