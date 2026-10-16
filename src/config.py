"""Corpus definition and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded PDFs: raw/<SYMBOL>/<FY>.pdf
PAGES = DATA / "pages.jsonl"  # one record per PDF page

# 10 Nifty 50 companies across sectors, with BSE scrip codes.
COMPANIES = {
    "RELIANCE": 500325,
    "TCS": 532540,
    "INFY": 500209,
    "HDFCBANK": 500180,
    "ICICIBANK": 532174,
    "ITC": 500875,
    "LT": 500510,
    "BHARTIARTL": 532454,
    "MARUTI": 532500,
    "SUNPHARMA": 524715,
}

# Annual reports for the two most recent financial years.
YEARS = ["FY2025", "FY2026"]
