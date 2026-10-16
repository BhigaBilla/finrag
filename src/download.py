"""Download annual-report PDFs from BSE's annual report archive.

BSE lists a company's annual reports at an undocumented JSON endpoint (it
needs the homepage cookie first). Each row has 'Year' (the FY end year),
'PDFDownload' (full URL) and 'Fld_ReSubmit' (1 for a revised filing). Run
with --list to see what BSE returns before downloading. A company whose report cannot be found is reported, not guessed:
drop the PDF into data/raw/<SYMBOL>/<FY>.pdf by hand instead.
"""
import argparse
import json
import re
import time

import requests

from .config import COMPANIES, RAW, YEARS

LIST_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport_New/w?scripcode={code}"
PDF_BASE = "https://www.bseindia.com/bseplus/AnnualReport/{code}/{name}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json, text/plain, */*",
}


def list_reports(session, code):
    r = session.get(LIST_URL.format(code=code), headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data.get("Table", data) if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def fy_label(row):
    """Map a BSE row to 'FY2025' etc. BSE's 'Year' is the FY end year ('2025')."""
    m = re.search(r"\b(20\d\d)\b", str(row.get("Year", "")))
    return f"FY{m.group(1)}" if m else None


def pdf_url(row, code):
    s = str(row.get("PDFDownload", "")).replace("\\", "")  # older rows have a stray '\'
    if not s.lower().endswith(".pdf"):
        return None
    return s if s.startswith("http") else PDF_BASE.format(code=code, name=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print BSE rows, download nothing")
    args = ap.parse_args()

    s = requests.Session()
    s.get("https://www.bseindia.com/", headers=HEADERS, timeout=30)  # cookies
    missing = []
    for sym, code in COMPANIES.items():
        rows = list_reports(s, code)
        if args.list:
            print(sym, json.dumps(rows[:4], indent=1))
            continue
        # Prefer the original filing; 'Revised' rows (Fld_ReSubmit=1) are resubmissions.
        found = {}
        for r in sorted(rows, key=lambda r: r.get("Fld_ReSubmit") == 0):
            found[fy_label(r)] = pdf_url(r, code)
        for fy in YEARS:
            out = RAW / sym / f"{fy}.pdf"
            if out.exists():
                print(f"{sym} {fy}: cached")
                continue
            url = found.get(fy)
            if not url:
                missing.append(f"{sym} {fy}")
                continue
            r = s.get(url, headers=HEADERS, timeout=120)
            if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                missing.append(f"{sym} {fy} (HTTP {r.status_code})")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r.content)
            print(f"{sym} {fy}: {len(r.content) / 1e6:.1f} MB")
            time.sleep(1)
    if missing:
        print("\nNot downloaded:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
