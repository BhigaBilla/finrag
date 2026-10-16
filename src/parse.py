"""Extract text from every PDF page into data/pages.jsonl.

Each record keeps company, financial year and 1-based page number, which is
what citations and the hit-rate evaluation point at.
"""
import json

import fitz  # PyMuPDF

from .config import PAGES, RAW


def iter_pages(pdf_path):
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            yield i, page.get_text("text")


def main():
    n_docs = n_pages = n_empty = 0
    with open(PAGES, "w", encoding="utf-8") as f:
        for pdf in sorted(RAW.glob("*/*.pdf")):
            n_docs += 1
            for page_no, text in iter_pages(pdf):
                text = " ".join(text.split())
                n_pages += 1
                if len(text) < 50:  # cover pages, full-page images
                    n_empty += 1
                    continue
                rec = {"company": pdf.parent.name, "fy": pdf.stem,
                       "page": page_no, "text": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{n_docs} PDFs, {n_pages} pages, {n_empty} skipped as near-empty")


if __name__ == "__main__":
    main()
