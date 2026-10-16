"""Split page text into overlapping word windows -> data/chunks.jsonl.

Chunks never cross a page boundary, so every chunk cites exactly one page.
Each chunk's text is prefixed with its source ("TCS FY2026 annual report,
page 123") so both BM25 and the embedding model see which company it is from.
"""
import json

from .config import CHUNKS, PAGES

CHUNK_WORDS = 250
OVERLAP_WORDS = 50


def split_words(words, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    if len(words) <= size:
        return [words]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def header(company, fy, page):
    return f"{company} {fy} annual report, page {page}."


def main():
    n = 0
    with open(PAGES, encoding="utf-8") as f, open(CHUNKS, "w", encoding="utf-8") as out:
        for line in f:
            p = json.loads(line)
            for i, words in enumerate(split_words(p["text"].split())):
                rec = {"id": f"{p['company']}|{p['fy']}|{p['page']}|{i}",
                       "company": p["company"], "fy": p["fy"], "page": p["page"],
                       "text": header(p["company"], p["fy"], p["page"]) + " " + " ".join(words)}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} chunks written to {CHUNKS}")


if __name__ == "__main__":
    main()
