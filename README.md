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

### Results (run 2026-10-17, CPU: Intel i5-1334U)

58 of 60 questions have an answer in the report (q09 Infosys guidance and
q14 Jio ARPU FY2025 are not stated anywhere, so they're excluded). 258 gold
pages in total, 4.4 per question. Search is filtered to the question's
company and year. Page-level metrics:

| method | hit@1 | hit@3 | hit@5 | hit@10 | MRR | s/query |
|---|---|---|---|---|---|---|
| BM25 | 0.448 | 0.690 | 0.828 | 0.879 | 0.605 | 0.24 |
| Dense (bge-small) | 0.534 | 0.793 | 0.879 | 0.897 | 0.673 | 0.04 |
| Hybrid (RRF) | 0.655 | 0.879 | 0.914 | 0.931 | 0.766 | 0.14 |
| Hybrid + MiniLM-L6 rerank | 0.569 | 0.845 | 0.879 | **0.966** | 0.710 | 1.59 |
| Hybrid + bge-reranker-base | **0.690** | 0.879 | 0.914 | 0.948 | **0.791** | 9.37 |

By question type (MRR, hybrid / bge rerank): narrative (n=17) 0.794 / 0.819,
numeric in prose (n=36) 0.792 / 0.809, table-only (n=5) 0.485 / 0.567.

Without the company/year filter (searching all 20 reports), hit@1 drops for
every method, e.g. hybrid 0.655 -> 0.345 and MiniLM rerank 0.569 -> 0.431,
while hit@10 stays close (0.914 / 0.966).

What this does and doesn't show:
- Hybrid clearly beats either retriever alone (MRR 0.766 vs 0.673 dense, 0.605 BM25).
- The bge reranker's gain over plain hybrid (hit@1 0.690 vs 0.655) is 2
  questions out of 58, which is within noise (one-sigma on hit@1 is about
  ±0.06). It costs ~65x the latency on CPU, so the default pipeline uses
  hybrid without reranking.
- The small MiniLM reranker (trained on web search) makes rank 1 worse than
  no reranking. It only helps hit@10.
- Metadata filtering matters more than any model choice, which is why the
  agent extracts company and year from the question before searching.
- Table-only answers are the weak spot (n=5, so treat as indicative only):
  plain-text PDF extraction scrambles table layout.

Limitations of the evaluation: the questions were written and the gold pages
judged by Claude (an LLM), not by a human annotator. Labels used pooling
plus direct text search; 34% of gold pages were found only by the direct
search, so pooling alone would have under-counted relevant pages. 58
questions is small, so differences under ~0.1 are not meaningful.

## Step 4: agent

LangGraph graph: **understand** (company and year found by name matching, no
LLM) -> **retrieve** (hybrid search, filtered by company and year) ->
**answer** (one LLM call that picks the passages containing the asked fact and
writes the answer with page citations like `[TCS FY2026 p.56]`, or reports
"not found") -> if nothing was found, **rewrite** the query once and try again.
A question that names none of the 10 companies gets a fixed out-of-scope reply
without any search or LLM call.

```bash
python -m src.agent "What was TCS's attrition rate in FY2026?"
```

The LLM sits behind LangChain's `init_chat_model` (default Gemini 3.8 Flash,
free tier, rate-limited to 8 requests/min), so the provider is a one-line
change in `.env`.

First run on 8 questions (2026-10-17): 6 correct, 2 partially correct, 0
hallucinated. All three "not in the reports" / out-of-scope questions were
refused correctly, and every citation checked pointed to a page that supports
the claim. The two partial answers both came from retrieval: for Maruti's
domestic sales the agent read a chart whose columns were scrambled by PDF
extraction and offered two figures; for ICICI's RoE it found the standalone
figure but not the consolidated one. Fixes since: retrieve 10 passages per
company instead of 6, route out-of-scope questions without searching, and
keep rewritten queries as plain keywords.

Cost: the project runs entirely on Gemini's free tier, which allows about 20
requests per day per model. The first version spent 3 to 6 LLM calls per
question (planning, grading, answering, rewrites), so the agent was redesigned:
routing is now rule-based (step 3 showed the company/year filter matters more
than any retriever choice, and name matching gets it for free), grading and
answering are one structured call, and only one rewrite is allowed. A typical
question now costs 1 call, the worst case 3, and out-of-scope questions 0.

## Step 5: MCP server and human approval

`src/mcp_server.py` exposes the search as an MCP server (stdio) with three
read-only tools: `list_reports`, `search_filings` (hybrid search with
company/year filters) and `get_page` (full page text, for checking a
citation). Any MCP client can use it, e.g. Claude Desktop:

```json
{"mcpServers": {"finrag": {"command": "D:/data/ai_projects/finrag/.venv/Scripts/python.exe",
  "args": ["-m", "src.mcp_server"], "cwd": "D:/data/ai_projects/finrag"}}}
```

`python scripts/mcp_smoke.py` starts the server as a client would and calls each tool.

Human approval: the agent pauses before searching (LangGraph `interrupt`) and
shows the plan (companies, year, search query). You can approve it, reject
it, or replace the query. The default policy `multi` pauses only when the
search fans out over several companies, where a wrong plan wastes the most:

```bash
python -m src.agent "Compare HDFC Bank and ICICI Bank gross NPA in FY2026" --approval multi
```

## Step 6: answer evaluation (in progress)

Rule-based routing on all 60 eval questions (no LLM): the company was right
for 60/60. The 38 questions that state a year all got the right year; the
other 22 state no year, so the agent searches both reports (for the eval, the
year of the gold answer is appended to these).

`src/answer_eval.py` runs the agent on every question, then has a different
free Gemini model grade each answer against the reference answer
(correct / partial / wrong / refused) and check faithfulness against the cited
passages only. Citation tags are also checked without an LLM. Both steps skip
finished questions and stop at the daily free quota, so they are spread over
several days:

```bash
python -m src.answer_eval run
JUDGE_MODEL=google_genai:<another free model> python -m src.answer_eval judge
python -m src.answer_eval summary
```
