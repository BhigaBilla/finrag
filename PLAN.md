# FinRAG build plan

Agentic RAG over Indian company annual reports: ask a question, get an answer
with page-level citations, or an honest "not in the filings".

## Data
- 10 Nifty 50 companies across sectors: Reliance, TCS, Infosys, HDFC Bank,
  ICICI Bank, ITC, L&T, Bharti Airtel, Maruti, Sun Pharma.
- Annual reports for FY2025 and FY2026 (20 PDFs, roughly 6,000-8,000 pages).
- Source: BSE's annual report archive (public PDFs). Downloads run on Bille's
  computer because the cloud sandbox cannot reach BSE or NSE.

## Steps (results reported after each, like the volatility project)
1. Download + parse PDFs to page-level text (company, FY, page). Report page
   counts and how many pages are tables/scans with no text.
2. Chunking + three retrievers: BM25, dense (local bge embeddings), hybrid
   (reciprocal rank fusion) + cross-encoder reranker.
3. Eval set: ~60 hand-checked questions with the gold page(s). Report hit-rate@k
   and MRR for BM25 vs dense vs hybrid vs hybrid+rerank. This is the main
   honest result for the CV.
4. LangGraph agent: route -> retrieve -> grade docs -> rewrite query or answer
   -> cite. Tools: search_filings, compare_companies, a calculator for ratios.
5. MCP tool server exposing the retrieval tools; human-approval interrupt
   before any expensive/multi-company tool run.
6. Answer quality: faithfulness (LLM judge, spot-checked by hand) and
   answer correctness on the eval set. LangSmith traces.
7. FastAPI endpoint + Dockerfile, README with setup, results and limitations.

## Models
- Embeddings and reranker: local and free (BAAI/bge-small-en-v1.5,
  BAAI/bge-reranker-base), run on CPU.
- LLM: one API model behind LangChain's init_chat_model, so swapping provider
  is one line. Same model for the faithfulness judge (limitation noted).
