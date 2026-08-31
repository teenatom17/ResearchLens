"""
main.py
--------
FastAPI backend for ResearchLens (Phase-1 demo scope):

    Upload PDF -> extract & clean (pdf_processor)
               -> chunk (chunking, using NLP sentence segmentation)
               -> embed & index (embeddings)
    Ask question -> retrieve + generate grounded answer (qa_engine)

Also exposes lightweight NLP-analysis endpoints (keywords, entities,
detected section headings) so you can visibly demonstrate the classical
NLP stage separately from the RAG/LLM stage — useful for a viva where
you need to show "here is the NLP part, independent of the LLM."

Run:
    uvicorn app.main:app --reload
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pdf_processor import (
    process_pdf,
    extract_headings_by_font,
    extract_title_page_metadata,
    extract_section_texts,
)
from app.chunking import chunk_pages
from app.embeddings import PaperIndex
from app.nlp_preprocessing import (
    extract_entities,
    extract_keywords_tfidf,
    detect_section_heading,
)
from app.qa_engine import answer_question

app = FastAPI(title="ResearchLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_papers"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory store for the demo. Swap for PostgreSQL for a persistent library.
PAPERS: dict[str, dict] = {}


class QuestionRequest(BaseModel):
    paper_id: str
    question: str


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    paper_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{paper_id}.pdf")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Stage 1: PDF processing
    pages = process_pdf(save_path)
    title_metadata = extract_title_page_metadata(save_path)

    # Stage 2: NLP-aware chunking
    chunks = chunk_pages(pages)

    # Stage 3: lightweight NLP analysis (keywords + entities)
    chunk_texts = [c["text"] for c in chunks]
    keywords = extract_keywords_tfidf(chunk_texts, top_k=10) if chunk_texts else []
    entities = extract_entities(" ".join(chunk_texts[:5]))  # sample first few chunks

    detected_sections = extract_headings_by_font(save_path)

    # Structural section-text lookup, used by qa_engine to answer direct
    # "give me the abstract" style questions without relying on embedding
    # search (which performs poorly for that specific case).
    section_texts = extract_section_texts(save_path)

    # Stage 4: embed + index (Knowledge Layer)
    index = PaperIndex()
    index.add_chunks(chunks, paper_id=paper_id)

    PAPERS[paper_id] = {
        "filename": file.filename,
        "title": title_metadata["title"],
        "authors": title_metadata["authors"],
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "keywords": keywords,
        "entities": entities,
        "sections": detected_sections,
        "section_texts": section_texts,
        "index": index,
    }

    return {
        "paper_id": paper_id,
        "filename": file.filename,
        "title": title_metadata["title"],
        "authors": title_metadata["authors"],
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "keywords": keywords,
        "entities": entities,
        "detected_sections": detected_sections,
    }


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    paper = PAPERS.get(req.paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found. Upload it first via /upload.")

    result = answer_question(
        paper["index"],
        req.question,
        paper_id=req.paper_id,
        section_texts=paper.get("section_texts"),
    )
    return result

from app.qa_engine import get_client

@app.get("/status")
async def status():
    """Quick live-vs-fallback check -- hit this before/during the demo."""
    try:
        get_client().models.generate_content(
            model="gemini-3.6-flash",
            contents="ping",
        )
        return {"gemini": "live"}
    except Exception as e:
        return {"gemini": "unavailable", "detail": str(e)[:200]}


@app.get("/papers")
async def list_papers():
    return [
        {"paper_id": pid, "filename": p["filename"], "num_pages": p["num_pages"]}
        for pid, p in PAPERS.items()
    ]


@app.get("/")
async def root():
    return {"status": "ResearchLens API running", "endpoints": ["/upload", "/ask", "/papers"]}