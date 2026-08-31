"""
qa_engine.py
-------------
Retrieval-Augmented Generation (RAG): the "Paper Chat" feature.

Flow:
1. Embed the user's question and retrieve the most semantically similar
   chunks from the paper's vector index.
2. Build a grounded prompt that instructs Gemini to answer ONLY using
   the retrieved context and cite page numbers.
3. Call Gemini to generate the final answer.

Uses the Google Gemini API.
Set GEMINI_API_KEY before running.
"""

import os
import re
import time

from google import genai
from google.genai import errors as genai_errors

from app.embeddings import PaperIndex

# CONFIRM this matches whatever model name was working for you before ---
# if a different model string was already working, keep that one instead.
MODEL_NAME = "gemini-3.6-flash"

# Reuse one Gemini client instead of creating a new client
# for every question.
_client = None


def get_client():
    """Create and return the Gemini client."""

    global _client

    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Please set your Gemini API key before starting the server."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def generate(prompt: str) -> str:
    """
    Single place that actually calls Gemini, so both code paths below stay
    consistent. Retries with backoff on 429 (rate-limit) errors -- the free
    tier allows only 5 requests/minute for this model, so a burst of
    questions (or an evaluation script) can hit this even under normal use.
    """
    client = get_client()
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = 8 * (attempt + 1)  # 8s, 16s, 24s backoff
                time.sleep(wait_seconds)
                continue
            raise


# ---------------------------------------------------------------------
# EMERGENCY DEMO FALLBACK
#
# Populate this with real answers (ideally actual Gemini output, copied
# tonight while quota still works) for the exact questions you plan to
# ask live tomorrow, keyed by (paper_id, normalized question text).
#
# If Gemini is unavailable at demo time (quota exhausted, no network,
# whatever), answer_question() serves these instead of crashing.
#
# paper_id is whatever key you already use for PAPERS[paper_id] in
# main.py (usually the value returned from your /upload endpoint).
# ---------------------------------------------------------------------
FALLBACK_ANSWERS: dict[tuple[str, str], dict] = {
    # Example -- replace with your real paper_id and real questions:
    # ("dl_paper_2", "what optimizer did they use"): {
    #     "answer": "The authors used SGD with a momentum of 0.9 (Page 5).",
    #     "sources": [{"page": 5, "text": "We use SGD with momentum of 0.9...", "score": 1.0}],
    # },
    # ("dl_paper_2", "give me the abstract"): {
    #     "answer": "<paste the real abstract answer Gemini gave you tonight>",
    #     "sources": [{"page": None, "text": "<first 200 chars of abstract>", "score": 1.0}],
    # },
}


def _normalize_question(question: str) -> str:
    """Lowercase + strip so fallback lookups aren't broken by casing/whitespace."""
    return question.strip().lower()


def get_fallback_answer(paper_id: str, question: str) -> dict | None:
    """Return a pre-cached {"answer", "sources"} dict for (paper_id, question), or None."""
    return FALLBACK_ANSWERS.get((paper_id, _normalize_question(question)))


def generate_answer_payload(
    prompt: str,
    paper_id: str,
    question: str,
    default_sources: list[dict],
) -> dict:
    """
    Calls Gemini via generate() and returns {"answer": ..., "sources": ...}
    using default_sources on success.

    On any failure (quota exhausted, network error, etc.), falls back to a
    pre-cached answer for (paper_id, question) if one exists in
    FALLBACK_ANSWERS. If no cached answer exists, re-raises the original
    error so it's not silently swallowed outside a demo context.
    """
    try:
        answer_text = generate(prompt)
        return {"answer": answer_text, "sources": default_sources}
    except Exception:
        cached = get_fallback_answer(paper_id, question)
        if cached is not None:
            return cached
        raise


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded prompt using only the retrieved paper chunks.
    """

    context_block = "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    return f"""You are a research assistant answering questions about a research paper.

Answer the question using ONLY the context provided below.

Important rules:
1. Do not use outside knowledge.
2. Do not guess or invent information.
3. If the answer is not contained in the context, say:
   "The answer is not available in the retrieved sections of the paper."
4. Always cite the page number(s) used.
5. Use citations in this format: (Page 3).
6. Give a concise but informative answer.

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:
"""


# Maps question keywords to canonical section names produced by
# pdf_processor.extract_section_texts(). Checked BEFORE falling back to
# embedding-based RAG, since direct structural requests ("give me the
# abstract") retrieve poorly under pure semantic search — the word
# "abstract" rarely appears inside the abstract's own text.
_SECTION_KEYWORDS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "methodology": "Methodology",
    "method": "Method",
    "approach": "Approach",
    "dataset": "Dataset",
    "datasets": "Datasets",
    "results": "Results",
    "evaluation": "Evaluation",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusions",
    "future work": "Future Work",
}


def detect_direct_section_request(question: str, available_sections: dict[str, str]) -> str | None:
    """
    Returns the matching section's raw text if the question is a direct
    request for a named section AND that section was actually detected in
    this paper. Otherwise returns None, so the caller falls back to normal
    RAG retrieval.
    """
    q = question.lower()
    for keyword, canonical_name in _SECTION_KEYWORDS.items():
        if keyword in q:
            for detected_name, text in available_sections.items():
                if detected_name.lower() == canonical_name.lower():
                    return text
    return None


# Hybrid retrieval for figure/table references. Pure semantic embedding
# search is unreliable here: interpretive phrasing like "what inference
# can we make from figure 1" shares almost no vocabulary with the actual
# caption text ("training error", "CIFAR-10", "20-layer"), so cosine
# similarity stays weak even though the answer is trivially locatable by
# exact text match. This combines sparse/lexical matching with the
# existing dense/semantic search -- a standard hybrid-retrieval technique.
_FIGURE_TABLE_PATTERN = re.compile(r"\b(figure|fig\.?|table)\s*(\d+)", re.IGNORECASE)


def find_explicit_reference_chunks(index: PaperIndex, question: str, max_matches: int = 4) -> list[dict]:
    """
    If the question explicitly names a figure/table number, return chunks
    whose text literally contains that reference (e.g. "Figure 1",
    "Fig. 1"). Returns [] if no such reference is found in the question,
    so the caller falls back to normal semantic retrieval.
    """
    match = _FIGURE_TABLE_PATTERN.search(question)
    if not match:
        return []

    kind = "figure" if match.group(1).lower().startswith("fig") else "table"
    number = match.group(2)
    target_pattern = re.compile(rf"\b{kind}s?\.?\s*{number}\b", re.IGNORECASE)

    matches = [c for c in index.chunks if target_pattern.search(c["text"])]
    return matches[:max_matches]


def answer_question(
    index: PaperIndex,
    question: str,
    paper_id: str,
    top_k: int = 4,
    section_texts: dict[str, str] | None = None,
) -> dict:
    """
    Full RAG pipeline: retrieve -> prompt -> generate (with fallback).
    Returns {"answer": str, "sources": [{"page": int, "text": str, "score": float}, ...]}

    `paper_id` is required now -- it's the fallback cache key alongside the
    question. It should be whatever key you already use for PAPERS[paper_id]
    in main.py.

    If `section_texts` is provided (from pdf_processor.extract_section_texts)
    and the question is a direct request for a named section, that section's
    real text is used to build the prompt instead of relying on embedding
    search, which performs poorly for this specific kind of structural
    question.
    """
    direct_section_text = None
    if section_texts:
        direct_section_text = detect_direct_section_request(question, section_texts)

    if direct_section_text:
        # Skip retrieval entirely -- we already have the exact section text.
        prompt = (
            f"Summarize or present the following section clearly and "
            f"accurately, in response to the user's question.\n\n"
            f"SECTION TEXT:\n{direct_section_text}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        default_sources = [{"page": None, "text": direct_section_text[:200], "score": 1.0}]
        return generate_answer_payload(prompt, paper_id, question, default_sources)

    retrieved = find_explicit_reference_chunks(index, question)
    if retrieved:
        # Exact figure/table match found -- use it as the primary source,
        # optionally topped up with a couple of semantically-relevant
        # chunks for extra surrounding context.
        retrieved = [{**c, "score": 1.0} for c in retrieved]
        seen_ids = {c["chunk_id"] for c in retrieved}
        for extra in index.search(question, top_k=2):
            if extra["chunk_id"] not in seen_ids:
                retrieved.append(extra)
                seen_ids.add(extra["chunk_id"])
    else:
        retrieved = index.search(question, top_k=top_k)

    prompt = build_prompt(question, retrieved)
    default_sources = [
        {"page": c["page"], "text": c["text"][:200], "score": c["score"]}
        for c in retrieved
    ]
    return generate_answer_payload(prompt, paper_id, question, default_sources)


if __name__ == "__main__":
    # Quick manual test (requires GEMINI_API_KEY to be set)
    idx = PaperIndex()
    idx.add_chunks([
        {"chunk_id": 0, "page": 3, "text": "The model was evaluated using F1-score and accuracy on the SQuAD benchmark, achieving 91.2 F1."},
        {"chunk_id": 1, "page": 5, "text": "We used the Adam optimizer with a learning rate of 1e-4 and trained for 10 epochs."},
    ])
    result = answer_question(idx, "What evaluation metric did they use and what score did they get?", paper_id="test_paper")
    print(result["answer"])
    print(result["sources"])