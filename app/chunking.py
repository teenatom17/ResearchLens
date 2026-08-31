"""
chunking.py
------------
Splits cleaned, page-tagged text into meaningful chunks for embedding.

Uses sentence boundaries (from nlp_preprocessing.segment_sentences) rather
than raw character/token cuts, so chunks don't split mid-sentence — this
matters for embedding quality and for giving coherent context to the LLM
during RAG generation.

Each chunk keeps a reference to the page(s) it came from, so answers can
be grounded with a page citation later.
"""

from app.nlp_preprocessing import segment_sentences

# Rough token estimate: ~4 characters per token (English average)
CHARS_PER_TOKEN = 4


def chunk_pages(pages: list[dict], target_tokens: int = 300, overlap_sentences: int = 1) -> list[dict]:
    """
    pages: [{"page": 1, "text": "..."}, ...]  (from pdf_processor.process_pdf)

    Returns: [
        {"chunk_id": 0, "page": 1, "text": "...", "sentences": [...]},
        ...
    ]
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    chunks = []
    chunk_id = 0

    for page in pages:
        sentences = segment_sentences(page["text"])
        if not sentences:
            continue

        current: list[str] = []
        current_len = 0

        for sent in sentences:
            current.append(sent)
            current_len += len(sent)

            if current_len >= target_chars:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": " ".join(current),
                })
                chunk_id += 1
                # keep last N sentences for overlap/context continuity
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_len = sum(len(s) for s in current)

        if current:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": " ".join(current),
            })
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    from app.pdf_processor import process_pdf
    import sys
    pages = process_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunks from {len(pages)} pages")
    for c in chunks[:2]:
        print(f"--- Chunk {c['chunk_id']} (page {c['page']}) ---")
        print(c["text"][:300])
