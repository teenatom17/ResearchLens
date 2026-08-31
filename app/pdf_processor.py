"""
pdf_processor.py
-----------------
Stage 1 of the pipeline: PDF & Document Processing.

Responsibilities:
- Extract raw text from a PDF, page by page (using PyMuPDF).
- Clean the text: strip headers/footers/page numbers, fix broken
  line-wraps and hyphenation that PDFs commonly introduce.

This stage does NOT do any NLP yet — it just produces clean,
page-tagged plain text that the NLP stage can work on.
"""

import re
import pymupdf  # PyMuPDF

from app.nlp_preprocessing import extract_entities


def _extract_text_blocks(page) -> list[dict]:
    """Collect plain-text blocks with their bounding boxes from a page."""
    page_dict = page.get_text("dict")
    text_blocks = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        line_texts = []
        for line in block.get("lines", []):
            spans = [span["text"].strip() for span in line.get("spans", []) if span["text"].strip()]
            if spans:
                line_texts.append(" ".join(spans))

        text = "\n".join(line_texts).strip()
        if not text:
            continue

        text_blocks.append(
            {
                "bbox": tuple(block["bbox"]),
                "text": text,
            }
        )

    return text_blocks


def _has_two_column_layout(blocks: list[dict], page_width: float) -> bool:
    """Heuristic: detect a clear left/right text split on the page."""
    if len(blocks) < 4:
        return False

    mid_x = page_width / 2
    left_blocks = []
    right_blocks = []

    for block in blocks:
        x0, _, x1, _ = block["bbox"]
        center_x = (x0 + x1) / 2
        if center_x < mid_x:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    if len(left_blocks) < 2 or len(right_blocks) < 2:
        return False

    if min(len(left_blocks), len(right_blocks)) / len(blocks) < 0.2:
        return False

    left_edge = max(block["bbox"][2] for block in left_blocks)
    right_edge = min(block["bbox"][0] for block in right_blocks)
    gap = right_edge - left_edge

    return gap >= page_width * 0.02


def _reconstruct_page_text(page) -> str:
    """Rebuild page text in reading order for single- and two-column layouts."""
    blocks = _extract_text_blocks(page)
    if not blocks:
        return ""

    if _has_two_column_layout(blocks, page.rect.width):
        mid_x = page.rect.width / 2
        left_blocks = [block for block in blocks if (block["bbox"][0] + block["bbox"][2]) / 2 < mid_x]
        right_blocks = [block for block in blocks if (block["bbox"][0] + block["bbox"][2]) / 2 >= mid_x]
        ordered_blocks = sorted(left_blocks, key=lambda block: (block["bbox"][1], block["bbox"][0]))
        ordered_blocks.extend(sorted(right_blocks, key=lambda block: (block["bbox"][1], block["bbox"][0])))
    else:
        ordered_blocks = sorted(blocks, key=lambda block: (block["bbox"][1], block["bbox"][0]))

    return "\n\n".join(block["text"] for block in ordered_blocks)


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = _reconstruct_page_text(page)
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Clean raw PDF text extraction artifacts:
    - Rejoin hyphenated line-break words: "informa-\ntion" -> "information"
    - Collapse single line breaks (mid-sentence wraps) into spaces,
      but preserve paragraph breaks (double newlines).
    - Strip standalone page-number lines and excessive whitespace.
    """
    # Rejoin hyphenated words broken across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove lines that are just numbers (likely page numbers)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)

    # Preserve paragraph breaks, collapse single newlines to spaces
    text = re.sub(r"\n{2,}", "<PARA>", text)
    text = text.replace("\n", " ")
    text = text.replace("<PARA>", "\n\n")

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Full Stage-1 pipeline: extract + clean, page by page.
    Returns [{"page": 1, "text": "cleaned text..."}, ...]
    """
    pages = extract_pages(pdf_path)
    for p in pages:
        p["text"] = clean_text(p["text"])
    return pages


def _collect_page_lines(page) -> tuple[list[dict], float | None]:
    """Extract readable line records and estimate body text size for one page."""
    page_dict = page.get_text("dict")
    sizes = []
    lines = []

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            span_records = []
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                span_records.append(
                    {
                        "text": text,
                        "size": span["size"],
                        "flags": span["flags"],
                        "bbox": span["bbox"],
                    }
                )
                sizes.append(span["size"])

            if not span_records:
                continue

            text = " ".join(span["text"] for span in span_records).strip()
            bbox = (
                min(span["bbox"][0] for span in span_records),
                min(span["bbox"][1] for span in span_records),
                max(span["bbox"][2] for span in span_records),
                max(span["bbox"][3] for span in span_records),
            )
            lines.append(
                {
                    "text": text,
                    "max_size": max(span["size"] for span in span_records),
                    "min_size": min(span["size"] for span in span_records),
                    "bbox": bbox,
                }
            )

    lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    body_size = max(set(sizes), key=sizes.count) if sizes else None
    return lines, body_size


def extract_title_page_metadata(pdf_path: str) -> dict:
    """
    Extract title + author candidates from the first page only.

    Title heuristic:
    - take the largest-font line(s) near the top of page 1

    Author heuristic:
    - read lines immediately below the title
    - stop at the Abstract heading or when body-sized text begins
    - run NER over that byline region and keep PERSON entities
    """
    doc = pymupdf.open(pdf_path)
    try:
        if len(doc) == 0:
            return {"title": None, "authors": []}

        page = doc[0]
        lines, body_size = _collect_page_lines(page)
        if not lines or body_size is None:
            return {"title": None, "authors": []}

        largest_size = max(line["max_size"] for line in lines)
        title_lines = [line for line in lines if line["max_size"] >= largest_size - 0.1]
        if not title_lines:
            return {"title": None, "authors": []}

        first_title_y = min(line["bbox"][1] for line in title_lines)
        title_lines = [
            line for line in lines
            if line["max_size"] >= largest_size - 0.1 and abs(line["bbox"][1] - first_title_y) <= 40
        ]
        title_lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        title = " ".join(line["text"] for line in title_lines).strip() or None
        title_bottom = max(line["bbox"][3] for line in title_lines)

        byline_lines = []
        for line in lines:
            if line["bbox"][1] <= title_bottom:
                continue

            normalized = line["text"].lower().strip(".: ")
            if normalized == "abstract" or normalized.startswith("abstract "):
                break

            if line["max_size"] <= body_size + 0.25:
                break

            byline_lines.append(line["text"])

        byline_text = "\n".join(byline_lines).strip()
        entities = extract_entities(byline_text) if byline_text else {}
        authors = entities.get("PERSON", [])

        return {"title": title, "authors": authors}
    finally:
        doc.close()


KNOWN_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "methodology", "method", "approach", "dataset", "datasets",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "limitations", "conclusion", "conclusions",
    "future work", "references", "acknowledgements", "acknowledgments",
]


def extract_headings_by_font(pdf_path: str) -> list[dict]:
    """
    Detects section headings using font-size/boldness cues rather than
    plain-text line matching. Body text in most papers sits around
    9-11pt; headings are typically larger and/or bold. We flag any
    span whose size is meaningfully above the page's most common
    (body) font size, and whose text matches a known IMRAD-style
    section name.

    This is more robust than matching text alone, since it survives
    cases where PDF extraction merges a heading and the following
    paragraph onto the same logical line.

    Returns: [{"section": "Introduction", "page": 2}, ...]
    """
    doc = pymupdf.open(pdf_path)
    detected = []

    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        sizes = []
        spans = []
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        sizes.append(span["size"])
                        spans.append((span["size"], span["flags"], text))

        if not sizes:
            continue

        # Most common size on the page ~= body text size
        body_size = max(set(sizes), key=sizes.count)

        for size, flags, text in spans:
            is_bold = bool(flags & 2**4)  # PyMuPDF bold flag bit
            larger_than_body = size > body_size + 0.5
            normalized = text.lower().strip(".: ")
            if (larger_than_body or is_bold) and normalized in KNOWN_SECTIONS:
                detected.append({"section": normalized.title(), "page": page_num})

    doc.close()
    return detected


def extract_section_texts(pdf_path: str) -> dict[str, str]:
    """
    Structural section lookup: returns the actual body text belonging to
    each detected section (Abstract, Introduction, Methodology, etc.),
    not just the heading location.

    This exists because pure semantic/embedding retrieval is unreliable
    for direct structural requests like "give me the abstract" — the word
    "abstract" rarely appears inside the abstract's own text, so it scores
    poorly against embedding search even though the answer is trivially
    locatable by structure. This function gives an exact, deterministic
    alternative for that specific case.

    Returns: {"Abstract": "full text...", "Introduction": "full text...", ...}
    """
    doc = pymupdf.open(pdf_path)
    sections: dict[str, list[str]] = {}
    current_section = None

    for page in doc:
        page_dict = page.get_text("dict")
        sizes = []
        spans = []
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                line_text_parts = []
                line_size = None
                line_flags = None
                for span in line["spans"]:
                    text = span["text"]
                    if text.strip():
                        line_text_parts.append(text)
                        line_size = span["size"]
                        line_flags = span["flags"]
                if line_text_parts:
                    spans.append((line_size, line_flags, " ".join(line_text_parts).strip()))
                    sizes.append(line_size)

        if not sizes:
            continue
        body_size = max(set(sizes), key=sizes.count)

        for size, flags, text in spans:
            is_bold = bool(flags and flags & 2**4)
            larger_than_body = size is not None and size > body_size + 0.5
            normalized = text.lower().strip(".: ")

            if (larger_than_body or is_bold) and normalized in KNOWN_SECTIONS:
                current_section = normalized.title()
                sections.setdefault(current_section, [])
                continue

            if current_section:
                sections[current_section].append(text)

    doc.close()
    return {name: clean_text(" ".join(parts)) for name, parts in sections.items()}


if __name__ == "__main__":
    import sys
    result = process_pdf(sys.argv[1])
    for p in result[:2]:
        print(f"--- Page {p['page']} ---")
        print(p["text"][:500])