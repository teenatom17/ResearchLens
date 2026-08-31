import json
import sys
import traceback
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_NAME = "sample_paper.pdf"
DEFAULT_QUESTIONS = [
    "What problem does this paper address, and what is the main proposed approach?",
    "What dataset or experimental setup did they use?",
    "What limitations, challenges, or future work does the paper mention?",
]


def print_header(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def print_json(label: str, value) -> None:
    print(f"{label}:")
    print(json.dumps(value, indent=2, ensure_ascii=True))


def resolve_pdf_path() -> Path:
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate.resolve()
    return (PROJECT_ROOT / DEFAULT_PDF_NAME).resolve()


def upload_paper(pdf_path: Path) -> dict:
    print_header("Stage 1 - Upload PDF")
    print(f"Uploading file: {pdf_path}")

    with pdf_path.open("rb") as pdf_file:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": (pdf_path.name, pdf_file, "application/pdf")},
            timeout=120,
        )

    print(f"HTTP {response.status_code}")
    response.raise_for_status()
    data = response.json()

    print(f"paper_id: {data.get('paper_id')}")
    print(f"num_pages: {data.get('num_pages')}")
    print(f"num_chunks: {data.get('num_chunks')}")
    print_json("keywords", data.get("keywords"))
    print_json("detected_sections", data.get("detected_sections"))

    return data


def ask_question(paper_id: str, question: str, question_number: int) -> dict:
    print_header(f"Stage 2 - Ask Question {question_number}")
    print(f"question: {question}")
    payload = {"paper_id": paper_id, "question": question}
    print_json("request", payload)

    response = requests.post(f"{BASE_URL}/ask", json=payload, timeout=120)
    print(f"HTTP {response.status_code}")
    response.raise_for_status()
    data = response.json()

    print(f"answer: {data.get('answer')}")
    print_json("sources", data.get("sources"))

    return data


def main() -> int:
    pdf_path = resolve_pdf_path()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    upload_data = upload_paper(pdf_path)
    for idx, question in enumerate(DEFAULT_QUESTIONS, start=1):
        ask_question(upload_data["paper_id"], question, idx)

    print_header("Done")
    print("API test completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_header("Error")
        print(f"{type(exc).__name__}: {exc}")

        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            print(f"Response status: {exc.response.status_code}")
            print("Response body:")
            print(exc.response.text)

        print("\nFull traceback:")
        traceback.print_exc()
        raise SystemExit(1)
