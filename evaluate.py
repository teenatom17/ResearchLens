"""
evaluate.py
------------
A small evaluation harness for ResearchLens.

Runs a fixed benchmark of question -> expected-answer-keyword pairs
against the live API (http://127.0.0.1:8000), scores whether each
retrieved answer actually contains the expected content, and prints a
summary accuracy report.

This turns "it seems to work" into a real, reportable number for your
project write-up (e.g. "87% pass rate on a 15-question benchmark across
3 papers").

Usage:
    1. Make sure uvicorn is running (app.main:app) and GEMINI_API_KEY is set.
    2. Update PAPER_FILES below with the actual PDF filenames you want to test.
    3. Run: python evaluate.py
"""

import time
import requests

API_BASE = "http://127.0.0.1:8000"

# Map a short label -> local PDF path (must exist on disk, same as what
# you've been uploading through the frontend).
PAPER_FILES = {
    "resnet": "dl_paper_2.pdf",
    "sample": "sample_paper.pdf",
    "real": "real_paper.pdf",
    "bert": "Nlp_4.pdf",
}


# Each test case: (paper_label, question, [list of keywords/phrases that
# should appear in a CORRECT answer, case-insensitive]).
# For "should decline" cases, use expect_decline=True instead of keywords.
BENCHMARK = [
    # --- ResNet paper ---
    ("resnet", "what dataset did they use for the main ImageNet experiments?",
     ["imagenet"]),
    ("resnet", "what optimizer did they use?",
     ["sgd", "stochastic gradient descent", "stochastic gradient"]),
    ("resnet", "give me the abstract",
     ["residual", "degradation", "152"]),
    ("resnet", "what does figure 1 represent?",
     ["cifar", "training error", "56-layer"]),
    ("resnet", "what is the top-5 error of their ensemble on imagenet?",
     ["3.57"]),
    ("resnet", "what is the capital of france?",
     None, True),  # deliberately unanswerable -> should decline

    # --- Sample synthetic paper ---
    ("sample", "what evaluation metric did they use?",
     ["f1"]),
    ("sample", "give me the conclusion",
     ["sciqa-net"]),
    ("sample", "what are the limitations?",
     ["multi-hop", "mathematical notation"]),
    ("sample", "what dataset size did they use?",
     ["10,000", "500"]),

    # --- Real paper (fill in with actual expected content) ---
    # ("real", "what methodology did they use?", ["..."]),
    # ("real", "what are the main results?", ["..."]),
    # =========================
    # REAL PAPER
    # =========================

    ("real", "what datasets did they use?",
     ["squad", "yahoo"]),

    ("real", "what methodology did they use?",
     ["bilstm", "bert", "tf-idf"]),

    ("real", "what are the evaluation metrics?",
     ["p@3", "p@5", "ndcg"]),

    ("real", "what was the best performing approach?",
     ["weighted", "bert"]),

    ("real", "what was the P@3 score of Weighted TF-IDF plus BERT?",
     ["0.534"]),



    # --- BERT paper ---
    ("bert", "what does BERT stand for?",
     ["bidirectional", "encoder", "representations", "transformers"]),

    ("bert", "what are the two pre-training tasks used by BERT?",
     ["masked", "language", "next sentence"]),

    ("bert", "what architecture does BERT use?",
     ["transformer", "encoder"]),

    ("bert", "what is the difference between BERT and GPT?",
     ["bidirectional", "left-to-right"]),

    ("bert", "what datasets were used for BERT pre-training?",
     ["bookscorpus", "wikipedia"]),

    ("bert", "how large are BERTBASE and BERTLARGE?",
     ["110M", "340M"]),

    ("bert", "how many layers does BERTBASE have?",
     ["12"]),

    ("bert", "how many layers does BERTLARGE have?",
     ["24"]),

    ("bert", "what is the hidden size of BERTBASE?",
     ["768"]),

    ("bert", "how many attention heads does BERTBASE use?",
     ["12"]),

    ("bert", "what percentage of tokens are masked during pre-training?",
     ["15%"]),

    ("bert", "how does masked language modeling work?",
     ["mask", "predict"]),

    ("bert", "what is next sentence prediction?",
     ["IsNext", "NotNext"]),

    ("bert", "what is the purpose of the NSP task?",
     ["sentence", "relationship"]),

    ("bert", "what tokenizer or vocabulary does BERT use?",
     ["WordPiece", "30,000"]),

    ("bert", "what are the three components of BERT input representation?",
     ["token", "segment", "position"]),

    ("bert", "what is the purpose of the CLS token?",
     ["classification"]),

    ("bert", "what is the purpose of the SEP token?",
     ["separate"]),

    ("bert", "what datasets are used for BERT evaluation?",
     ["GLUE", "SQuAD", "SWAG"]),

    ("bert", "what is the SQuAD v1.1 dataset?",
     ["100k", "question", "answer"]),

    ("bert", "what evaluation metric is used for SQuAD?",
     ["F1", "EM"]),

    ("bert", "what was the SQuAD v1.1 test F1 score achieved by BERTLARGE?",
     ["93.2"]),

    ("bert", "what was the SQuAD v2.0 test F1 score achieved by BERTLARGE?",
     ["83.1"]),

    ("bert", "what GLUE score did BERT achieve?",
     ["80.5"]),

    ("bert", "what MultiNLI accuracy did BERT achieve?",
     ["86.7"]),

    ("bert", "what optimizer was used for training BERT?",
     ["Adam"]),

    ("bert", "what batch size was used during BERT pre-training?",
     ["256", "128,000"]),

    ("bert", "how many pre-training steps were used?",
     ["1M", "1,000,000"]),

    ("bert", "how does BERT perform compared with OpenAI GPT?",
     ["outperforms", "GPT"]),

    ("bert", "what is the main contribution of BERT?",
     ["bidirectional", "pre-training"]),

    # Deliberately unanswerable question
    ("bert", "what is the capital of France?",
     None, True),
]




def upload_paper(path: str) -> str:
    with open(path, "rb") as f:
        resp = requests.post(f"{API_BASE}/upload", files={"file": f})
    resp.raise_for_status()
    return resp.json()["paper_id"]


def ask(paper_id: str, question: str) -> dict:
    resp = requests.post(f"{API_BASE}/ask", json={"paper_id": paper_id, "question": question})
    resp.raise_for_status()
    return resp.json()


def run_benchmark():
    print("=" * 70)
    print("Uploading papers...")
    print("=" * 70)
    paper_ids = {}
    for label, path in PAPER_FILES.items():
        try:
            pid = upload_paper(path)
            paper_ids[label] = pid
            print(f"  {label:10s} -> paper_id={pid}  ({path})")
        except Exception as e:
            print(f"  {label:10s} -> FAILED to upload {path}: {e}")

    print()
    print("=" * 70)
    print("Running benchmark questions...")
    print("=" * 70)

    results = []
    for entry in BENCHMARK:
        if len(entry) == 4:
            label, question, keywords, expect_decline = entry
        else:
            label, question, keywords = entry
            expect_decline = False

        if label not in paper_ids:
            print(f"[SKIP] {label} not uploaded, skipping: {question!r}")
            continue

        try:
            result = ask(paper_ids[label], question)
        except Exception as e:
            print(f"[ERROR] {label} | {question!r} -> {e}")
            results.append({"label": label, "question": question, "pass": False, "reason": "request failed"})
            continue

        answer = result.get("answer", "")
        answer_lower = answer.lower()

        if expect_decline:
            decline_phrases = ["not available", "cannot find", "not contained", "no relevant"]
            passed = any(p in answer_lower for p in decline_phrases)
            reason = "correctly declined" if passed else "did NOT decline (possible hallucination)"
        else:
            passed = any(kw.lower() in answer_lower for kw in keywords)
            missing = [kw for kw in keywords if kw.lower() not in answer_lower]
            reason = "keyword(s) found" if passed else f"missing keywords: {missing}"

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label:8s} | {question}")
        print(f"        -> {reason}")
        print(f"        -> answer: {answer[:150].strip()}...")
        print()

        results.append({"label": label, "question": question, "pass": passed, "reason": reason})
        time.sleep(13)  # free-tier Gemini limit is 5 requests/minute -- stay well under it

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    if total == 0:
        print("  No tests were run (papers failed to upload — is uvicorn running?).")
        return results
    print(f"  {passed}/{total} passed ({100*passed/total:.1f}%)")
    print()
    failures = [r for r in results if not r["pass"]]
    if failures:
        print("  Failed cases:")
        for r in failures:
            print(f"    - [{r['label']}] {r['question']}  ({r['reason']})")
    else:
        print("  All test cases passed!")

    return results


if __name__ == "__main__":
    run_benchmark()