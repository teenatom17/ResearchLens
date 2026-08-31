"""
embeddings.py
--------------
Knowledge Layer (vector part): turns text chunks into dense semantic
vectors using a pretrained Sentence-BERT model, indexes them with FAISS
for fast approximate nearest-neighbour search, and exposes a
`semantic_search` function used by the RAG QA engine.

NLP technique: sentence embeddings + cosine similarity (semantic search),
as opposed to keyword/lexical search — this is what lets a question like
"how did they evaluate the model?" retrieve a passage that says
"we measured performance using F1-score" even with no literal word overlap.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, strong general-purpose embedding model
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class PaperIndex:
    """
    Wraps a FAISS index + the chunk metadata for one paper (or a whole
    library, if chunks from multiple papers are added with a paper_id).
    """

    def __init__(self):
        self.model = get_model()
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # inner product on normalized vectors = cosine similarity
        self.chunks: list[dict] = []  # parallel list: metadata for each vector

    def add_chunks(self, chunks: list[dict], paper_id: str = "paper_1"):
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.index.add(np.array(vectors, dtype="float32"))
        for c in chunks:
            meta = dict(c)
            meta["paper_id"] = paper_id
            self.chunks.append(meta)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vec, dtype="float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results


if __name__ == "__main__":
    idx = PaperIndex()
    idx.add_chunks([
        {"chunk_id": 0, "page": 1, "text": "The model was evaluated using F1-score and accuracy on the SQuAD benchmark."},
        {"chunk_id": 1, "page": 2, "text": "We used the Adam optimizer with a learning rate of 1e-4."},
        {"chunk_id": 2, "page": 3, "text": "Related work includes BERT and RoBERTa for question answering."},
    ])
    results = idx.search("how was the model's performance measured?", top_k=2)
    for r in results:
        print(f"[score={r['score']:.3f}, page={r['page']}] {r['text']}")
