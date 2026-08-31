"""
nlp_preprocessing.py
---------------------
Core classical-NLP stage. This is where the explicit "NLP techniques"
your academic report should talk about live:

1. Sentence segmentation      (spaCy sentencizer / dependency parser)
2. Tokenization                (spaCy tokenizer)
3. Stopword removal & lemmatization (spaCy)
4. Part-of-speech tagging      (spaCy tagger)
5. Named Entity Recognition (NER) -> used to pull candidate
   authors / organizations / dates for metadata
6. Keyword extraction via TF-IDF (scikit-learn)

Downstream stages (chunking, embeddings) build on top of these
outputs rather than working on raw text directly.
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

_nlp = spacy.load("en_core_web_sm")


def segment_sentences(text: str) -> list[str]:
    """Sentence segmentation using spaCy's dependency-parse-based sentencizer."""
    doc = _nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Tokenization + lemmatization (+ optional stopword removal).
    e.g. "The models were trained using datasets" ->
         ["model", "train", "datasets"]  (stopwords removed, lemmatized)
    """
    doc = _nlp(text)
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if remove_stopwords and tok.is_stop:
            continue
        tokens.append(tok.lemma_.lower())
    return tokens


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Named Entity Recognition (NER).
    Used to pull candidate metadata: PERSON (authors), ORG (institutions),
    DATE (publication year), GPE (locations), etc.
    """
    doc = _nlp(text)
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities


def extract_keywords_tfidf(documents: list[str], top_k: int = 10) -> list[str]:
    """
    Keyword extraction using TF-IDF over a set of chunks/sections.
    `documents` is typically the list of section texts or chunk texts
    for a single paper — TF-IDF scores terms that are locally frequent
    but globally rare across the chunks, which surfaces paper-specific
    keywords (model names, dataset names, method names, etc.).
    """
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams, e.g. "neural network"
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    scores = tfidf_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[:top_k]]


def detect_section_heading(line: str) -> str | None:
    """
    Lightweight heuristic section classifier (Phase-1 substitute for a
    trained classifier). Matches common IMRAD-style headings.
    A trained ML classifier (e.g. logistic regression / small transformer
    over heading text + font features) is a natural extension of this.
    """
    known_sections = [
        "abstract", "introduction", "related work", "background",
        "methodology", "method", "approach", "dataset", "datasets",
        "experiments", "experimental setup", "results", "evaluation",
        "discussion", "limitations", "conclusion", "conclusions",
        "future work", "references",
    ]
    normalized = line.strip().lower().strip(".:")
    for sec in known_sections:
        if normalized == sec or (len(normalized) < 40 and normalized.startswith(sec)):
            return sec.title()
    return None


if __name__ == "__main__":
    sample = (
        "Dr. Alice Chen and Bob Smith from Stanford University proposed a new "
        "transformer-based model in 2023. The model was evaluated on the "
        "SQuAD dataset and achieved state-of-the-art accuracy."
    )
    print("Sentences:", segment_sentences(sample))
    print("Tokens/Lemmas:", tokenize_and_lemmatize(sample))
    print("Entities:", extract_entities(sample))
