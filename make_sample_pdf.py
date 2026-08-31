from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("sample_paper.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

content = [
    ("Title", "Efficient Transformer-Based Question Answering for Scientific Documents"),
    ("Heading2", "Abstract"),
    ("BodyText", "In this paper we propose a novel transformer-based architecture for question "
                 "answering over scientific documents. Our method, called SciQA-Net, combines "
                 "a retrieval module with a fine-tuned language model. We evaluate our approach "
                 "on a custom dataset of 10,000 research paper excerpts and achieve an F1-score "
                 "of 87.4, outperforming baseline BERT models by 5.2 points."),
    ("Heading2", "Introduction"),
    ("BodyText", "Research papers are dense, technical documents that are time-consuming to read. "
                 "Automated question answering systems can help researchers quickly extract "
                 "relevant information. Prior work by Devlin et al. introduced BERT, which "
                 "revolutionized natural language understanding tasks. However, applying BERT "
                 "directly to long scientific documents remains challenging due to context length "
                 "limitations."),
    ("Heading2", "Methodology"),
    ("BodyText", "SciQA-Net consists of two components: a dense retriever built on Sentence-BERT "
                 "embeddings, and a generator based on a fine-tuned T5 model. Given a question, the "
                 "retriever first identifies the top-5 most relevant passages using cosine "
                 "similarity over sentence embeddings. These passages are then concatenated and "
                 "fed to the generator, which produces a natural language answer grounded in the "
                 "retrieved context."),
    ("Heading2", "Dataset"),
    ("BodyText", "We constructed a dataset of 10,000 question-answer pairs derived from 500 "
                 "computer science research papers published between 2018 and 2023. Questions were "
                 "generated using a combination of automated templates and manual annotation by "
                 "graduate student volunteers."),
    ("Heading2", "Results"),
    ("BodyText", "Our model achieves an F1-score of 87.4 and an exact-match score of 79.1 on the "
                 "held-out test set. This represents a 5.2 point improvement over the BERT-large "
                 "baseline, and a 3.1 point improvement over RoBERTa. We also report a mean answer "
                 "latency of 340 milliseconds per query, which is fast enough for interactive use."),
    ("Heading2", "Limitations"),
    ("BodyText", "Our approach struggles with questions requiring multi-hop reasoning across "
                 "distant sections of a paper, and with mathematical notation, which is often "
                 "poorly represented in the extracted text. Future work should address both of "
                 "these limitations, potentially through structured document parsing."),
    ("Heading2", "Conclusion"),
    ("BodyText", "We presented SciQA-Net, a retrieval-augmented question answering system for "
                 "scientific documents, and demonstrated strong performance improvements over "
                 "existing baselines. We release our dataset and code to encourage further "
                 "research in this direction."),
]

for style_name, text in content:
    story.append(Paragraph(text, styles[style_name]))
    story.append(Spacer(1, 12))

doc.build(story)
print("sample_paper.pdf created")
