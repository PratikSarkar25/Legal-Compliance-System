from pathlib import Path

from src.document_processing.pipeline import DocumentProcessingPipeline
from src.legal_nlp.pipeline import LegalNLPPipeline


def test_legal_nlp_pipeline():

    # -----------------------------------------
    # Sample Contract
    # -----------------------------------------

    sample_pdf = Path(
        "data/sample_contracts/sample.pdf"
    )

    assert sample_pdf.exists()

    # -----------------------------------------
    # Document Processing
    # -----------------------------------------

    document_pipeline = DocumentProcessingPipeline(
        file_path=sample_pdf
    )

    document = document_pipeline.process()

    assert document is not None
    assert document.filename == sample_pdf.name
    assert len(document.pages) > 0

    # -----------------------------------------
    # Legal NLP Processing
    # -----------------------------------------

    legal_nlp_pipeline = LegalNLPPipeline()

    result = legal_nlp_pipeline.process(document)

    # -----------------------------------------
    # Validate Final Result
    # -----------------------------------------

    assert result is not None

    assert hasattr(result, "clauses")
    assert hasattr(result, "entities")
    assert hasattr(result, "relations")
    assert hasattr(result, "embeddings")