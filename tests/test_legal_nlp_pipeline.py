"""
Integration tests for the complete Legal NLP pipeline.
"""

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

    document_pipeline = DocumentProcessingPipeline()

    processed_document = document_pipeline.process(
        sample_pdf
    )

    assert processed_document is not None

    # -----------------------------------------
    # Legal NLP
    # -----------------------------------------

    legal_pipeline = LegalNLPPipeline()

    legal_document = legal_pipeline.process(
        processed_document
    )

    assert legal_document is not None

    # -----------------------------------------
    # Output validation
    # -----------------------------------------

    assert legal_document.document_id is not None

    assert isinstance(
        legal_document.clauses,
        list,
    )

    assert isinstance(
        legal_document.entities,
        list,
    )

    assert isinstance(
        legal_document.relations,
        list,
    )

    assert isinstance(
        legal_document.embeddings,
        list,
    )