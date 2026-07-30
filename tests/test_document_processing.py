from pathlib import Path

from src.document_processing.pipeline import DocumentProcessingPipeline


SAMPLE_PDF = Path("data/sample_contracts/sample.pdf")


def test_document_processing_pipeline():

    pipeline = DocumentProcessingPipeline(SAMPLE_PDF)

    document = pipeline.process()

    assert document.filename == SAMPLE_PDF.name

    assert len(document.pages) > 0

    #assert document.metadata.page_count == len(document.pages)
    
    assert document.metadata.num_pages == len(document.pages)

    assert len(document.full_cleaned_text) > 0

    assert document.extraction_method == "pymupdf"