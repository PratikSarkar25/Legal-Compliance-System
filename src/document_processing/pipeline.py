"""
Document processing pipeline.

This module orchestrates the complete document processing workflow:

PDF
 │
 ▼
Parser
 │
 ▼
Metadata Extraction
 │
 ▼
Text Cleaning
 │
 ▼
ProcessedDocument
"""

from pathlib import Path

from src.document_processing.cleaner import TextCleaner
from src.document_processing.metadata import PDFMetadataExtractor
from src.document_processing.parser import PDFParser
from src.document_processing.schemas import ProcessedDocument


class DocumentProcessingPipeline:
    """Complete PDF document processing pipeline."""

    def __init__(self, file_path: str | Path):

        self.file_path = Path(file_path)

        self.parser = PDFParser(self.file_path)
        self.metadata_extractor = PDFMetadataExtractor(self.file_path)
        self.cleaner = TextCleaner()

    def process(self) -> ProcessedDocument:
        """
        Execute the complete document processing pipeline.

        Returns:
            ProcessedDocument
        """

        # Step 1: Extract metadata
        metadata = self.metadata_extractor.extract()

        # Step 2: Extract page contents
        pages = self.parser.extract_pages()

        # Step 3: Clean each page
        cleaned_pages = []

        #for page in pages:

        #   page.cleaned_text = self.cleaner.clean_page_text(page.text)

        #    cleaned_pages.append(page)

        for i, page in enumerate(pages):

            print(f"\nPage {i+1}")
            print("Raw text length:", len(page.text))
            print("First 100 chars:", repr(page.text[:100]))

            page.cleaned_text = self.cleaner.clean_page_text(page.text)

            print("Cleaned length:", len(page.cleaned_text))

            cleaned_pages.append(page)

        # Step 4: Concatenate cleaned document
        full_text = "\n\n".join(
            page.cleaned_text
            for page in cleaned_pages
            if page.cleaned_text
        )

        # Step 5: Return structured document
        return ProcessedDocument(
            filename=self.file_path.name,
            file_path=self.file_path,
            metadata=metadata,
            pages=cleaned_pages,
            full_cleaned_text=full_text,
            extraction_method="pymupdf",
        )