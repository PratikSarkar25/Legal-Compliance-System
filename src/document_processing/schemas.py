from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from pathlib import Path


class DocumentMetadata(BaseModel):
    """Metadata extracted from the PDF document."""

    num_pages: int = Field(
        ...,
        description="Total number of pages in the PDF."
    )

    title: Optional[str] = Field(
        default=None,
        description="Document title extracted from PDF metadata."
    )

    author: Optional[str] = Field(
        default=None,
        description="Author extracted from PDF metadata."
    )

    creation_date: Optional[str] = Field(
        default=None,
        description="Creation date stored in the PDF metadata."
    )

    modification_date: Optional[str] = Field(
        default=None,
        description="Last modification date stored in the PDF metadata."
    )

    file_size_bytes: int = Field(
        ...,
        description="Size of the PDF file in bytes."
    )

    language: Optional[str] = Field(
        default=None,
        description="Detected document language (e.g., English)."
    )

    is_scanned: bool = Field(
        default=False,
        description="True if the PDF appears to be scanned and may require OCR."
    )


class PageContent(BaseModel):
    """Represents the extracted content of a single page."""

    page_number: int = Field(
        ...,
        description="1-indexed page number."
    )

    text: str = Field(
        ...,
        description="Raw extracted text from the page."
    )

    cleaned_text: Optional[str] = Field(
        default=None,
        description="Cleaned version of the extracted text."
    )


class ProcessedDocument(BaseModel):
    """Output of the document processing pipeline."""

    filename: str = Field(
        ...,
        description="Name of the PDF file."
    )

    file_path: Path = Field(
        ...,
        description="Absolute or relative path of the PDF."
    )

    metadata: DocumentMetadata

    pages: List[PageContent]

    full_cleaned_text: Optional[str] = Field(
        default=None,
        description="Concatenated cleaned text from all pages."
    )

    extraction_method: str = Field(
        default="pymupdf",
        description="Method used to extract the document."
    )

    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the document was processed."
    )