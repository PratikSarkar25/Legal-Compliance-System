"""
Custom exceptions for the document processing module.
"""


class DocumentProcessingError(Exception):
    """Base exception for all document processing operations."""
    ...


class PDFParsingError(DocumentProcessingError):
    """Raised when PDF parsing or file streaming fails."""
    ...


class MetadataExtractionError(DocumentProcessingError):
    """Raised when metadata extraction fails or returns corrupted headers."""
    ...


class TextCleaningError(DocumentProcessingError):
    """Raised when text cleaning or regex normalization fails."""
    ...