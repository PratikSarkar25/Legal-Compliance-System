from pathlib import Path
from typing import Optional

import fitz

#from src.document_processing.parser import PDFParsingError
from src.document_processing.exceptions import MetadataExtractionError
from src.document_processing.schemas import DocumentMetadata


class PDFMetadataExtractor:
    """Extract metadata from a PDF using PyMuPDF."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def extract(self) -> DocumentMetadata:
        """Extract document-level metadata."""

        file_size = self.file_path.stat().st_size

        try:
            with fitz.open(str(self.file_path)) as doc:

                metadata = doc.metadata or {}

                return DocumentMetadata(
                    num_pages=len(doc),
                    title=self._clean(metadata.get("title")),
                    author=self._clean(metadata.get("author")),
                    creation_date=self._clean(metadata.get("creationDate")),
                    modification_date=self._clean(metadata.get("modDate")),
                    file_size_bytes=file_size,
                )

        except Exception as exc:
            raise MetadataExtractionError(
                f"Failed to extract metadata from '{self.file_path}': {exc}"
            ) from exc

    @staticmethod
    def _clean(value: Optional[str]) -> Optional[str]:
        """Convert empty metadata values to None."""

        if value is None:
            return None

        value = value.strip()

        return value if value else None