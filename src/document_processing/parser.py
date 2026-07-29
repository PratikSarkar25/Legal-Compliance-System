from pathlib import Path
from typing import List

import fitz

from src.document_processing.exceptions import PDFParsingError
from src.document_processing.schemas import PageContent


class PDFParser:
    """Extract text from PDF files using PyMuPDF."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {self.file_path}")

    def extract_pages(self) -> List[PageContent]:
        """
        Extract text from every page of the PDF.

        Returns:
            List[PageContent]
        """

        pages: List[PageContent] = []

        try:
            with fitz.open(str(self.file_path)) as doc:

                for page_number, page in enumerate(doc, start=1):

                    text = page.get_text("text").strip()

                    pages.append(
                        PageContent(
                            page_number=page_number,
                            text=text,
                        )
                    )

        except Exception as exc:
            raise PDFParsingError(
                f"Unable to parse '{self.file_path}': {exc}"
            ) from exc

        return pages