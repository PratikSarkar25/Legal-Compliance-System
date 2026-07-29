"""
Text cleaning utilities for legal documents.

This module removes common PDF artifacts while preserving the
structure required for downstream Legal NLP tasks.
"""

import re
from typing import List, Optional
from src.document_processing.exceptions import TextCleaningError


class TextCleaner:
    """Clean and normalize extracted PDF text."""

    PAGE_NUMBER_PATTERNS = (
        re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),
        re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+\s*$", re.MULTILINE),
    )

    def __init__(
        self,
        custom_header_patterns: Optional[List[str]] = None,
        custom_footer_patterns: Optional[List[str]] = None,
    ) -> None:

        self.header_patterns = (
            [re.compile(p, re.IGNORECASE) for p in custom_header_patterns]
            if custom_header_patterns
            else []
        )

        self.footer_patterns = (
            [re.compile(p, re.IGNORECASE) for p in custom_footer_patterns]
            if custom_footer_patterns
            else []
        )

    def clean_page_text(self, text: str) -> str:
        """
        Clean the extracted text from a single PDF page.
        """

        if not text or not text.strip():
            return ""

        text = self._remove_headers(text)
        text = self._remove_page_numbers(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_whitespace(text)
        text = self._strip_lines(text)

        return text

    def clean_document(self, pages: List[str]) -> List[str]:
        """
        Clean every page in the document.
        """

        return [self.clean_page_text(page) for page in pages]

    def _remove_headers(self, text: str) -> str:
        """Remove custom document headers and footers."""

        cleaned = text

        for pattern in self.header_patterns:
            cleaned = pattern.sub("", cleaned)

        for pattern in self.footer_patterns:
            cleaned = pattern.sub("", cleaned)

        return cleaned

    def _remove_page_numbers(self, text: str) -> str:
        """Remove common page numbering patterns."""

        cleaned = text

        for pattern in self.PAGE_NUMBER_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        return cleaned

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """
        Join words split across line breaks.

        Example:
            agree-
            ment

        becomes:

            agreement
        """

        return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Normalize spacing while preserving paragraph structure.
        """

        text = re.sub(r"[^\S\r\n]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def _strip_lines(text: str) -> str:
        """
        Strip leading/trailing whitespace from every line.
        """

        lines = [line.strip() for line in text.splitlines()]

        return "\n".join(lines).strip()