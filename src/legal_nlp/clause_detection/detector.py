"""
CUAD Clause Detection Module.

This module performs legal clause extraction using the
CUAD fine-tuned RoBERTa Question Answering model.
"""

from typing import List
import torch
from transformers import pipeline

from src.document_processing.schemas import ProcessedDocument
from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.prompts.loader import load_cuad_questions
from src.legal_nlp.schemas import (
    ClauseSpan,
    DetectedClause,
)
from src.legal_nlp.exceptions import ClauseDetectionError
from src.legal_nlp.utils import chunk_text_with_stride


class ClauseDetector:
    """
    Detect legal clauses from a processed contract using
    the CUAD Question Answering model.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        self.config = config or LegalNLPConfig()
        self.model_loader = ModelLoader(self.config)
        self.tokenizer, self.model = self.model_loader.load_cuad_model()

        self.qa_pipeline = pipeline(
            task="question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.config.device == "cuda" else -1,
        )

        self.questions = load_cuad_questions()
        if not self.questions:
            raise ClauseDetectionError("No CUAD questions were loaded.")

    @torch.inference_mode()
    def detect(
        self,
        document: ProcessedDocument,
    ) -> List[DetectedClause]:
        """
        Detect legal clauses from the document.
        Parameters
        --------
        document: ProcessedDocument

        Returns
        -------
        List[DetectedClause]
            List of detected legal clause extracted using CUAD
            QUESTION ANSWERING model 
        """
        if document is None:
            raise ClauseDetectionError("Input document is None.")

        detected_clauses: List[DetectedClause] = []

        for page in document.pages:
            chunks = chunk_text_with_stride(
                text=page.cleaned_text or page.text,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                stride=self.config.stride,
            )

            for chunk in chunks:
                for question in self.questions:
                    result = self.qa_pipeline(
                        question=question["prompt"],
                        context=chunk["text"],
                    )

                    # Ignore weak predictions.
                    if result["score"] < self.config.detection_threshold:
                        continue

                    answer = result["answer"].strip()

                    # Ignore empty or meaningless answers.
                    if not answer:
                        continue

                    if answer.lower() in {"", "[cls]", "[sep]"}:
                        continue

                    span = ClauseSpan(
                        start_char=chunk["start_char"] + result["start"],
                        end_char=chunk["start_char"] + result["end"],
                        text=answer,
                        confidence=result["score"],
                    )

                    clause = DetectedClause(
                        clause_id=(
                            f"page_{page.page_number}_"
                            f"{question['clause_type']}_"
                            f"{len(detected_clauses) + 1}"
                        ),
                        category=question["display_name"],
                        text=answer,
                        confidence=result["score"],
                        page_number=page.page_number,
                        spans=[span],
                    )

                    detected_clauses.append(clause)

                    #print(
                    #    f"[{question['display_name']}] "
                    #    f"{result['score']:.3f} -> "
                    #    f"{result['answer']}"
                    #)

        # Deduplicate and sort clauses after scanning all pages and chunks
        detected_clauses = self._deduplicate_clauses(detected_clauses)
        detected_clauses = self._sort_clauses(detected_clauses)

        return detected_clauses

    def _deduplicate_clauses(
        self,
        clauses: List[DetectedClause],
    ) -> List[DetectedClause]:
        """
        Remove duplicate clause detections caused by overlapping chunks.

        If multiple detections have the same category and nearly identical
        text, keep the one with the highest confidence.
        """
        unique = {}

        for clause in clauses:
            key = (
                clause.category,
                clause.text.strip().lower(),
            )

            if key not in unique:
                unique[key] = clause
                continue

            if clause.confidence > unique[key].confidence:
                unique[key] = clause

        return list(unique.values())

    def _sort_clauses(
        self,
        clauses: List[DetectedClause],
    ) -> List[DetectedClause]:
        """
        Sort clauses by page number and character position.
        """
        return sorted(
            clauses,
            key=lambda clause: (
                clause.page_number,
                clause.spans[0].start_char if clause.spans else 0,
            ),
        )