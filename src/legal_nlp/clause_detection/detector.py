"""
CUAD Clause Detection Module.

This module performs legal clause extraction using the
CUAD fine-tuned RoBERTa Question Answering model.
"""

from typing import List, Dict, Any

import torch

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

        self.tokenizer, self.model = (
            self.model_loader.load_cuad_model()
        )

        self.questions = load_cuad_questions()

        if not self.questions:
            raise ClauseDetectionError(
                "No CUAD questions were loaded."
            )

    @torch.inference_mode()
    def detect(
        self,
        document: ProcessedDocument,
    ) -> List[DetectedClause]:
        """
        Detect legal clauses from the document.

        Parameters
        ----------
        document : ProcessedDocument
            Processed contract document.

        Returns
        -------
        List[DetectedClause]
            List of clauses detected using the CUAD
            extractive Question Answering model.
        """

        if document is None:
            raise ClauseDetectionError(
                "Input document is None."
            )

        detected_clauses: List[DetectedClause] = []

        for page in document.pages:

            page_text = page.cleaned_text or page.text

            if not page_text or not page_text.strip():
                continue

            chunks = chunk_text_with_stride(
                text=page_text,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                stride=self.config.stride,
            )

            for chunk in chunks:

                for question in self.questions:

                    result = self._run_qa(
                        question=question["prompt"],
                        context=chunk["text"],
                    )

                    if result is None:
                        continue

                    if (
                        result["score"]
                        < self.config.detection_threshold
                    ):
                        continue

                    answer = result["answer"].strip()

                    if not answer:
                        continue

                    if answer.lower() in {
                        "[cls]",
                        "[sep]",
                    }:
                        continue

                    span = ClauseSpan(
                        start_char=(
                            chunk["start_char"]
                            + result["start"]
                        ),
                        end_char=(
                            chunk["start_char"]
                            + result["end"]
                        ),
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

        # Remove duplicate detections caused by
        # overlapping chunks.
        detected_clauses = self._deduplicate_clauses(
            detected_clauses
        )

        # Sort clauses by document position.
        detected_clauses = self._sort_clauses(
            detected_clauses
        )

        return detected_clauses

    @torch.inference_mode()
    def _run_qa(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Any] | None:
        """
        Run extractive Question Answering directly using
        the CUAD model.

        This avoids the deprecated/unavailable
        `question-answering` Transformers pipeline task.
        """

        if not question or not context:
            return None

        encoding = self.tokenizer(
            question,
            context,
            max_length=self.config.max_seq_length,
            truncation="only_second",
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Identify which tokens belong to the context.
        sequence_ids = encoding.sequence_ids(0)

        context_token_indices = [
            i
            for i, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
        ]

        if not context_token_indices:
            return None

        # Move model inputs to the configured device.
        model_inputs = {
            key: value.to(self.config.device)
            for key, value in encoding.items()
            if key != "offset_mapping"
        }

        outputs = self.model(**model_inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Restrict predictions to context tokens.
        context_start = context_token_indices[0]
        context_end = context_token_indices[-1]

        start_logits = start_logits.clone()
        end_logits = end_logits.clone()

        start_mask = torch.full_like(
            start_logits,
            float("-inf"),
        )

        end_mask = torch.full_like(
            end_logits,
            float("-inf"),
        )

        start_mask[
            context_start : context_end + 1
        ] = start_logits[
            context_start : context_end + 1
        ]

        end_mask[
            context_start : context_end + 1
        ] = end_logits[
            context_start : context_end + 1
        ]

        start_probs = torch.softmax(
            start_mask,
            dim=-1,
        )

        end_probs = torch.softmax(
            end_mask,
            dim=-1,
        )

        # Best start and end positions.
        start_index = int(
            torch.argmax(start_probs).item()
        )

        end_index = int(
            torch.argmax(end_probs).item()
        )

        # Ensure the answer span is valid.
        if end_index < start_index:
            return None

        # Prevent extremely long accidental answers.
        max_answer_tokens = 128

        if (
            end_index - start_index + 1
            > max_answer_tokens
        ):
            end_index = (
                start_index
                + max_answer_tokens
                - 1
            )

            if end_index > context_end:
                end_index = context_end

        offsets = encoding["offset_mapping"][0]

        start_char = int(
            offsets[start_index][0].item()
        )

        end_char = int(
            offsets[end_index][1].item()
        )

        if end_char <= start_char:
            return None

        answer = context[start_char:end_char].strip()

        if not answer:
            return None

        # Combine start/end probabilities into a
        # simple span confidence.
        score = float(
            (
                start_probs[start_index]
                * end_probs[end_index]
            ).item()
        )

        return {
            "answer": answer,
            "score": score,
            "start": start_char,
            "end": end_char,
        }

    def _deduplicate_clauses(
        self,
        clauses: List[DetectedClause],
    ) -> List[DetectedClause]:
        """
        Remove duplicate clause detections caused by
        overlapping chunks.

        If multiple detections have the same category
        and nearly identical text, keep the one with
        the highest confidence.
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

            if (
                clause.confidence
                > unique[key].confidence
            ):
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
                clause.page_number or 0,
                (
                    clause.spans[0].start_char
                    if clause.spans
                    else 0
                ),
            ),
        )