"""
Clause Classifier Module.

This module provides the ClauseClassifier class responsible for running
pre-trained LEDGAR LegalBERT classification on segmented clauses.
"""

from typing import Any, Dict, List

import torch

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.exceptions import ClauseClassificationError
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.schemas import DetectedClause


class ClauseClassifier:
    """
    Classifies segmented clauses into standardized legal categories using
    a pre-trained LEDGAR sequence classification model.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        """
        Initialize the ClauseClassifier.
        """
        self.config = config or LegalNLPConfig()

        self.model_loader = ModelLoader(self.config)

        # tokenizer, model = self.model_loader.load_ledgar_model()

        # Will be initialized after enabling inference.
        self.classifier_pipeline = None

    @torch.inference_mode()
    def classify_clauses(
        self,
        clauses: List[DetectedClause],
    ) -> List[DetectedClause]:
        """
        Classify segmented clauses using the LEDGAR model.

        Parameters
        ----------
        clauses : List[DetectedClause]
            Segmented clauses produced by the ClauseSegmenter.

        Returns
        -------
        List[DetectedClause]
            Clauses enriched with LEDGAR classification results.
        """

        if clauses is None:
            raise ClauseClassificationError(
                "Input clauses cannot be None for classification."
            )

        if not clauses:
            return []

        classified_clauses: List[DetectedClause] = []

        for clause in clauses:

            try:

                prediction = self._classify_text(
                    clause.text,
                )

                #
                # Preserve the CUAD category.
                # Store LEDGAR prediction separately.
                #
                clause.classification_label = prediction["label"]
                clause.classification_score = prediction["score"]

                classified_clauses.append(clause)

            except Exception as e:

                raise ClauseClassificationError(
                    f"Failed to classify clause "
                    f"{clause.clause_id}: {e}"
                ) from e

        return classified_clauses

    def _classify_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Perform low-level text classification on a single clause.
        """

        if not text or not text.strip():

            return {
                "label": "UNKNOWN",
                "score": 0.0,
            }

        #
        # Future implementation:
        #
        # return self.classifier_pipeline(text)[0]
        #

        return {
            "label": "UNCLASSIFIED",
            "score": 0.0,
        }