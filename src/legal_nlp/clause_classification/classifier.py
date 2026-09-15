"""
LEDGAR clause classification module.
"""

from typing import Any, Dict, List

import torch

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.exceptions import ClauseClassificationError
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.schemas import DetectedClause


class ClauseClassifier:
    """
    Classifies legal clauses using a pretrained 100-class LEDGAR classifier.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        self.config = config or LegalNLPConfig()
        self.model_loader = ModelLoader(self.config)

        self.tokenizer, self.model = self.model_loader.load_ledgar_model()

    @torch.inference_mode()
    def classify_clauses(
        self,
        clauses: List[DetectedClause],
    ) -> List[DetectedClause]:
        """
        Add LEDGAR labels and confidence scores to detected clauses.
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
                prediction = self._classify_text(clause.text)

                clause.classification_label = prediction["label"]
                clause.classification_score = prediction["score"]

                classified_clauses.append(clause)

            except Exception as error:
                raise ClauseClassificationError(
                    f"Failed to classify clause {clause.clause_id}: {error}"
                ) from error

        return classified_clauses

    @torch.inference_mode()
    def _classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify one clause using the LEDGAR model.
        """

        if not text or not text.strip():
            return {
                "label": "UNKNOWN",
                "score": 0.0,
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.config.truncation,
            padding=False,
            max_length=self.config.max_seq_length,
        )

        inputs = {
            key: value.to(self.config.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0, predicted_id].item())

        label = self.model.config.id2label.get(
            predicted_id,
            f"LABEL_{predicted_id}",
        )

        return {
            "label": label,
            "score": confidence,
            "label_id": predicted_id,
        }
