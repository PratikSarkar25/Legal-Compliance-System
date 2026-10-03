"""
Contract NLI analyzer.

Compares a contract passage with a legal hypothesis and predicts:
Entailment, Contradiction, or NotMentioned.
"""

from typing import Any, Dict

import torch

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.exceptions import ModelLoadingError
from src.legal_nlp.loader import ModelLoader


class ContractNLIAnalyzer:
    """
    Performs Contract NLI inference using a pretrained Legal-BERT model.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        self.config = config or LegalNLPConfig()

        self.model_loader = ModelLoader(self.config)

        self.tokenizer, self.model = (
            self.model_loader.load_contract_nli_model()
        )

    @torch.inference_mode()
    def analyze(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict[str, Any]:
        """
        Compare a contract passage with a legal hypothesis.

        Returns
        -------
        Dict[str, Any]
            Predicted label and confidence scores.
        """

        if not premise or not premise.strip():
            raise ValueError("Premise cannot be empty.")

        if not hypothesis or not hypothesis.strip():
            raise ValueError("Hypothesis cannot be empty.")

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        )

        inputs = {
            key: value.to(self.config.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        prediction_id = torch.argmax(probabilities).item()

        label = self.model.config.id2label[prediction_id]

        confidence = probabilities[prediction_id].item()

        scores = {
            self.model.config.id2label[index]: probability.item()
            for index, probability in enumerate(probabilities)
        }

        return {
            "label": label,
            "confidence": confidence,
            "scores": scores,
        }
