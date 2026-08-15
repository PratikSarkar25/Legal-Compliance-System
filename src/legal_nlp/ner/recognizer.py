"""
NER Recognizer Module.

This module provides the NERRecognizer class responsible for running pre-trained
named entity recognition on segmented clauses to isolate key legal entities
and return them as structured LegalEntity objects.
"""

from typing import List, Dict, Any

import torch
from transformers import pipeline

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.schemas import (
    DetectedClause,
    LegalEntity,
)
from src.legal_nlp.exceptions import NERExtractionError


class NERRecognizer:
    """
    Extracts legal and general entities (e.g., Organizations, Monetary Values, 
    Dates, Locations) from segmented clauses using a pre-trained Token Classification pipeline,
    returning structured LegalEntity objects.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        """
        Initialize the NERRecognizer with configuration and model loading infrastructure.
        """
        self.config = config or LegalNLPConfig()
        self.model_loader = ModelLoader(self.config)

        # Load the pretrained GLiNER model
        self.ner_model = self.model_loader.load_ner_model()

        # Entity types requested from GLiNER
        self.entity_labels = [
            "person",
            "organization",
            "location",
            "date",
            "money",
            "duration",
            "law",
            "court",
            "contract",
            "legal obligation",
        ]


        # tokenizer, model = self.model_loader.load_ner_model()

        # Placeholder pipeline setup until model weights/paths are defined
        #self.ner_pipeline = None  

    @torch.inference_mode()
    def extract_entities(
        self,
        clauses: List[DetectedClause],
    ) -> List[LegalEntity]:
        """
        Process segmented clauses and extract structured legal entities.
        """

        if clauses is None:
            raise NERExtractionError(
                "Input clauses cannot be None for entity extraction."
            )

        if not clauses:
            return []

        entities: List[LegalEntity] = []

        for clause in clauses:
            try:
                clause_entities = self._recognize_entities(clause)
                entities.extend(clause_entities)

            except Exception as e:
                raise NERExtractionError(
                    f"Failed to extract entities for clause "
                    f"{clause.clause_id}: {str(e)}"
                ) from e

        return entities

    def _recognize_entities(
        self,
        clause: DetectedClause,
    ) -> List[LegalEntity]:
        """
        Perform GLiNER inference on a single clause and convert
        predictions into LegalEntity objects.
        """

        if not clause.text or not clause.text.strip():
            return []

        try:
            predictions = self.ner_model.predict_entities(
                clause.text,
                self.entity_labels,
                threshold=self.config.ner_threshold,
            )

            entities: List[LegalEntity] = []

            for index, prediction in enumerate(predictions):
                entity_text = prediction["text"]
                label = prediction["label"]
                confidence = float(prediction["score"])
                start_char = int(prediction["start"])
                end_char = int(prediction["end"])

                entity = LegalEntity(
                    entity_id=f"{clause.clause_id}_entity_{index}",
                    clause_id=clause.clause_id,
                    text=entity_text,
                    label=label,
                    confidence=confidence,
                    start_char=start_char,
                    end_char=end_char,
                )

                entities.append(entity)

            return entities

        except Exception as e:
            raise NERExtractionError(
                f"NER inference failed for clause "
                f"{clause.clause_id}: {str(e)}"
            ) from e