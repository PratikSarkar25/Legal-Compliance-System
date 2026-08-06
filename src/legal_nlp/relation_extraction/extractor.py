"""
Relation Extractor Module.

This module provides the RelationExtractor class responsible for mapping
semantic relationships between classified clauses and recognized legal entities.
"""

from typing import List

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.schemas import (
    DetectedClause,
    LegalEntity,
    LegalRelation,
)
from src.legal_nlp.exceptions import RelationExtractionError


class RelationExtractor:
    """
    Build semantic relationships between classified clauses
    and recognized legal entities.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        """
        Initialize the RelationExtractor.
        """

        self.config = config or LegalNLPConfig()

        self.model_loader = ModelLoader(self.config)

        # Future implementation
        # self.model = self.model_loader.load_relation_model()

    def extract_relations(
        self,
        clauses: List[DetectedClause],
        entities: List[LegalEntity],
    ) -> List[LegalRelation]:
        """
        Construct semantic relationships between clauses
        and legal entities.

        Parameters
        ----------
        clauses : List[DetectedClause]
            Classified legal clauses.

        entities : List[LegalEntity]
            Extracted legal entities.

        Returns
        -------
        List[LegalRelation]
            Extracted semantic relationships.
        """

        if clauses is None or entities is None:
            raise RelationExtractionError(
                "Input clauses and entities cannot be None."
            )

        if not clauses or not entities:
            return []

        relations: List[LegalRelation] = []

        for clause in clauses:

            try:

                clause_relations = (
                    self._extract_clause_entity_relations(
                        clause,
                        entities,
                    )
                )

                relations.extend(clause_relations)

            except Exception as e:

                raise RelationExtractionError(
                    f"Failed to extract relations for "
                    f"clause {clause.clause_id}: {e}"
                ) from e

        return relations

    def _extract_clause_entity_relations(
        self,
        clause: DetectedClause,
        entities: List[LegalEntity],
    ) -> List[LegalRelation]:
        """
        Extract relationships between a clause and
        the legal entities contained within it.
        """

        matched_relations: List[LegalRelation] = []

        for entity in entities:

            if entity.clause_id != clause.clause_id:
                continue

            relation = LegalRelation(
                relation_id=(
                    f"{clause.clause_id}_"
                    f"{entity.entity_id}"
                ),
                subject_id=clause.clause_id,
                predicate="CONTAINS_ENTITY",
                object_id=entity.entity_id,
                confidence=entity.confidence,
            )

            matched_relations.append(relation)

        return matched_relations