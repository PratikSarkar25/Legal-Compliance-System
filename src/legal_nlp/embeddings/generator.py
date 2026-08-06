"""
Embedding Generator Module.

Generate dense vector embeddings for legal clauses.
"""

from typing import List

import torch

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.schemas import (
    ClauseEmbedding,
    DetectedClause,
)
from src.legal_nlp.exceptions import EmbeddingGenerationError


class EmbeddingGenerator:
    """
    Generate dense vector embeddings for legal clauses.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        self.config = config or LegalNLPConfig()

        self.model_loader = ModelLoader(self.config)

        # Future implementation
        # self.embedding_model = self.model_loader.load_embedding_model()

    @torch.inference_mode()
    def generate_embeddings(
        self,
        clauses: List[DetectedClause],
    ) -> List[ClauseEmbedding]:
        """
        Generate embeddings for classified clauses.

        Parameters
        ----------
        clauses : List[DetectedClause]

        Returns
        -------
        List[ClauseEmbedding]
        """

        if clauses is None:
            raise EmbeddingGenerationError(
                "Input clauses cannot be None."
            )

        if not clauses:
            return []

        embeddings: List[ClauseEmbedding] = []

        for clause in clauses:

            try:

                vector = self._encode_text(
                    clause.text,
                )

                embedding = ClauseEmbedding(
                    clause_id=clause.clause_id,
                    vector=vector,
                    model_name=self.config.embedding_model_name,
                )

                embeddings.append(embedding)

            except Exception as e:

                raise EmbeddingGenerationError(
                    f"Failed to generate embedding "
                    f"for clause {clause.clause_id}: {e}"
                ) from e

        return embeddings

    def _encode_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Encode a single clause into a dense vector.
        """

        if not text or not text.strip():
            return []

        #
        # Future implementation:
        #
        # return self.embedding_model.encode(
        #     text,
        #     convert_to_numpy=True,
        # ).tolist()
        #

        return []