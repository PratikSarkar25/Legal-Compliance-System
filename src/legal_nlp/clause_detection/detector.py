from typing import List, Optional

import torch

from src.document_processing.schemas import StructuredDocument
from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.exceptions import ClauseDetectionError
from src.legal_nlp.loader import ModelLoader
from src.legal_nlp.schemas import ClauseDetectionResult
from src.legal_nlp.utils import chunk_text_with_stride


class ClauseDetector:
    """
    CUAD-based Clause Detection module.

    Responsibilities
    ----------------
    1. Receive a StructuredDocument.
    2. Split long contract text into overlapping chunks.
    3. Ask CUAD legal questions.
    4. Extract clause spans.
    5. Return ClauseDetectionResult objects.

    NOTE:
    This module ONLY performs clause detection.
    """

    def __init__(self, config: Optional[LegalNLPConfig] = None):

        self.config = config or LegalNLPConfig()

        self.loader = ModelLoader(self.config)

        self.tokenizer, self.model = self.loader.load_cuad_model()

    @torch.inference_mode()
    def detect(
        self,
        document: StructuredDocument,
    ) -> List[ClauseDetectionResult]:
        """
        Detect legal clauses using the CUAD Question Answering model.

        Parameters
        ----------
        document : StructuredDocument

        Returns
        -------
        List[ClauseDetectionResult]
        """

        if document is None:
            raise ClauseDetectionError("Input document is None.")

        raise NotImplementedError(
            "QA inference will be implemented after adding CUAD questions."
        )