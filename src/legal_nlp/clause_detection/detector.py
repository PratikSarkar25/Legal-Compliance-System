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
from src.legal_nlp.schemas import DetectedClause
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
            model=model,
            tokenizer=tokenizer,
            device=0 if self.config.device == "cuda" else -1,
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
        parameters
        --------
        document: ProcessedDocument

        Returns
        -------
        List[DetectedClause]
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

                #
                # QA inference will be added
                # in the next commit.
                #
                pass

            return detected_clauses
        
        #raise NotImplementedError(
        #    "Clause detection implementation will be added next."
        #)