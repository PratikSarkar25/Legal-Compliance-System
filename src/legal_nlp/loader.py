from typing import Any, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.exceptions import ModelLoadingError


class ModelLoader:
    """
    Loads and caches all transformer models used in the Legal NLP layer.

    Responsibilities
    ----------------
    - Download models if not cached
    - Cache tokenizer/model pairs
    - Return ready-to-use models

    This class DOES NOT perform inference.
    """

    def __init__(self, config: Optional[LegalNLPConfig] = None):

        self.config = config or LegalNLPConfig()

        self._cache: Dict[str, Any] = {}

    # =====================================================
    # CUAD Question Answering
    # =====================================================

    def load_cuad_model(self) -> Tuple[Any, Any]:

        cache_key = "cuad"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.cuad_model_name,
                cache_dir=self.config.cuad_dir,
            )

            model = AutoModelForQuestionAnswering.from_pretrained(
                self.config.cuad_model_name,
                cache_dir=self.config.cuad_dir,
            ).to(self.config.device)

            model.eval()

            self._cache[cache_key] = (tokenizer, model)

            return tokenizer, model

        except Exception as e:
            raise ModelLoadingError(
                f"Unable to load CUAD model: {e}"
            ) from e

    # =====================================================
    # LEDGAR
    # =====================================================

    def load_ledgar_model(self) -> Tuple[Any, Any]:

        cache_key = "ledgar"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.ledgar_model_name,
                cache_dir=self.config.ledgar_dir,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.ledgar_model_name,
                cache_dir=self.config.ledgar_dir,
            ).to(self.config.device)

            model.eval()

            self._cache[cache_key] = (tokenizer, model)

            return tokenizer, model

        except Exception as e:
            raise ModelLoadingError(
                f"Unable to load LEDGAR model: {e}"
            ) from e

    # =====================================================
    # Legal NER
    # =====================================================

    def load_ner_model(self) -> Tuple[Any, Any]:

        cache_key = "ner"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.ner_model_name,
                cache_dir=self.config.ner_dir,
            )

            model = AutoModelForTokenClassification.from_pretrained(
                self.config.ner_model_name,
                cache_dir=self.config.ner_dir,
            ).to(self.config.device)

            model.eval()

            self._cache[cache_key] = (tokenizer, model)

            return tokenizer, model

        except Exception as e:
            raise ModelLoadingError(
                f"Unable to load NER model: {e}"
            ) from e

    # =====================================================
    # Embedding Model
    # =====================================================

    def load_embedding_model(self) -> SentenceTransformer:

        cache_key = "embedding"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:

            model = SentenceTransformer(
                self.config.embedding_model_name,
                cache_folder=str(self.config.embedding_dir),
                device=self.config.device,
            )

            self._cache[cache_key] = model

            return model

        except Exception as e:
            raise ModelLoadingError(
                f"Unable to load embedding model: {e}"
            ) from e

    # =====================================================
    # Utilities
    # =====================================================

    def clear_cache(self) -> None:
        """Clear all cached models."""

        self._cache.clear()