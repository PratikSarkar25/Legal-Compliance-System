from typing import Any, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.exceptions import ModelLoadingError


class ModelLoader:
    """
    Centralized manager responsible for loading and caching all
    transformer models used by the Legal NLP layer.

    This class ONLY loads models.
    It does NOT perform inference.
    """

    def __init__(self, config: Optional[LegalNLPConfig] = None):
        self.config = config or LegalNLPConfig()
        self._cache: Dict[str, Any] = {}

    # ==========================================================
    # CUAD
    # ==========================================================

    def load_cuad_model(self) -> Tuple[Any, Any]:
        """
        Load the CUAD fine-tuned RoBERTa model.

        Returns:
            (tokenizer, model)
        """

        cache_key = f"cuad_{self.config.cuad_model_name}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.cuad_model_name,
                cache_dir=self.config.cuad_dir,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.cuad_model_name,
                cache_dir=self.config.cuad_dir,
            ).to(self.config.device)

            model.eval()

            self._cache[cache_key] = (tokenizer, model)

            return tokenizer, model

        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load CUAD model '{self.config.cuad_model_name}': {e}"
            ) from e

    # ==========================================================
    # LEDGAR
    # ==========================================================

    def load_ledgar_model(self) -> Tuple[Any, Any]:
        """
        Load the LEDGAR LegalBERT model.

        Returns:
            (tokenizer, model)
        """

        cache_key = f"ledgar_{self.config.ledgar_model_name}"

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
                f"Failed to load LEDGAR model '{self.config.ledgar_model_name}': {e}"
            ) from e

    # ==========================================================
    # Legal NER
    # ==========================================================

    def load_ner_model(self) -> Tuple[Any, Any]:
        """
        Load the Legal NER model.

        Returns:
            (tokenizer, model)
        """

        cache_key = f"ner_{self.config.ner_model_name}"

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
                f"Failed to load NER model '{self.config.ner_model_name}': {e}"
            ) from e

    # ==========================================================
    # Embedding Model
    # ==========================================================

    def load_embedding_model(self) -> SentenceTransformer:
        """
        Load the sentence embedding model.

        Returns:
            SentenceTransformer model.
        """

        cache_key = f"embedding_{self.config.embedding_model_name}"

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
                f"Failed to load embedding model '{self.config.embedding_model_name}': {e}"
            ) from e

    # ==========================================================
    # Cache Management
    # ==========================================================

    def clear_cache(self) -> None:
        """
        Clear all cached models.
        """

        self._cache.clear()