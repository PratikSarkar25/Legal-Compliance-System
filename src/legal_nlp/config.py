from pathlib import Path

import torch

from pydantic import BaseModel, Field


class LegalNLPConfig(BaseModel):
    """
    Configuration for the Legal NLP Layer.
    """

    # ========================================================
    # CUAD
    # ========================================================

    cuad_model_name: str = Field(
        default="marshmellow77/roberta-base-cuad"
    )

    # ========================================================
    # LEDGAR
    # ========================================================

    ledgar_model_name: str = Field(
        default="nlpaueb/legal-bert-base-uncased"
    )

    # ========================================================
    # Legal NER
    # ========================================================

    ner_model_name: str = Field(
        default=""
    )

    # ========================================================
    # Embedding Model
    # ========================================================

    embedding_model_name: str = Field(
        default="BAAI/bge-m3"
    )

    # ========================================================
    # Runtime
    # ========================================================

    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size: int = Field(
        default=8
    )

    max_seq_length: int = Field(
        default=512
    )

    stride: int = Field(
        default=128
    )

    padding: str = Field(
        default="max_length"
    )

    truncation: bool = Field(
        default=True
    )

    # ========================================================
    # Thresholds
    # ========================================================

    detection_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1
    )

    classification_threshold: float = Field(
        default=0.6,
        ge=0,
        le=1
    )

    ner_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1
    )

    # ========================================================
    # Model Cache
    # ========================================================

    model_root: Path = Path("models")

    cuad_dir: Path = Path("models/cuad")

    ledgar_dir: Path = Path("models/ledgar")

    ner_dir: Path = Path("models/ner")

    embedding_dir: Path = Path("models/embeddings")