"""
Custom exceptions for the Legal NLP layer.
"""


class LegalNLPError(Exception):
    """
    Base exception for all Legal NLP errors.
    """
    pass


# ==========================================================
# Model Loading
# ==========================================================

class ModelLoadingError(LegalNLPError):
    """Raised when a pretrained model cannot be loaded."""
    pass


# ==========================================================
# Clause Detection
# ==========================================================

class ClauseDetectionError(LegalNLPError):
    """Raised during CUAD clause detection."""
    pass


# ==========================================================
# Clause Builder
# ==========================================================

class ClauseBuilderError(LegalNLPError):
    """Raised when clause normalization or validation fails."""
    pass


# ==========================================================
# Clause Segmentation
# ==========================================================

class ClauseSegmentationError(LegalNLPError):
    """Raised when clause segmentation fails."""
    pass


# ==========================================================
# Named Entity Recognition
# ==========================================================

class NERExtractionError(LegalNLPError):
    """Raised during named entity recognition."""
    pass


# ==========================================================
# Clause Classification
# ==========================================================

class ClauseClassificationError(LegalNLPError):
    """Raised during LEDGAR classification."""
    pass


# ==========================================================
# Relation Extraction
# ==========================================================

class RelationExtractionError(LegalNLPError):
    """Raised during relation extraction."""
    pass


# ==========================================================
# Embedding Generation
# ==========================================================

class EmbeddingGenerationError(LegalNLPError):
    """Raised during embedding generation."""
    pass