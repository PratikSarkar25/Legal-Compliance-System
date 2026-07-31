from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# Stage 1
# CUAD → Clause Detection
# ============================================================

class ClauseDetectionResult(BaseModel):
    """
    Output produced by the Clause Detection module.
    """
    clause_id: str = Field(..., description="Unique clause identifier.")
    clause_type: str = Field(..., description="Detected CUAD clause type.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    page_number: Optional[int] = None


# ============================================================
# Stage 2
# CUAD → Clause Segmentation
# ============================================================

class ClauseSegment(BaseModel):
    """
    Output produced by Clause Segmentation.
    """
    clause_id: str

    text: str

    start_char: int

    end_char: int

    page_number: Optional[int] = None


# ============================================================
# Stage 3
# CUAD → Legal NER
# ============================================================

class LegalEntity(BaseModel):
    """
    Legal entity extracted from contract text.
    """

    entity_id: str

    clause_id: str

    text: str

    label: str

    start_char: int

    end_char: int

    confidence: float = Field(..., ge=0.0, le=1.0)

    page_number: Optional[int] = None


# ============================================================
# Stage 4
# Clause Object Builder
# ============================================================

class ClauseObject(BaseModel):
    """
    Unified clause object created by merging

    1. Detection
    2. Segmentation
    3. Legal NER
    """

    clause_id: str

    clause_type: str

    text: str

    page_number: Optional[int] = None

    detection_confidence: float

    entities: List[LegalEntity] = Field(default_factory=list)


# ============================================================
# Stage 5
# LEDGAR Classification
# ============================================================

class ClassifiedClause(BaseModel):
    """
    Clause after LEDGAR classification.
    """

    clause: ClauseObject

    ledgar_label: str

    confidence: float = Field(..., ge=0.0, le=1.0)


# ============================================================
# Stage 6
# Relation Extraction
# ============================================================

class LegalRelation(BaseModel):
    """
    Knowledge graph relation.
    """

    relation_id: str

    subject_id: str

    predicate: str

    object_id: str

    confidence: float = Field(default=1.0)


# ============================================================
# Stage 7
# Embedding Generation
# ============================================================

class ClauseEmbedding(BaseModel):
    """
    Dense embedding generated for FAISS.
    """

    clause_id: str

    vector: List[float]

    model_name: str


# ============================================================
# Final Output
# ============================================================

class ProcessedLegalDocument(BaseModel):
    """
    Final object produced by the Legal NLP Layer.
    """

    document_id: str

    document_path: str

    clauses: List[ClauseObject] = Field(default_factory=list)

    classified_clauses: List[ClassifiedClause] = Field(default_factory=list)

    relations: List[LegalRelation] = Field(default_factory=list)

    embeddings: List[ClauseEmbedding] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)