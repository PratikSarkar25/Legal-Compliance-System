"""
Legal NLP Pipeline.

Coordinates the complete Legal NLP workflow by orchestrating all
individual processing modules.
"""

from src.document_processing.schemas import ProcessedDocument

from src.legal_nlp.clause_builder.builder import ClauseBuilder
from src.legal_nlp.clause_classification.classifier import ClauseClassifier
from src.legal_nlp.clause_detection.detector import ClauseDetector
from src.legal_nlp.clause_segmentation.segmenter import ClauseSegmenter
from src.legal_nlp.config import LegalNLPConfig
from src.legal_nlp.embeddings.generator import EmbeddingGenerator
from src.legal_nlp.ner.recognizer import NERRecognizer
from src.legal_nlp.relation_extraction.extractor import RelationExtractor
from src.legal_nlp.schemas import ProcessedLegalDocument


class LegalNLPPipeline:
    """
    Complete Legal NLP processing pipeline.
    """

    def __init__(
        self,
        config: LegalNLPConfig | None = None,
    ):
        self.config = config or LegalNLPConfig()

        self.detector = ClauseDetector(self.config)
        self.builder = ClauseBuilder(self.config)
        self.segmenter = ClauseSegmenter(self.config)
        self.ner = NERRecognizer(self.config)
        self.classifier = ClauseClassifier(self.config)
        self.relation_extractor = RelationExtractor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)

    def process(
        self,
        document: ProcessedDocument,
    ) -> ProcessedLegalDocument:
        """
        Execute the complete Legal NLP pipeline.
        """

        # 1. Clause Detection (CUAD)
        clauses = self.detector.detect(document)

        # 2. Clause Builder
        clauses = self.builder.build(
            document=document,
            raw_clauses=clauses,
        )

        # 3. Clause Segmentation
        clauses = self.segmenter.segment(clauses)

        # 4. Named Entity Recognition
        entities = self.ner.extract_entities(clauses)

        # 5. Clause Classification (LEDGAR)
        clauses = self.classifier.classify_clauses(clauses)

        # 6. Relation Extraction
        relations = self.relation_extractor.extract_relations(
            clauses=clauses,
            entities=entities,
        )

        # 7. Embedding Generation
        embeddings = self.embedding_generator.generate_embeddings(
            clauses
        )

        return ProcessedLegalDocument(
            document_id=document.filename,
            filename=document.filename,
            clauses=clauses,
            entities=entities,
            relations=relations,
            embeddings=embeddings,
            metadata={
                "num_pages": document.metadata.num_pages,
                "language": document.metadata.language,
                "extraction_method": document.extraction_method,
                "processed_at": str(document.processed_at),
            },
        )