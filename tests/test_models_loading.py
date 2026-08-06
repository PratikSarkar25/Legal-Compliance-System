"""
Model loading tests.
"""

from src.legal_nlp.loader import ModelLoader


def test_model_loading():

    loader = ModelLoader()

    tokenizer, model = loader.load_cuad_model()

    assert tokenizer is not None

    assert model is not None


def test_embedding_model():

    loader = ModelLoader()

    model = loader.load_embedding_model()

    assert model is not None