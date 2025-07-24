"""Test configuration and fixtures for NLP Syntax Exercises."""

import pytest
import torch


@pytest.fixture
def sample_text():
    """Sample text for testing NLP functions."""
    return "Natural language processing is fascinating and powerful!"


@pytest.fixture
def sample_texts():
    """Sample texts for batch testing."""
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_model_name():
    """Mock model name for testing without downloading large models."""
    return "distilbert-base-uncased"  # Smaller model for faster tests


# Test data constants
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a powerful programming language.",
    "Machine learning models require large datasets.",
    "Natural language processing involves understanding text.",
]

SAMPLE_TOKENS = ["hello", "world", "tokenization", "example"]

SAMPLE_MULTILINGUAL_TEXT = {
    "english": "Hello, how are you?",
    "spanish": "Hola, ¿cómo estás?",
    "french": "Bonjour, comment allez-vous?",
    "german": "Hallo, wie geht es dir?",
}
