#!/usr/bin/env python3
"""
Production-ready configurable tokenizer for NLP pipelines.

Features:
- Configurable tokenization modes (regex, whitespace)
- Comprehensive text normalization options
- Memory-efficient streaming processing
- Deterministic execution and validation
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Iterator
from enum import Enum
import numpy as np
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenizationMode(Enum):
    """Supported tokenization modes."""
    REGEX = "regex"
    WHITESPACE = "whitespace"


class TextNormalization(Enum):
    """Supported text normalization options."""
    LOWERCASE = "lowercase"
    STRIP_ACCENTS = "strip_accents"
    COLLAPSE_WHITESPACE = "collapse_whitespace"
    NONE = "none"


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer behavior."""
    # Tokenization mode
    mode: TokenizationMode = TokenizationMode.REGEX
    
    # Normalization options
    normalization: List[TextNormalization] = None
    
    # Token filtering
    min_token_length: int = 1
    max_token_length: int = 100
    filter_numeric: bool = False
    filter_punctuation: bool = False
    
    # Regex pattern (used in REGEX mode)
    regex_pattern: str = r'\b\w+\b'
    
    # Special tokens
    special_tokens: Dict[str, int] = None
    
    # Performance and memory
    batch_size: int = 1000
    streaming: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.normalization is None:
            self.normalization = [TextNormalization.LOWERCASE]
        
        if self.special_tokens is None:
            self.special_tokens = {}
        
        # Validate parameters
        if self.min_token_length < 0:
            raise ValueError("min_token_length must be non-negative")
        if self.max_token_length <= self.min_token_length:
            raise ValueError("max_token_length must be greater than min_token_length")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Set seed for reproducibility
        np.random.seed(self.seed)


class TextNormalizer:
    """Handles text normalization operations."""
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    @staticmethod
    def strip_accents(text: str) -> str:
        """Basic accent stripping using regex."""
        # Simple implementation - in production, consider using unicodedata
        text = re.sub(r'[áàâä]', 'a', text)
        text = re.sub(r'[éèêë]', 'e', text)
        text = re.sub(r'[íìîï]', 'i', text)
        text = re.sub(r'[óòôö]', 'o', text)
        text = re.sub(r'[úùûü]', 'u', text)
        text = re.sub(r'[ç]', 'c', text)
        return text
    
    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into single space."""
        return re.sub(r'\s+', ' ', text).strip()
    
    @classmethod
    def normalize_text(cls, text: str, normalization_ops: List[TextNormalization]) -> str:
        """Apply sequence of normalization operations to text."""
        if TextNormalization.NONE in normalization_ops:
            return text
        
        normalized_text = text
        for op in normalization_ops:
            if op == TextNormalization.LOWERCASE:
                normalized_text = cls.lowercase(normalized_text)
            elif op == TextNormalization.STRIP_ACCENTS:
                normalized_text = cls.strip_accents(normalized_text)
            elif op == TextNormalization.COLLAPSE_WHITESPACE:
                normalized_text = cls.collapse_whitespace(normalized_text)
        
        return normalized_text


class TokenFilter:
    """Handles token filtering logic."""
    
    @staticmethod
    def is_numeric(token: str) -> bool:
        """Check if token consists only of numeric characters."""
        return token.isdigit()
    
    @staticmethod
    def is_punctuation(token: str) -> bool:
        """Check if token consists only of punctuation."""
        return all(not char.isalnum() for char in token) and len(token) > 0
    
    @classmethod
    def filter_token(cls, token: str, config: TokenizerConfig) -> bool:
        """Determine if token should be filtered out."""
        # Length filtering
        if not (config.min_token_length <= len(token) <= config.max_token_length):
            return True
        
        # Numeric filtering
        if config.filter_numeric and cls.is_numeric(token):
            return True
        
        # Punctuation filtering
        if config.filter_punctuation and cls.is_punctuation(token):
            return True
        
        return False


class PipelineTokenizer:
    """
    Configurable, production-ready tokenizer for NLP pipelines.
    
    Features:
    - Multiple tokenization modes
    - Comprehensive text normalization
    - Flexible token filtering
    - Memory-efficient streaming
    - Deterministic execution
    """
    
    def __init__(self, config: TokenizerConfig):
        """Initialize tokenizer with configuration."""
        self.config = config
        self.normalizer = TextNormalizer()
        self.filter = TokenFilter()
        
        # Validate regex pattern if using regex mode
        if config.mode == TokenizationMode.REGEX:
            try:
                re.compile(config.regex_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        
        logger.info(f"Tokenizer initialized with mode: {config.mode.value}")
    
    def tokenize_single(self, text: str) -> List[str]:
        """
        Tokenize a single text string.
        
        Args:
            text: Input text string to tokenize
            
        Returns:
            List of tokens
            
        Raises:
            ValueError: If text is not a string or empty
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")
        
        if not text.strip():
            return []
        
        try:
            # Apply normalization
            normalized_text = self.normalizer.normalize_text(
                text, self.config.normalization
            )
            
            # Tokenize based on mode
            if self.config.mode == TokenizationMode.WHITESPACE:
                raw_tokens = normalized_text.split()
            else:  # REGEX mode
                raw_tokens = re.findall(self.config.regex_pattern, normalized_text)
            
            # Filter tokens
            filtered_tokens = [
                token for token in raw_tokens
                if not self.filter.filter_token(token, self.config)
            ]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            List of token lists
            
        Raises:
            ValueError: If texts is not a list or contains non-string elements
        """
        if not isinstance(texts, list):
            raise ValueError(f"Expected list, got {type(texts)}")
        
        if not texts:
            return []
        
        # Validate all elements are strings
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Element {i} is not a string: {type(text)}")
        
        try:
            return [self.tokenize_single(text) for text in texts]
        except Exception as e:
            logger.error(f"Error in batch tokenization: {e}")
            raise
    
    def tokenize_stream(self, text_iterator: Iterator[str]) -> Iterator[List[str]]:
        """
        Tokenize a stream of texts (memory-efficient for large datasets).
        
        Args:
            text_iterator: Iterator yielding text strings
            
        Yields:
            Lists of tokens for each text
            
        Raises:
            ValueError: If text_iterator is not an iterator
        """
        if not hasattr(text_iterator, '__iter__'):
            raise ValueError("text_iterator must be an iterator")
        
        batch = []
        for text in text_iterator:
            if not isinstance(text, str):
                logger.warning(f"Skipping non-string element in stream: {type(text)}")
                continue
            
            batch.append(text)
            
            if len(batch) >= self.config.batch_size:
                try:
                    yield from self.tokenize_batch(batch)
                    batch = []
                except Exception as e:
                    logger.error(f"Error processing batch in stream: {e}")
                    # Continue with next batch
                    batch = []
        
        # Process remaining texts
        if batch:
            try:
                yield from self.tokenize_batch(batch)
            except Exception as e:
                logger.error(f"Error processing final batch in stream: {e}")
    
    def get_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary mapping tokens to frequencies
            
        Raises:
            ValueError: If texts is empty or invalid
        """
        if not texts:
            raise ValueError("Cannot build vocabulary from empty text list")
        
        vocab = {}
        all_tokens = self.tokenize_batch(texts)
        
        for tokens in all_tokens:
            for token in tokens:
                vocab[token] = vocab.get(token, 0) + 1
        
        # Add special tokens
        for special_token in self.config.special_tokens:
            if special_token not in vocab:
                vocab[special_token] = 0
        
        return vocab
    
    def __call__(self, text_input) -> List[str]:
        """Make tokenizer callable for single text inputs."""
        if isinstance(text_input, str):
            return self.tokenize_single(text_input)
        elif isinstance(text_input, list):
            return self.tokenize_batch(text_input)
        else:
            raise ValueError(f"Unsupported input type: {type(text_input)}")


def create_default_tokenizer() -> PipelineTokenizer:
    """Create tokenizer with sensible defaults."""
    config = TokenizerConfig(
        mode=TokenizationMode.REGEX,
        normalization=[
            TextNormalization.LOWERCASE,
            TextNormalization.COLLAPSE_WHITESPACE
        ],
        min_token_length=2,
        max_token_length=50,
        filter_numeric=False,
        filter_punctuation=True,
        regex_pattern=r'\b\w+\b',
        special_tokens={'[UNK]': 0, '[PAD]': 1},
        streaming=True,
        batch_size=1000,
        seed=42
    )
    return PipelineTokenizer(config)


def deterministic_verification():
    """
    Run deterministic verification of tokenizer functionality.
    
    This routine validates that all core components work correctly
    and produce deterministic, repeatable output.
    """
    logger.info("Starting deterministic verification...")
    
    # Test configuration
    test_config = TokenizerConfig(
        mode=TokenizationMode.REGEX,
        normalization=[TextNormalization.LOWERCASE],
        min_token_length=2,
        filter_punctuation=True,
        seed=42
    )
    
    tokenizer = PipelineTokenizer(test_config)
    
    # Test cases with expected results
    test_cases = [
        (
            "Hello World! This is a test.",
            ["hello", "world", "this", "test"]
        ),
        (
            "Multiple   spaces   here",
            ["multiple", "spaces", "here"]
        ),
        (
            "Short words: a b c d longwordhere",
            ["short", "words", "longwordhere"]
        ),
        (
            "Numbers 123 and punctuation!!!",
            ["numbers", "and", "punctuation"]
        )
    ]
    
    all_passed = True
    
    for i, (input_text, expected_tokens) in enumerate(test_cases):
        try:
            result = tokenizer.tokenize_single(input_text)
            
            if result == expected_tokens:
                logger.info(f"✓ Test case {i+1} passed")
            else:
                logger.error(f"✗ Test case {i+1} failed")
                logger.error(f"  Input: {input_text}")
                logger.error(f"  Expected: {expected_tokens}")
                logger.error(f"  Got: {result}")
                all_passed = False
                
        except Exception as e:
            logger.error(f"✗ Test case {i+1} raised exception: {e}")
            all_passed = False
    
    # Test batch processing
    try:
        batch_input = [case[0] for case in test_cases]
        batch_result = tokenizer.tokenize_batch(batch_input)
        
        if len(batch_result) == len(test_cases):
            logger.info("✓ Batch processing test passed")
        else:
            logger.error("✗ Batch processing test failed")
            all_passed = False
    except Exception as e:
        logger.error(f"✗ Batch processing test raised exception: {e}")
        all_passed = False
    
    # Test vocabulary building
    try:
        vocab = tokenizer.get_vocabulary([case[0] for case in test_cases])
        expected_vocab_size = len(set(
            token for case in test_cases for token in case[1]
        ))
        
        if len(vocab) >= expected_vocab_size:
            logger.info("✓ Vocabulary building test passed")
        else:
            logger.error("✗ Vocabulary building test failed")
            all_passed = False
    except Exception as e:
        logger.error(f"✗ Vocabulary building test raised exception: {e}")
        all_passed = False
    
    # Test determinism
    try:
        test_text = "Deterministic test with multiple runs"
        results = []
        
        for _ in range(3):
            result = tokenizer.tokenize_single(test_text)
            results.append(result)
        
        if all(r == results[0] for r in results):
            logger.info("✓ Determinism test passed")
        else:
            logger.error("✗ Determinism test failed")
            all_passed = False
    except Exception as e:
        logger.error(f"✗ Determinism test raised exception: {e}")
        all_passed = False
    
    if all_passed:
        logger.info("🎉 All verification tests passed!")
    else:
        logger.error("❌ Some verification tests failed!")
    
    return all_passed


def main():
    """Main execution function with example usage."""
    logger.info("Pipeline Tokenizer Demo")
    
    # Run verification first
    if not deterministic_verification():
        logger.error("Verification failed - exiting")
        return
    
    # Demo with different configurations
    logger.info("\n--- Demo: Different Tokenization Modes ---")
    
    # Regex mode demo
    regex_config = TokenizerConfig(
        mode=TokenizationMode.REGEX,
        normalization=[TextNormalization.LOWERCASE],
        regex_pattern=r'\b\w+\b'
    )
    regex_tokenizer = PipelineTokenizer(regex_config)
    
    sample_text = "Hello World! This demonstrates regex tokenization."
    regex_tokens = regex_tokenizer(sample_text)
    logger.info(f"Regex tokens: {regex_tokens}")
    
    # Whitespace mode demo
    whitespace_config = TokenizerConfig(
        mode=TokenizationMode.WHITESPACE,
        normalization=[
            TextNormalization.LOWERCASE,
            TextNormalization.COLLAPSE_WHITESPACE
        ]
    )
    whitespace_tokenizer = PipelineTokenizer(whitespace_config)
    whitespace_tokens = whitespace_tokenizer(sample_text)
    logger.info(f"Whitespace tokens: {whitespace_tokens}")
    
    # Batch processing demo
    logger.info("\n--- Demo: Batch Processing ---")
    batch_texts = [
        "First document with some text.",
        "Second document here!",
        "Third one with numbers 123."
    ]
    batch_tokens = regex_tokenizer.tokenize_batch(batch_texts)
    for i, tokens in enumerate(batch_tokens):
        logger.info(f"Document {i+1}: {tokens}")
    
    # Vocabulary demo
    logger.info("\n--- Demo: Vocabulary Building ---")
    vocab = regex_tokenizer.get_vocabulary(batch_texts)
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Sample vocabulary items: {list(vocab.items())[:5]}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    # Set global seeds for maximum reproducibility
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        raise