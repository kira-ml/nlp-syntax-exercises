"""
18_tokenize_with_sentencepiece.py

Train and apply a SentencePiece model (unigroup or BPE) on a sample corpus using Python CLI subprocess calls.
Concepts: Subword tokenization, external tool integration via subprocess.
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SentencePieceConfig:
    """Configuration for SentencePiece model training and tokenization."""
    
    # Model parameters
    vocab_size: int = 8000
    model_type: str = "unigram"  # "unigram" or "bpe"
    character_coverage: float = 1.0
    max_sentence_length: int = 4192
    
    # Training parameters
    input_sentence_size: int = 1000000
    shuffle_input_sentence: bool = True
    seed: int = 42
    
    # File paths
    model_prefix: str = "spm_model"
    temp_dir: str = "./spm_temp"
    
    # Tokenization parameters
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True


class SentencePieceTrainer:
    """Handles SentencePiece model training and tokenization via subprocess calls."""
    
    def __init__(self, config: SentencePieceConfig):
        self.config = config
        self.model_path = None
        self._ensure_dependencies()
        
    def _ensure_dependencies(self) -> None:
        """Verify SentencePiece CLI is available."""
        try:
            result = subprocess.run(
                ["spm_train", "--help"], 
                capture_output=True, 
                text=True,
                check=False
            )
            if result.returncode != 0:
                logger.warning("SentencePiece CLI might not be properly installed")
        except FileNotFoundError:
            raise RuntimeError(
                "SentencePiece CLI not found. Please install: "
                "pip install sentencepiece and ensure spm_train is in PATH"
            )
    
    def _prepare_temp_directory(self) -> None:
        """Create and configure temporary directory for model files."""
        os.makedirs(self.config.temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {self.config.temp_dir}")
    
    def _write_corpus_to_file(self, corpus: List[str], filepath: str) -> None:
        """Write corpus to temporary file for SentencePiece processing."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in corpus:
                f.write(sentence.strip() + '\n')
        logger.info(f"Written {len(corpus)} sentences to {filepath}")
    
    def _build_spm_train_command(self, input_file: str) -> List[str]:
        """Build SentencePiece training command with configured parameters."""
        cmd = [
            "spm_train",
            f"--input={input_file}",
            f"--model_prefix={os.path.join(self.config.temp_dir, self.config.model_prefix)}",
            f"--vocab_size={self.config.vocab_size}",
            f"--model_type={self.config.model_type}",
            f"--character_coverage={self.config.character_coverage}",
            f"--max_sentence_length={self.config.max_sentence_length}",
            f"--input_sentence_size={self.config.input_sentence_size}",
            f"--seed={self.config.seed}",
        ]
        
        if self.config.shuffle_input_sentence:
            cmd.append("--shuffle_input_sentence=true")
        else:
            cmd.append("--shuffle_input_sentence=false")
            
        if self.config.model_type == "bpe":
            cmd.append("--split_by_whitespace=false")
        
        return cmd
    
    def _build_spm_encode_command(self, input_file: str, output_file: str) -> List[str]:
        """Build SentencePiece encoding command."""
        if not self.model_path:
            raise ValueError("Model must be trained before encoding")
            
        cmd = [
            "spm_encode",
            f"--model={self.model_path}",
            f"--input={input_file}",
            f"--output={output_file}",
        ]
        
        if self.config.add_dummy_prefix:
            cmd.append("--add_dummy_prefix=true")
        else:
            cmd.append("--add_dummy_prefix=false")
            
        if self.config.remove_extra_whitespaces:
            cmd.append("--remove_extra_whitespaces=true")
        else:
            cmd.append("--remove_extra_whitespaces=false")
        
        return cmd
    
    def train(self, corpus: List[str]) -> str:
        """
        Train SentencePiece model on provided corpus.
        
        Args:
            corpus: List of text sentences for training
            
        Returns:
            Path to trained model file
            
        Raises:
            RuntimeError: If training process fails
        """
        logger.info(f"Starting SentencePiece training with {len(corpus)} sentences")
        logger.info(f"Model type: {self.config.model_type}, Vocab size: {self.config.vocab_size}")
        
        self._prepare_temp_directory()
        
        # Write corpus to temporary file
        input_file = os.path.join(self.config.temp_dir, "corpus.txt")
        self._write_corpus_to_file(corpus, input_file)
        
        # Build and execute training command
        cmd = self._build_spm_train_command(input_file)
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600  # 1 hour timeout
            )
            logger.info("SentencePiece training completed successfully")
            logger.debug(f"Training stdout: {result.stdout}")
            
            if result.stderr:
                logger.warning(f"Training stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with return code {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"SentencePiece training failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            logger.error("Training timed out after 1 hour")
            raise RuntimeError("SentencePiece training timed out")
        
        # Set model path
        self.model_path = os.path.join(self.config.temp_dir, f"{self.config.model_prefix}.model")
        
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model file not found at expected path: {self.model_path}")
            
        logger.info(f"Model saved to: {self.model_path}")
        return self.model_path
    
    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize texts using trained SentencePiece model.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of tokenized sentences as lists of subword tokens
        """
        if not self.model_path:
            raise ValueError("Model must be trained before tokenization")
            
        logger.info(f"Tokenizing {len(texts)} texts")
        
        # Write texts to temporary file
        input_file = os.path.join(self.config.temp_dir, "input_texts.txt")
        output_file = os.path.join(self.config.temp_dir, "tokenized.txt")
        
        self._write_corpus_to_file(texts, input_file)
        
        # Build and execute encoding command
        cmd = self._build_spm_encode_command(input_file, output_file)
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.stderr:
                logger.warning(f"Encoding stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Tokenization failed with return code {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"SentencePiece tokenization failed: {e.stderr}") from e
        
        # Read and parse tokenized output
        tokenized_sentences = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                tokenized_sentences.append(tokens)
        
        logger.info(f"Successfully tokenized {len(tokenized_sentences)} sentences")
        return tokenized_sentences
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Extract vocabulary from trained model.
        
        Returns:
            Dictionary mapping tokens to their IDs
        """
        if not self.model_path:
            raise ValueError("Model must be trained before extracting vocabulary")
            
        vocab_file = self.model_path.replace('.model', '.vocab')
        vocabulary = {}
        
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        token_id = int(parts[1])
                        vocabulary[token] = token_id
        except FileNotFoundError:
            logger.warning(f"Vocabulary file not found: {vocab_file}")
            return {}
        except Exception as e:
            logger.error(f"Error reading vocabulary file: {e}")
            return {}
            
        logger.info(f"Loaded vocabulary with {len(vocabulary)} tokens")
        return vocabulary
    
    def cleanup(self) -> None:
        """Clean up temporary files and directories."""
        try:
            import shutil
            if os.path.exists(self.config.temp_dir):
                shutil.rmtree(self.config.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.config.temp_dir}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


class SentencePieceAnalyzer:
    """Provides analysis utilities for SentencePiece tokenization results."""
    
    @staticmethod
    def analyze_tokenization(tokenized_sentences: List[List[str]], 
                           original_texts: List[str]) -> Dict[str, Any]:
        """
        Analyze tokenization results.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            original_texts: Original input texts
            
        Returns:
            Dictionary with analysis metrics
        """
        total_tokens = sum(len(tokens) for tokens in tokenized_sentences)
        total_chars = sum(len(text) for text in original_texts)
        avg_tokens_per_sentence = total_tokens / len(tokenized_sentences) if tokenized_sentences else 0
        avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
        
        # Calculate compression ratio (characters per token)
        compression_ratio = avg_chars_per_token
        
        # Find unique tokens
        all_tokens = [token for sentence in tokenized_sentences for token in sentence]
        unique_tokens = set(all_tokens)
        
        return {
            "num_sentences": len(tokenized_sentences),
            "total_tokens": total_tokens,
            "unique_tokens": len(unique_tokens),
            "avg_tokens_per_sentence": round(avg_tokens_per_sentence, 2),
            "avg_chars_per_token": round(avg_chars_per_token, 2),
            "compression_ratio": round(compression_ratio, 2),
            "token_types": len(unique_tokens)
        }
    
    @staticmethod
    def print_tokenization_examples(original_texts: List[str], 
                                  tokenized_sentences: List[List[str]], 
                                  num_examples: int = 3) -> None:
        """Print tokenization examples for inspection."""
        logger.info(f"Tokenization Examples (showing {min(num_examples, len(original_texts))}):")
        for i in range(min(num_examples, len(original_texts))):
            logger.info(f"Original: {original_texts[i]}")
            logger.info(f"Tokenized: {' '.join(tokenized_sentences[i])}")
            logger.info("-" * 50)


def create_sample_corpus() -> List[str]:
    """Create a sample corpus for demonstration purposes."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "SentencePiece is an unsupervised text tokenizer and detokenizer.",
        "It implements subword units like Byte Pair Encoding (BPE) and unigram language model.",
        "This library provides a Python wrapper for the original C++ implementation.",
        "Subword tokenization is useful for handling rare words and out-of-vocabulary tokens.",
        "Machine learning models benefit from subword tokenization in many NLP tasks.",
        "The unigram language model is based on a probabilistic approach to tokenization.",
        "BPE merges the most frequent pairs of bytes or characters iteratively.",
        "This is a sample text for demonstrating SentencePiece tokenization.",
        "Natural language processing requires robust tokenization methods.",
        "Subword units help models generalize better to unseen words.",
        "The tokenizer can handle multiple languages and writing systems.",
        "Pre-trained models are available for many languages and domains.",
        "Custom vocabulary sizes can be specified during training.",
        "The library supports both training and inference modes.",
        "Text normalization is applied before tokenization by default.",
        "Special tokens can be added for specific tasks like machine translation.",
        "The model can be saved and loaded for reuse in different applications.",
        "Batch processing is supported for efficient tokenization of large texts.",
        "Integration with deep learning frameworks is straightforward.",
    ]


def deterministic_verification() -> None:
    """Execute deterministic verification of all core components."""
    logger.info("Starting deterministic verification...")
    
    # Set fixed seed for reproducibility
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    
    # Test configuration
    config = SentencePieceConfig(
        vocab_size=1000,
        model_type="unigram",
        character_coverage=0.9995,
        seed=42
    )
    
    # Create sample corpus
    corpus = create_sample_corpus()
    test_texts = [
        "This is a test sentence for tokenization.",
        "Subword tokenization handles rare words effectively.",
        "Machine learning models use tokenization for text processing."
    ]
    
    trainer = SentencePieceTrainer(config)
    
    try:
        # Test training
        model_path = trainer.train(corpus)
        assert os.path.exists(model_path), "Model file should exist"
        logger.info("✓ Model training completed successfully")
        
        # Test tokenization
        tokenized = trainer.tokenize(test_texts)
        assert len(tokenized) == len(test_texts), "Should tokenize all input texts"
        assert all(isinstance(tokens, list) for tokens in tokenized), "All outputs should be token lists"
        logger.info("✓ Tokenization completed successfully")
        
        # Test vocabulary extraction
        vocab = trainer.get_vocabulary()
        assert len(vocab) > 0, "Vocabulary should not be empty"
        logger.info(f"✓ Vocabulary extracted with {len(vocab)} tokens")
        
        # Test analysis
        analysis = SentencePieceAnalyzer.analyze_tokenization(tokenized, test_texts)
        expected_keys = {"num_sentences", "total_tokens", "unique_tokens", "avg_tokens_per_sentence"}
        assert expected_keys.issubset(analysis.keys()), "Analysis should contain all expected metrics"
        logger.info("✓ Analysis completed successfully")
        
        # Print verification results
        logger.info("Deterministic Verification Results:")
        logger.info(f"  - Model type: {config.model_type}")
        logger.info(f"  - Vocab size: {len(vocab)}")
        logger.info(f"  - Tokenized sentences: {len(tokenized)}")
        logger.info(f"  - Average tokens per sentence: {analysis['avg_tokens_per_sentence']}")
        logger.info(f"  - Unique tokens: {analysis['unique_tokens']}")
        
        # Show examples
        SentencePieceAnalyzer.print_tokenization_examples(test_texts, tokenized, num_examples=2)
        
        logger.info("✓ All core components verified successfully")
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        raise
    finally:
        trainer.cleanup()


def main() -> None:
    """Main execution function."""
    # Configuration for production use
    config = SentencePieceConfig(
        vocab_size=8000,
        model_type="bpe",  # Try "unigram" for alternative approach
        character_coverage=1.0,
        seed=42
    )
    
    # Sample corpus - replace with your actual data
    corpus = create_sample_corpus()
    
    trainer = SentencePieceTrainer(config)
    
    try:
        # Train model
        model_path = trainer.train(corpus)
        
        # Tokenize sample texts
        test_texts = [
            "This is an example sentence for tokenization.",
            "Subword tokenization helps with rare words like 'antidisestablishmentarianism'.",
            "The model can handle multiple languages: こんにちは, Hello, Bonjour!"
        ]
        
        tokenized = trainer.tokenize(test_texts)
        
        # Analyze results
        analysis = SentencePieceAnalyzer.analyze_tokenization(tokenized, test_texts)
        
        logger.info("Tokenization Analysis:")
        for key, value in analysis.items():
            logger.info(f"  {key}: {value}")
        
        # Show examples
        SentencePieceAnalyzer.print_tokenization_examples(test_texts, tokenized)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    # Run deterministic verification when executed directly
    deterministic_verification()
    
    # Uncomment to run main production pipeline
    # main()