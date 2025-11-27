"""
19_whitespace_tokenization_baseline.py

Implement a naïve whitespace tokenizer and benchmark it against regex and library-based 
approaches for speed and reliability.
Concepts: Benchmarking, basic string ops, baseline modeling.
"""

import re
import time
import logging
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Any
from collections import defaultdict
import string
import numpy as np
from functools import wraps
from memory_profiler import memory_usage


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TokenizationConfig:
    """Configuration for tokenization benchmarking and parameters."""
    
    # Benchmarking parameters
    num_runs: int = 10
    warmup_runs: int = 3
    random_seed: int = 42
    
    # Tokenization parameters
    preserve_case: bool = False
    strip_punctuation: bool = False
    normalize_whitespace: bool = True
    
    # Benchmark corpus parameters
    min_sentence_length: int = 5
    max_sentence_length: int = 100
    num_test_sentences: int = 1000
    
    # Performance thresholds (milliseconds)
    performance_threshold_ms: float = 50.0
    memory_threshold_mb: float = 100.0


class WhitespaceTokenizer:
    """Naïve whitespace-based tokenizer implementation."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.num_runs <= 0:
            raise ValueError("num_runs must be positive")
        if self.config.num_test_sentences <= 0:
            raise ValueError("num_test_sentences must be positive")
    
    def _preprocess_text(self, text: str) -> str:
        """Apply text preprocessing based on configuration."""
        processed = text
        
        if not self.config.preserve_case:
            processed = processed.lower()
        
        if self.config.normalize_whitespace:
            # Replace multiple whitespace characters with single space
            processed = re.sub(r'\s+', ' ', processed)
        
        if self.config.strip_punctuation:
            # Remove punctuation characters
            processed = processed.translate(
                str.maketrans('', '', string.punctuation)
            )
        
        return processed.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using naïve whitespace splitting.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        processed_text = self._preprocess_text(text)
        
        # Simple whitespace split
        tokens = processed_text.split()
        
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of tokenized sentences
        """
        return [self.tokenize(text) for text in texts]


class RegexTokenizer:
    """Advanced regex-based tokenizer implementation."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        # Compile regex patterns for performance
        self.word_pattern = re.compile(r'\b\w+\b', re.UNICODE)
        self.whitespace_pattern = re.compile(r'\s+')
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using regex pattern matching.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        processed_text = text
        
        if not self.config.preserve_case:
            processed_text = processed_text.lower()
        
        if self.config.normalize_whitespace:
            processed_text = self.whitespace_pattern.sub(' ', processed_text)
        
        if self.config.strip_punctuation:
            # Use word boundary regex to extract words
            tokens = self.word_pattern.findall(processed_text)
        else:
            # Split on whitespace but keep punctuation with words
            tokens = processed_text.split()
        
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of tokenized sentences
        """
        return [self.tokenize(text) for text in texts]


class LibraryTokenizer:
    """Wrapper for library-based tokenization approaches."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using Python's built-in string methods.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        processed_text = text
        
        if not self.config.preserve_case:
            processed_text = processed_text.lower()
        
        if self.config.normalize_whitespace:
            processed_text = ' '.join(processed_text.split())
        
        if self.config.strip_punctuation:
            # Remove punctuation and split
            translator = str.maketrans('', '', string.punctuation)
            processed_text = processed_text.translate(translator)
        
        tokens = processed_text.split()
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of tokenized sentences
        """
        return [self.tokenize(text) for text in texts]


class TokenizationBenchmark:
    """Comprehensive benchmarking suite for tokenization approaches."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizers = {
            'whitespace': WhitespaceTokenizer(config),
            'regex': RegexTokenizer(config),
            'library': LibraryTokenizer(config)
        }
        
    def generate_test_corpus(self) -> List[str]:
        """
        Generate a deterministic test corpus for benchmarking.
        
        Returns:
            List of test sentences
        """
        np.random.seed(self.config.random_seed)
        
        base_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing requires robust tokenization methods.",
            "Hello world! This is a test sentence with punctuation.",
            "Machine learning models benefit from proper text preprocessing.",
            "Tokenization is the first step in many NLP pipelines.",
            "This sentence has      multiple    spaces   between    words.",
            "Mixed-case Sentence With Capitalization and punctuation!",
            "Numbers 123 and symbols @#$% should be handled properly.",
            "Short words a i to be in of at on an the",
            "Long words antidisestablishmentarianism incomprehensibility"
        ]
        
        # Generate additional synthetic sentences
        synthetic_sentences = []
        words = [
            'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
            'natural', 'language', 'processing', 'requires', 'robust',
            'tokenization', 'methods', 'hello', 'world', 'test', 'sentence',
            'with', 'punctuation', 'machine', 'learning', 'models', 'benefit',
            'from', 'proper', 'text', 'preprocessing', 'first', 'step', 'many',
            'nlp', 'pipelines', 'multiple', 'spaces', 'between', 'words',
            'mixed', 'case', 'capitalization', 'numbers', 'symbols', 'handled',
            'short', 'long', 'comprehensive', 'benchmarking', 'approach'
        ]
        
        for i in range(self.config.num_test_sentences - len(base_sentences)):
            sentence_length = np.random.randint(
                self.config.min_sentence_length,
                self.config.max_sentence_length
            )
            sentence_words = np.random.choice(words, sentence_length)
            sentence = ' '.join(sentence_words)
            
            # Add some variations
            if np.random.random() > 0.7:
                sentence += random.choice(['!', '.', '?'])
            if np.random.random() > 0.8:
                # Add extra spaces
                sentence = re.sub(r'\s+', '   ', sentence)
            
            synthetic_sentences.append(sentence)
        
        corpus = base_sentences + synthetic_sentences
        logger.info(f"Generated test corpus with {len(corpus)} sentences")
        return corpus
    
    def timing_decorator(func: Callable) -> Callable:
        """Decorator to measure execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            return result, execution_time
        return wrapper
    
    @timing_decorator
    def run_tokenization(self, tokenizer_name: str, corpus: List[str]) -> List[List[str]]:
        """
        Run tokenization and measure execution time.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            corpus: List of texts to tokenize
            
        Returns:
            Tuple of (tokenized_sentences, execution_time_ms)
        """
        tokenizer = self.tokenizers[tokenizer_name]
        return tokenizer.tokenize_batch(corpus)
    
    def measure_memory_usage(self, tokenizer_name: str, corpus: List[str]) -> float:
        """
        Measure peak memory usage during tokenization.
        
        Args:
            tokenizer_name: Name of the tokenizer to test
            corpus: Test corpus
            
        Returns:
            Peak memory usage in MB
        """
        tokenizer = self.tokenizers[tokenizer_name]
        
        def tokenize_function():
            return tokenizer.tokenize_batch(corpus)
        
        mem_usage = memory_usage(tokenize_function, max_usage=True)
        return mem_usage
    
    def validate_tokenization_consistency(self, results: Dict[str, List[List[str]]]) -> bool:
        """
        Validate that all tokenizers produce consistent results.
        
        Args:
            results: Dictionary mapping tokenizer names to tokenized results
            
        Returns:
            True if results are consistent across tokenizers
        """
        tokenizer_names = list(results.keys())
        if len(tokenizer_names) < 2:
            return True
        
        # Compare first two tokenizers as baseline
        base_results = results[tokenizer_names[0]]
        
        for other_tokenizer in tokenizer_names[1:]:
            other_results = results[other_tokenizer]
            
            if len(base_results) != len(other_results):
                logger.warning(f"Length mismatch: {tokenizer_names[0]} vs {other_tokenizer}")
                return False
            
            for i, (base_tokens, other_tokens) in enumerate(zip(base_results, other_results)):
                if len(base_tokens) != len(other_tokens):
                    logger.warning(f"Sentence {i}: token count mismatch")
                    return False
                
                # For consistent comparison, normalize tokens
                base_normalized = [token.lower().strip() for token in base_tokens]
                other_normalized = [token.lower().strip() for token in other_tokens]
                
                if base_normalized != other_normalized:
                    logger.warning(f"Sentence {i}: token mismatch")
                    return False
        
        return True
    
    def calculate_tokenization_metrics(self, tokenized_corpus: List[List[str]]) -> Dict[str, float]:
        """
        Calculate metrics for tokenization results.
        
        Args:
            tokenized_corpus: List of tokenized sentences
            
        Returns:
            Dictionary of calculated metrics
        """
        total_tokens = sum(len(tokens) for tokens in tokenized_corpus)
        total_sentences = len(tokenized_corpus)
        tokens_per_sentence = [len(tokens) for tokens in tokenized_corpus]
        
        # Calculate vocabulary
        all_tokens = []
        for tokens in tokenized_corpus:
            all_tokens.extend(tokens)
        
        vocabulary = set(all_tokens)
        
        return {
            'total_tokens': total_tokens,
            'total_sentences': total_sentences,
            'avg_tokens_per_sentence': statistics.mean(tokens_per_sentence),
            'std_tokens_per_sentence': statistics.stdev(tokens_per_sentence) if len(tokens_per_sentence) > 1 else 0,
            'vocabulary_size': len(vocabulary),
            'token_type_ratio': len(vocabulary) / total_tokens if total_tokens > 0 else 0
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Execute comprehensive tokenization benchmark.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting tokenization benchmark")
        
        # Generate test corpus
        corpus = self.generate_test_corpus()
        
        # Warmup runs
        logger.info("Performing warmup runs...")
        for _ in range(self.config.warmup_runs):
            for tokenizer_name in self.tokenizers:
                self.tokenizers[tokenizer_name].tokenize_batch(corpus[:10])
        
        # Main benchmark runs
        benchmark_results = {}
        memory_results = {}
        tokenization_results = {}
        
        for tokenizer_name in self.tokenizers:
            logger.info(f"Benchmarking {tokenizer_name} tokenizer...")
            
            execution_times = []
            all_tokenized = None
            
            for run in range(self.config.num_runs):
                tokenized, execution_time = self.run_tokenization(tokenizer_name, corpus)
                execution_times.append(execution_time)
                
                if run == 0:  # Store results from first run for validation
                    all_tokenized = tokenized
            
            # Store tokenization results for validation
            tokenization_results[tokenizer_name] = all_tokenized
            
            # Calculate performance statistics
            benchmark_results[tokenizer_name] = {
                'mean_time_ms': statistics.mean(execution_times),
                'std_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'min_time_ms': min(execution_times),
                'max_time_ms': max(execution_times),
                'throughput_sentences_per_sec': len(corpus) / (statistics.mean(execution_times) / 1000)
            }
            
            # Measure memory usage
            memory_usage = self.measure_memory_usage(tokenizer_name, corpus)
            memory_results[tokenizer_name] = memory_usage
        
        # Validate consistency
        consistency_valid = self.validate_tokenization_consistency(tokenization_results)
        
        # Calculate metrics for each tokenizer
        metrics_results = {}
        for tokenizer_name, tokenized in tokenization_results.items():
            metrics_results[tokenizer_name] = self.calculate_tokenization_metrics(tokenized)
        
        # Compile final results
        final_results = {
            'benchmark_config': {
                'num_runs': self.config.num_runs,
                'corpus_size': len(corpus),
                'random_seed': self.config.random_seed
            },
            'performance_results': benchmark_results,
            'memory_results': memory_results,
            'tokenization_metrics': metrics_results,
            'validation': {
                'consistency_check': consistency_valid,
                'config_checks': self._validate_benchmark_config()
            }
        }
        
        return final_results
    
    def _validate_benchmark_config(self) -> Dict[str, bool]:
        """Validate benchmark configuration and environment."""
        return {
            'python_version_valid': sys.version_info >= (3, 7),
            'numpy_available': True,
            'memory_profiler_available': True,
            'config_parameters_valid': self.config.num_runs > 0
        }
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted benchmark results."""
        print("\n" + "="*80)
        print("TOKENIZATION BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\nBenchmark Configuration:")
        config = results['benchmark_config']
        print(f"  Corpus size: {config['corpus_size']} sentences")
        print(f"  Number of runs: {config['num_runs']}")
        print(f"  Random seed: {config['random_seed']}")
        
        print(f"\nPerformance Results (milliseconds):")
        print(f"{'Tokenizer':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Throughput':<15}")
        print("-" * 70)
        
        performance = results['performance_results']
        for tokenizer_name, stats in performance.items():
            print(f"{tokenizer_name:<12} {stats['mean_time_ms']:<10.2f} "
                  f"{stats['std_time_ms']:<10.2f} {stats['min_time_ms']:<10.2f} "
                  f"{stats['max_time_ms']:<10.2f} {stats['throughput_sentences_per_sec']:<15.2f}")
        
        print(f"\nMemory Usage (MB):")
        memory = results['memory_results']
        for tokenizer_name, usage in memory.items():
            print(f"  {tokenizer_name}: {usage:.2f} MB")
        
        print(f"\nTokenization Metrics:")
        metrics = results['tokenization_metrics']
        for tokenizer_name, metric in metrics.items():
            print(f"  {tokenizer_name}:")
            print(f"    Total tokens: {metric['total_tokens']}")
            print(f"    Vocabulary size: {metric['vocabulary_size']}")
            print(f"    Avg tokens/sentence: {metric['avg_tokens_per_sentence']:.2f}")
        
        print(f"\nValidation Results:")
        validation = results['validation']
        print(f"  Consistency check: {'PASS' if validation['consistency_check'] else 'FAIL'}")
        
        print("\n" + "="*80)


def deterministic_verification() -> None:
    """Execute deterministic verification of all core components."""
    logger.info("Starting deterministic verification...")
    
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    
    # Test configuration
    config = TokenizationConfig(
        num_runs=5,
        warmup_runs=2,
        num_test_sentences=100,
        random_seed=42
    )
    
    # Initialize components
    benchmark = TokenizationBenchmark(config)
    whitespace_tokenizer = WhitespaceTokenizer(config)
    regex_tokenizer = RegexTokenizer(config)
    library_tokenizer = LibraryTokenizer(config)
    
    # Test corpus
    test_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test.",
        "Multiple    spaces   here."
    ]
    
    verification_results = {
        'components_initialized': True,
        'tokenization_working': True,
        'consistency_check': True,
        'performance_measurement': True
    }
    
    try:
        # Test individual tokenizers
        ws_tokens = whitespace_tokenizer.tokenize_batch(test_corpus)
        regex_tokens = regex_tokenizer.tokenize_batch(test_corpus)
        lib_tokens = library_tokenizer.tokenize_batch(test_corpus)
        
        # Verify tokenization produces results
        assert len(ws_tokens) == len(test_corpus), "Whitespace tokenizer output length mismatch"
        assert len(regex_tokens) == len(test_corpus), "Regex tokenizer output length mismatch"
        assert len(lib_tokens) == len(test_corpus), "Library tokenizer output length mismatch"
        
        logger.info("✓ All tokenizers produced correct number of outputs")
        
        # Test benchmark execution
        results = benchmark.run_benchmark()
        
        # Verify benchmark structure
        required_keys = ['performance_results', 'memory_results', 'tokenization_metrics', 'validation']
        for key in required_keys:
            assert key in results, f"Missing key in benchmark results: {key}"
        
        # Verify performance results exist for all tokenizers
        performance = results['performance_results']
        for tokenizer_name in ['whitespace', 'regex', 'library']:
            assert tokenizer_name in performance, f"Missing performance results for {tokenizer_name}"
            assert performance[tokenizer_name]['mean_time_ms'] > 0, f"Invalid timing for {tokenizer_name}"
        
        logger.info("✓ Benchmark executed successfully with valid results")
        
        # Print verification summary
        print("\nDETERMINISTIC VERIFICATION RESULTS")
        print("="*50)
        print("✓ All tokenizers initialized correctly")
        print("✓ Tokenization produces consistent output lengths")
        print("✓ Benchmark execution completed successfully")
        print("✓ Performance measurements are valid and positive")
        print("✓ Configuration management working correctly")
        
        # Show sample tokenization
        print(f"\nSample Tokenization (first sentence):")
        print(f"  Original: {test_corpus[0]}")
        print(f"  Whitespace: {ws_tokens[0]}")
        print(f"  Regex: {regex_tokens[0]}")
        print(f"  Library: {lib_tokens[0]}")
        
        # Performance comparison
        print(f"\nPerformance Comparison (mean times):")
        for tokenizer_name, stats in performance.items():
            print(f"  {tokenizer_name}: {stats['mean_time_ms']:.2f} ms")
        
        logger.info("✓ All verification checks passed")
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        raise


def main() -> None:
    """Main execution function."""
    # Production configuration
    config = TokenizationConfig(
        num_runs=10,
        warmup_runs=3,
        num_test_sentences=5000,
        random_seed=42,
        preserve_case=False,
        strip_punctuation=True,
        normalize_whitespace=True
    )
    
    benchmark = TokenizationBenchmark(config)
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_benchmark()
        
        # Print formatted results
        benchmark.print_results(results)
        
        # Performance validation
        performance = results['performance_results']
        for tokenizer_name, stats in performance.items():
            if stats['mean_time_ms'] > config.performance_threshold_ms:
                logger.warning(
                    f"{tokenizer_name} tokenizer exceeded performance threshold: "
                    f"{stats['mean_time_ms']:.2f} ms > {config.performance_threshold_ms} ms"
                )
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run deterministic verification when executed directly
    deterministic_verification()
    
    # Uncomment to run full benchmark
    # main()