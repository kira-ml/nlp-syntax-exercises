import os
import sys
import logging
from typing import (
    Iterator, List, Optional, Union, Tuple, Generator, 
    Callable, Any, Dict, Set, Protocol
)
from pathlib import Path
from abc import ABC, abstractmethod
import mmap
import gc
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from contextlib import contextmanager
import psutil
import numpy as np

# Try to import tokenizers (HuggingFace) or fallback to basic tokenization
try:
    from tokenizers import Tokenizer, Encoding
    from tokenizers.models import BPE, WordPiece
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel
    from tokenizers.processors import TemplateProcessing
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    Tokenizer = Any
    Encoding = Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenizerProtocol(Protocol):
    """Protocol defining the interface for tokenizer implementations."""
    
    def encode(self, text: str, **kwargs) -> Any:
        ...
    
    def decode(self, tokens: Any, **kwargs) -> str:
        ...

class MemoryMonitor:
    """Monitor system memory usage during processing."""
    
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self.initial_memory = self.get_memory_usage()
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds threshold."""
        current_memory = self.get_memory_usage()
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > self.threshold_mb:
            logger.warning(f"Memory pressure detected: {current_memory:.2f} MB "
                          f"(+{memory_increase:.2f} MB)")
            return True
        return False
    
    @contextmanager
    def memory_guard(self):
        """Context manager to monitor memory during operations."""
        self.initial_memory = self.get_memory_usage()
        yield
        if self.check_memory_pressure():
            gc.collect()  # Force garbage collection

class BaseTokenizer(ABC):
    """Abstract base class for tokenizer implementations."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass

class SimpleWordTokenizer(BaseTokenizer):
    """Simple whitespace-based tokenizer for demonstration."""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self._next_id = 0
    
    def _get_or_create_token_id(self, token: str) -> int:
        """Get existing token ID or create new one."""
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self.reverse_vocab[self._next_id] = token
            self._next_id += 1
        return self.vocab[token]
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = text.strip().split()
        return [self._get_or_create_token_id(token) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return ' '.join(self.reverse_vocab.get(tid, '<UNK>') for tid in token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

class HuggingFaceTokenizerWrapper(BaseTokenizer):
    """Wrapper for HuggingFace tokenizers with streaming support."""
    
    def __init__(self, tokenizer: Tokenizer):
        if not HAS_TOKENIZERS:
            raise ImportError("HuggingFace tokenizers library not available")
        self.tokenizer = tokenizer
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids if hasattr(encoding, 'ids') else encoding
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

class TextFileStreamer:
    """Stream text files with configurable chunking strategies."""
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 chunk_size: int = 8192,
                 delimiter: str = '\n',
                 encoding: str = 'utf-8'):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.delimiter = delimiter
        self.encoding = encoding
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def stream_lines(self) -> Iterator[str]:
        """Stream file line by line."""
        with open(self.file_path, 'r', encoding=self.encoding, buffering=self.chunk_size) as f:
            for line in f:
                yield line.rstrip('\n\r')
    
    def stream_chunks(self) -> Iterator[str]:
        """Stream file in fixed-size chunks."""
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
    
    def stream_paragraphs(self) -> Iterator[str]:
        """Stream file paragraph by paragraph."""
        current_paragraph = []
        for line in self.stream_lines():
            if line.strip() == '':
                if current_paragraph:
                    yield '\n'.join(current_paragraph)
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Yield last paragraph if exists
        if current_paragraph:
            yield '\n'.join(current_paragraph)
    
    @contextmanager
    def memory_mapped_stream(self) -> Iterator[mmap.mmap]:
        """Memory-mapped file streaming for very large files."""
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                yield mmapped_file

class BatchTokenizer:
    """High-performance batch tokenizer with memory management."""
    
    def __init__(self,
                 tokenizer: BaseTokenizer,
                 batch_size: int = 1000,
                 max_length: Optional[int] = None,
                 pad_token_id: int = 0,
                 memory_threshold_mb: int = 1000,
                 num_workers: int = 1):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.memory_monitor = MemoryMonitor(threshold_mb=memory_threshold_mb)
        self.num_workers = min(num_workers, os.cpu_count() or 1)
        
        logger.info(f"BatchTokenizer initialized with batch_size={batch_size}, "
                   f"workers={self.num_workers}")
    
    def _tokenize_single(self, text: str) -> List[int]:
        """Tokenize a single text sample."""
        try:
            tokens = self.tokenizer.encode(text)
            if self.max_length:
                tokens = tokens[:self.max_length]
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error for text: {text[:100]}... Error: {e}")
            return []
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Tokenize a batch of texts."""
        if self.num_workers > 1 and len(texts) > 10:
            return self._tokenize_parallel(texts)
        else:
            return [self._tokenize_single(text) for text in texts]
    
    def _tokenize_parallel(self, texts: List[str]) -> List[List[int]]:
        """Tokenize texts in parallel using ThreadPoolExecutor."""
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._tokenize_single, text): i 
                for i, text in enumerate(texts)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Parallel tokenization error at index {index}: {e}")
                    results[index] = []
        
        return results
    
    def _pad_batch(self, batch_tokens: List[List[int]]) -> np.ndarray:
        """Pad token sequences to same length."""
        if not batch_tokens:
            return np.array([])
        
        max_len = max(len(tokens) for tokens in batch_tokens) if not self.max_length else self.max_length
        padded_batch = []
        
        for tokens in batch_tokens:
            if len(tokens) < max_len:
                padded_tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
            else:
                padded_tokens = tokens[:max_len]
            padded_batch.append(padded_tokens)
        
        return np.array(padded_batch, dtype=np.int32)
    
    def stream_tokenize(self, 
                       text_stream: Iterator[str],
                       return_tensors: bool = False) -> Iterator[Union[List[List[int]], np.ndarray]]:
        """
        Stream tokenize texts with memory-efficient batching.
        
        Args:
            text_stream: Iterator of text samples
            return_tensors: Whether to return numpy arrays instead of lists
            
        Yields:
            Batches of tokenized texts
        """
        batch_texts: List[str] = []
        processed_count = 0
        batch_count = 0
        
        with self.memory_monitor.memory_guard():
            for text in text_stream:
                if not text.strip():  # Skip empty texts
                    continue
                    
                batch_texts.append(text)
                processed_count += 1
                
                # Process batch when full
                if len(batch_texts) >= self.batch_size:
                    logger.debug(f"Processing batch {batch_count + 1} ({len(batch_texts)} items)")
                    
                    batch_tokens = self.tokenize_batch(batch_texts)
                    batch_tokens = [tokens for tokens in batch_tokens if tokens]  # Remove empty
                    
                    if batch_tokens:
                        if return_tensors:
                            yield self._pad_batch(batch_tokens)
                        else:
                            yield batch_tokens
                    
                    batch_texts = []
                    batch_count += 1
                    
                    # Periodic memory cleanup
                    if batch_count % 10 == 0:
                        if self.memory_monitor.check_memory_pressure():
                            gc.collect()
            
            # Process remaining items
            if batch_texts:
                logger.debug(f"Processing final batch ({len(batch_texts)} items)")
                batch_tokens = self.tokenize_batch(batch_texts)
                batch_tokens = [tokens for tokens in batch_tokens if tokens]
                
                if batch_tokens:
                    if return_tensors:
                        yield self._pad_batch(batch_tokens)
                    else:
                        yield batch_tokens
        
        logger.info(f"Tokenization complete: {processed_count} texts processed in {batch_count + 1} batches")

class TokenizationPipeline:
    """End-to-end tokenization pipeline with multiple input sources."""
    
    def __init__(self, 
                 tokenizer: BaseTokenizer,
                 batch_size: int = 1000,
                 max_length: Optional[int] = None,
                 num_workers: int = 1,
                 memory_threshold_mb: int = 1000):
        self.batch_tokenizer = BatchTokenizer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            memory_threshold_mb=memory_threshold_mb
        )
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    stream_type: str = 'lines',
                    return_tensors: bool = False) -> Iterator[Union[List[List[int]], np.ndarray]]:
        """
        Process a text file with streaming tokenization.
        
        Args:
            file_path: Path to input file
            stream_type: Type of streaming ('lines', 'chunks', 'paragraphs')
            return_tensors: Whether to return numpy arrays
            
        Yields:
            Batches of tokenized texts
        """
        streamer = TextFileStreamer(file_path)
        
        # Select streaming method
        stream_methods = {
            'lines': streamer.stream_lines,
            'chunks': streamer.stream_chunks,
            'paragraphs': streamer.stream_paragraphs
        }
        
        if stream_type not in stream_methods:
            raise ValueError(f"Invalid stream_type: {stream_type}. "
                           f"Choose from {list(stream_methods.keys())}")
        
        text_stream = stream_methods[stream_type]()
        yield from self.batch_tokenizer.stream_tokenize(text_stream, return_tensors)
    
    def process_multiple_files(self, 
                              file_paths: List[Union[str, Path]],
                              stream_type: str = 'lines',
                              return_tensors: bool = False) -> Iterator[Union[List[List[int]], np.ndarray]]:
        """
        Process multiple files sequentially.
        
        Args:
            file_paths: List of file paths
            stream_type: Type of streaming
            return_tensors: Whether to return numpy arrays
            
        Yields:
            Batches of tokenized texts from all files
        """
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            yield from self.process_file(file_path, stream_type, return_tensors)

# Utility functions for common use cases
def create_huggingface_tokenizer(model_name: str = "bert-base-uncased") -> BaseTokenizer:
    """Create a HuggingFace tokenizer if available."""
    if not HAS_TOKENIZERS:
        raise ImportError("HuggingFace tokenizers library required for this function")
    
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HuggingFaceTokenizerWrapper(hf_tokenizer)
    except Exception as e:
        logger.warning(f"Failed to load HuggingFace tokenizer: {e}")
        return SimpleWordTokenizer()

def save_tokenized_batches(batches: Iterator[Union[List[List[int]], np.ndarray]], 
                          output_path: Union[str, Path],
                          format: str = 'npy') -> int:
    """
    Save tokenized batches to disk.
    
    Args:
        batches: Iterator of tokenized batches
        output_path: Output directory path
        format: Output format ('npy', 'txt')
        
    Returns:
        Number of batches saved
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_count = 0
    for i, batch in enumerate(batches):
        batch_count += 1
        if format == 'npy':
            np.save(output_path / f"batch_{i:06d}.npy", batch)
        elif format == 'txt':
            with open(output_path / f"batch_{i:06d}.txt", 'w') as f:
                for sample in batch:
                    f.write(' '.join(map(str, sample)) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {batch_count} batches to {output_path}")
    return batch_count

# Main execution function
def main():
    """Main execution function demonstrating usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch tokenize large text files with memory efficiency")
    parser.add_argument('input_files', nargs='+', help='Input file paths')
    parser.add_argument('--output_dir', required=True, help='Output directory for tokenized batches')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads')
    parser.add_argument('--stream_type', choices=['lines', 'chunks', 'paragraphs'], 
                       default='lines', help='Text streaming method')
    parser.add_argument('--return_tensors', action='store_true', help='Return numpy arrays')
    parser.add_argument('--format', choices=['npy', 'txt'], default='npy', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize tokenizer
        if HAS_TOKENIZERS:
            tokenizer = SimpleWordTokenizer()  # In practice, use create_huggingface_tokenizer()
        else:
            tokenizer = SimpleWordTokenizer()
            logger.info("Using simple word tokenizer (install tokenizers for better performance)")
        
        # Create pipeline
        pipeline = TokenizationPipeline(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.workers
        )
        
        # Process files
        batches = pipeline.process_multiple_files(
            file_paths=args.input_files,
            stream_type=args.stream_type,
            return_tensors=args.return_tensors
        )
        
        # Save results
        saved_count = save_tokenized_batches(batches, args.output_dir, args.format)
        
        print(f"Successfully processed {len(args.input_files)} files")
        print(f"Generated {saved_count} batches in {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

# Example usage and testing
if __name__ == "__main__":
    # Example of direct usage
    if len(sys.argv) == 1:
        # Demo mode
        logger.info("Running in demo mode...")
        
        # Create sample data
        sample_text = "This is a sample sentence for tokenization.\n" * 10000
        with open("sample.txt", "w") as f:
            f.write(sample_text)
        
        # Initialize components
        tokenizer = SimpleWordTokenizer()
        pipeline = TokenizationPipeline(tokenizer, batch_size=100, max_length=50)
        
        # Process with streaming
        batches = list(pipeline.process_file("sample.txt", stream_type='lines'))
        print(f"Processed {len(batches)} batches")
        print(f"First batch shape: {np.array(batches[0]).shape}")
        
        # Cleanup
        os.remove("sample.txt")
    else:
        main()