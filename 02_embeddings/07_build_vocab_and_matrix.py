#!/usr/bin/env python3
"""
07_build_vocab_and_matrix.py

Purpose:
    Builds a vocabulary from a text corpus and constructs an embedding matrix
    by aligning it with pre-trained word vectors. This is a crucial step when
    preparing data for neural network models that use pre-trained embeddings.

Concepts:
    1. Vocabulary Building - Creating a mapping from words to indices
    2. Matrix Construction - Building a 2D numpy array of word vectors
    3. Token-to-Index Mapping - Creating bidirectional dictionaries (word↔index)

Key Operations:
    - Counting word frequencies in a corpus
    - Creating vocabulary with special tokens
    - Aligning with pre-trained embeddings
    - Handling out-of-vocabulary (OOV) words
    - Building the embedding matrix

Dependencies:
    - numpy: For matrix operations
    - collections.Counter: For efficient frequency counting
    - pickle: For saving vocabulary objects (optional)
"""

import numpy as np
from collections import Counter
import pickle
import sys
from typing import List, Dict, Tuple, Optional, Set
import json


class VocabularyBuilder:
    """
    A class to build and manage vocabulary from text data.
    
    Think of a vocabulary as the "dictionary" that your model understands.
    Each unique word gets a unique ID (index), similar to how a real dictionary
    has page numbers for each word.
    
    Why build a vocabulary?
    1. Neural networks work with numbers, not words
    2. We need a consistent mapping: word → index → embedding vector
    3. It reduces memory by replacing strings with integers
    """
    
    def __init__(self, 
                 min_frequency: int = 1,
                 max_vocab_size: Optional[int] = None,
                 special_tokens: List[str] = None):
        """
        Initialize the vocabulary builder with parameters.
        
        Parameters:
        -----------
        min_frequency : int
            Minimum number of times a word must appear to be included.
            This filters out rare words that might be noise or typos.
            
        max_vocab_size : Optional[int]
            Maximum number of words to include in vocabulary.
            If None, include all words above min_frequency.
            Useful for memory constraints.
            
        special_tokens : List[str]
            Special tokens to add to vocabulary (e.g., [PAD], [UNK]).
            These get the first indices (0, 1, 2, ...).
        """
        
        # Store parameters
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        
        # Default special tokens if none provided
        # [PAD] = Padding token (for sequences of different lengths)
        # [UNK] = Unknown word token (for words not in vocabulary)
        # [BOS] = Beginning of sentence
        # [EOS] = End of sentence
        if special_tokens is None:
            self.special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        else:
            self.special_tokens = special_tokens
        
        # Initialize vocabulary storage
        self.word_to_index: Dict[str, int] = {}  # Word → Index
        self.index_to_word: Dict[int, str] = {}  # Index → Word
        self.word_counts: Counter = Counter()    # Word → Frequency
        
        # Statistics
        self.vocab_size: int = 0
        self.total_tokens: int = 0
        self.oov_count: int = 0  # Out-Of-Vocabulary count
        
    def build_from_corpus(self, corpus: List[str]) -> None:
        """
        Build vocabulary from a list of text documents.
        
        Process:
        1. Count word frequencies across entire corpus
        2. Filter by min_frequency
        3. Sort by frequency (most common first)
        4. Apply max_vocab_size limit
        5. Create word↔index mappings
        
        Parameters:
        -----------
        corpus : List[str]
            List of text documents (sentences, paragraphs, etc.)
        """
        
        print("="*70)
        print("BUILDING VOCABULARY FROM CORPUS")
        print("="*70)
        
        # ========================================================================
        # Step 1: Count word frequencies
        # ========================================================================
        print("\n1. Counting word frequencies...")
        
        # Reset counts
        self.word_counts = Counter()
        self.total_tokens = 0
        
        # Process each document in the corpus
        for i, document in enumerate(corpus):
            # Simple tokenization: split by whitespace
            # In practice, you might use more sophisticated tokenization
            tokens = document.lower().split()  # Lowercase for consistency
            
            # Update counts
            self.word_counts.update(tokens)
            self.total_tokens += len(tokens)
            
            # Progress reporting for large corpora
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1:,} documents, {len(self.word_counts):,} unique words")
        
        print(f"  Total unique words: {len(self.word_counts):,}")
        print(f"  Total tokens: {self.total_tokens:,}")
        
        # ========================================================================
        # Step 2: Filter by minimum frequency
        # ========================================================================
        print(f"\n2. Filtering words (min frequency ≥ {self.min_frequency})...")
        
        # Create filtered word list
        filtered_words = [(word, count) for word, count in self.word_counts.items() 
                         if count >= self.min_frequency]
        
        print(f"  Words after filtering: {len(filtered_words):,}")
        print(f"  Words filtered out: {len(self.word_counts) - len(filtered_words):,}")
        
        # ========================================================================
        # Step 3: Sort by frequency (most common first)
        # ========================================================================
        print("\n3. Sorting by frequency (most common → least)...")
        
        # Sort by count in descending order
        # This ensures common words get lower indices (better for some algorithms)
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 10 most frequent words
        print(f"  Top 10 words: {[word for word, _ in filtered_words[:10]]}")
        
        # ========================================================================
        # Step 4: Apply vocabulary size limit
        # ========================================================================
        if self.max_vocab_size is not None:
            print(f"\n4. Limiting to {self.max_vocab_size:,} most frequent words...")
            
            # Take only the top N words
            if len(filtered_words) > self.max_vocab_size:
                filtered_words = filtered_words[:self.max_vocab_size]
                print(f"  Limited to {len(filtered_words):,} words")
        
        # ========================================================================
        # Step 5: Create vocabulary mappings
        # ========================================================================
        print("\n5. Creating vocabulary mappings...")
        
        # Start with special tokens
        # These get indices 0, 1, 2, ...
        current_index = 0
        
        # Add special tokens first (they always get the first indices)
        for token in self.special_tokens:
            self.word_to_index[token] = current_index
            self.index_to_word[current_index] = token
            current_index += 1
        
        # Add regular vocabulary words
        for word, count in filtered_words:
            self.word_to_index[word] = current_index
            self.index_to_word[current_index] = word
            current_index += 1
        
        # Update vocabulary size
        self.vocab_size = current_index
        
        print(f"  Total vocabulary size: {self.vocab_size:,}")
        print(f"  Special tokens: {self.special_tokens}")
        
        # ========================================================================
        # Step 6: Calculate coverage statistics
        # ========================================================================
        print("\n6. Calculating coverage statistics...")
        
        # Calculate what percentage of tokens are covered by our vocabulary
        covered_count = sum(count for word, count in self.word_counts.items() 
                          if word in self.word_to_index)
        coverage = (covered_count / self.total_tokens) * 100
        
        print(f"  Token coverage: {coverage:.2f}%")
        print(f"  {covered_count:,} of {self.total_tokens:,} tokens covered")
    
    def word_to_idx(self, word: str) -> int:
        """
        Convert a word to its index.
        
        This is the core lookup function. If the word isn't in vocabulary,
        return the index for [UNK] (unknown token).
        
        Parameters:
        -----------
        word : str
            The word to look up
            
        Returns:
        --------
        int
            Index of the word, or index of [UNK] if word not found
        """
        
        # Convert to lowercase for consistency
        word_lower = word.lower()
        
        # Look up word
        if word_lower in self.word_to_index:
            return self.word_to_index[word_lower]
        else:
            # Word not in vocabulary → use unknown token
            self.oov_count += 1
            return self.word_to_index['[UNK]']
    
    def idx_to_word(self, idx: int) -> str:
        """
        Convert an index back to its word.
        
        This is the inverse of word_to_idx().
        
        Parameters:
        -----------
        idx : int
            Index to look up
            
        Returns:
        --------
        str
            Word at that index, or [UNK] if index out of range
        """
        
        if idx in self.index_to_word:
            return self.index_to_word[idx]
        else:
            return '[UNK]'  # Default for invalid indices
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary to disk for later use.
        
        Why save vocabulary?
        1. Need same vocabulary during training and inference
        2. Can share vocabulary between different models
        3. Avoid rebuilding from scratch
        
        Parameters:
        -----------
        filepath : str
            Path to save vocabulary file
        """
        
        # Create dictionary with all vocabulary data
        vocab_data = {
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'word_counts': dict(self.word_counts),
            'vocab_size': self.vocab_size,
            'total_tokens': self.total_tokens,
            'special_tokens': self.special_tokens,
            'min_frequency': self.min_frequency,
            'max_vocab_size': self.max_vocab_size
        }
        
        # Save as pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {filepath}")
        
        # Also save as human-readable JSON
        json_path = filepath.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            # Convert to JSON-serializable format
            json_data = {
                'vocab_size': self.vocab_size,
                'total_tokens': self.total_tokens,
                'special_tokens': self.special_tokens,
                'min_frequency': self.min_frequency,
                'max_vocab_size': self.max_vocab_size,
                'sample_words': {k: v for i, (k, v) in enumerate(self.word_counts.items()) 
                               if i < 100}  # First 100 words as sample
            }
            json.dump(json_data, f, indent=2)
        
        print(f"JSON summary saved to {json_path}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to vocabulary file
        """
        
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Restore vocabulary state
        self.word_to_index = vocab_data['word_to_index']
        self.index_to_word = vocab_data['index_to_word']
        self.word_counts = Counter(vocab_data['word_counts'])
        self.vocab_size = vocab_data['vocab_size']
        self.total_tokens = vocab_data['total_tokens']
        self.special_tokens = vocab_data['special_tokens']
        self.min_frequency = vocab_data['min_frequency']
        self.max_vocab_size = vocab_data['max_vocab_size']
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {self.vocab_size:,}")


class EmbeddingMatrixBuilder:
    """
    Builds an embedding matrix by aligning vocabulary with pre-trained vectors.
    
    An embedding matrix is a 2D array where:
    - Rows correspond to word indices (0, 1, 2, ...)
    - Columns are embedding dimensions (e.g., 300 for Word2Vec)
    
    This matrix is typically the first layer in a neural network.
    """
    
    def __init__(self, 
                 embedding_dim: int = 300,
                 initialization: str = 'random'):
        """
        Initialize the embedding matrix builder.
        
        Parameters:
        -----------
        embedding_dim : int
            Dimensionality of word vectors
            Must match the pre-trained embeddings you're using
            
        initialization : str
            How to initialize vectors for OOV words:
            - 'random': Random normal distribution
            - 'zeros': All zeros
            - 'uniform': Uniform distribution [-0.1, 0.1]
        """
        
        self.embedding_dim = embedding_dim
        self.initialization = initialization
        
        # Storage for the matrix
        self.embedding_matrix: Optional[np.ndarray] = None
        
        # Statistics
        self.covered_count: int = 0
        self.oov_count: int = 0
    
    def build_matrix(self, 
                    vocabulary: VocabularyBuilder,
                    pre_trained_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build embedding matrix by aligning vocabulary with pre-trained vectors.
        
        Process:
        1. Create empty matrix of shape (vocab_size, embedding_dim)
        2. For each word in vocabulary:
           - If word has pre-trained vector: copy it
           - If word is OOV: initialize randomly
        3. Special tokens get special initialization
        
        Parameters:
        -----------
        vocabulary : VocabularyBuilder
            Vocabulary containing word→index mappings
            
        pre_trained_vectors : Dict[str, np.ndarray]
            Dictionary of pre-trained word vectors
            Format: {'word': np.array([dim1, dim2, ...])}
            
        Returns:
        --------
        np.ndarray
            Embedding matrix of shape (vocab_size, embedding_dim)
        """
        
        print("\n" + "="*70)
        print("BUILDING EMBEDDING MATRIX")
        print("="*70)
        
        # ========================================================================
        # Step 1: Initialize empty matrix
        # ========================================================================
        print(f"\n1. Initializing matrix: shape = ({vocabulary.vocab_size}, {self.embedding_dim})")
        
        # Initialize matrix with zeros
        # We'll fill it row by row
        self.embedding_matrix = np.zeros((vocabulary.vocab_size, self.embedding_dim))
        
        # ========================================================================
        # Step 2: Initialize special tokens
        # ========================================================================
        print("\n2. Initializing special tokens...")
        
        for token in vocabulary.special_tokens:
            idx = vocabulary.word_to_index[token]
            
            if token == '[PAD]':
                # Padding token: all zeros (so it doesn't affect calculations)
                self.embedding_matrix[idx] = np.zeros(self.embedding_dim)
            elif token == '[UNK]':
                # Unknown token: small random values
                self.embedding_matrix[idx] = self._initialize_vector()
            else:
                # Other special tokens: small random values
                self.embedding_matrix[idx] = self._initialize_vector()
        
        # ========================================================================
        # Step 3: Align with pre-trained vectors
        # ========================================================================
        print("\n3. Aligning with pre-trained vectors...")
        
        self.covered_count = 0
        self.oov_count = 0
        
        # Process each word in vocabulary
        for word, idx in vocabulary.word_to_index.items():
            # Skip special tokens (already initialized)
            if word in vocabulary.special_tokens:
                continue
            
            # Check if word exists in pre-trained vectors
            if word in pre_trained_vectors:
                # Word has a pre-trained vector
                vector = pre_trained_vectors[word]
                
                # Verify dimension matches
                if len(vector) != self.embedding_dim:
                    print(f"  Warning: Dimension mismatch for '{word}': "
                          f"expected {self.embedding_dim}, got {len(vector)}")
                    # Use initialization instead
                    self.embedding_matrix[idx] = self._initialize_vector()
                    self.oov_count += 1
                else:
                    # Copy the pre-trained vector
                    self.embedding_matrix[idx] = vector
                    self.covered_count += 1
            else:
                # Word not in pre-trained vectors → initialize
                self.embedding_matrix[idx] = self._initialize_vector()
                self.oov_count += 1
        
        # ========================================================================
        # Step 4: Calculate statistics
        # ========================================================================
        print("\n4. Calculating alignment statistics...")
        
        coverage = (self.covered_count / (vocabulary.vocab_size - len(vocabulary.special_tokens))) * 100
        
        print(f"  Words covered by pre-trained vectors: {self.covered_count:,}")
        print(f"  Words initialized randomly (OOV): {self.oov_count:,}")
        print(f"  Coverage: {coverage:.2f}%")
        
        # ========================================================================
        # Step 5: Normalize vectors (optional but recommended)
        # ========================================================================
        print("\n5. Normalizing vectors to unit length...")
        
        # Normalize each row (word vector) to have unit length
        # This helps with cosine similarity calculations
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero for zero vectors
        self.embedding_matrix = self.embedding_matrix / norms
        
        print("  Normalization complete")
        
        return self.embedding_matrix
    
    def _initialize_vector(self) -> np.ndarray:
        """
        Initialize a vector based on the chosen initialization strategy.
        
        Returns:
        --------
        np.ndarray
            Initialized vector of shape (embedding_dim,)
        """
        
        if self.initialization == 'zeros':
            return np.zeros(self.embedding_dim)
        elif self.initialization == 'random':
            # Small random values from normal distribution
            return np.random.randn(self.embedding_dim) * 0.01
        elif self.initialization == 'uniform':
            # Uniform distribution
            return np.random.uniform(-0.1, 0.1, self.embedding_dim)
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")
    
    def save_matrix(self, filepath: str) -> None:
        """
        Save embedding matrix to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save matrix (numpy .npy format)
        """
        
        if self.embedding_matrix is None:
            raise ValueError("Matrix not built yet. Call build_matrix() first.")
        
        np.save(filepath, self.embedding_matrix)
        print(f"Embedding matrix saved to {filepath}")
        
        # Also save metadata
        metadata = {
            'shape': self.embedding_matrix.shape,
            'covered_count': self.covered_count,
            'oov_count': self.oov_count,
            'embedding_dim': self.embedding_dim,
            'initialization': self.initialization
        }
        
        meta_path = filepath.replace('.npy', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {meta_path}")


def demonstrate_workflow():
    """
    Demonstrates the complete vocabulary and embedding matrix building workflow.
    
    This example shows:
    1. Creating a sample corpus
    2. Building vocabulary
    3. Creating sample pre-trained vectors
    4. Building embedding matrix
    5. Using the results
    """
    
    print("="*70)
    print("DEMONSTRATION: Complete Vocabulary & Embedding Matrix Workflow")
    print("="*70)
    
    # ============================================================================
    # Step 1: Create sample corpus
    # ============================================================================
    print("\n" + "-"*40)
    print("STEP 1: Sample Corpus")
    print("-"*40)
    
    # Create a small sample corpus
    # In practice, this would be much larger (thousands/millions of documents)
    sample_corpus = [
        "natural language processing is a fascinating field",
        "machine learning models need lots of data",
        "deep learning uses neural networks with many layers",
        "word embeddings represent words as vectors",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming industries",
        "vocabulary building is an essential preprocessing step",
        "embedding matrices align words with vector representations"
    ]
    
    print(f"Corpus size: {len(sample_corpus)} documents")
    print("Sample documents:")
    for i, doc in enumerate(sample_corpus[:3]):
        print(f"  {i+1}. {doc}")
    
    # ============================================================================
    # Step 2: Build vocabulary
    # ============================================================================
    print("\n" + "-"*40)
    print("STEP 2: Building Vocabulary")
    print("-"*40)
    
    # Initialize vocabulary builder
    # Let's set min_frequency=2 to filter out rare words
    vocab_builder = VocabularyBuilder(
        min_frequency=1,      # Include words that appear at least once
        max_vocab_size=50,    # Limit to 50 most frequent words
        special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    )
    
    # Build vocabulary from corpus
    vocab_builder.build_from_corpus(sample_corpus)
    
    # Test vocabulary lookup
    test_words = ["natural", "language", "xyzzy", "[PAD]"]
    print(f"\nVocabulary lookup test:")
    for word in test_words:
        idx = vocab_builder.word_to_idx(word)
        word_back = vocab_builder.idx_to_word(idx)
        print(f"  '{word}' → index {idx} → '{word_back}'")
    
    # ============================================================================
    # Step 3: Create sample pre-trained vectors
    # ============================================================================
    print("\n" + "-"*40)
    print("STEP 3: Sample Pre-trained Vectors")
    print("-"*40)
    
    # In practice, you would load real pre-trained vectors (Word2Vec, GloVe, etc.)
    # For this demo, we'll create synthetic vectors
    
    # Define which words have pre-trained vectors
    # Some words will be "covered", others will be "OOV"
    words_with_vectors = {
        "natural", "language", "processing", "machine", "learning",
        "deep", "neural", "networks", "word", "embeddings",
        "artificial", "intelligence", "vocabulary", "matrices"
    }
    
    # Create synthetic vectors (300-dimensional, like Word2Vec)
    print(f"Creating synthetic vectors for {len(words_with_vectors)} words...")
    
    pre_trained_vectors = {}
    np.random.seed(42)  # For reproducibility
    
    for word in words_with_vectors:
        # Create random vector (in practice, these would be meaningful embeddings)
        vector = np.random.randn(300)  # 300 dimensions
        # Normalize to unit length (common practice)
        vector = vector / np.linalg.norm(vector)
        pre_trained_vectors[word] = vector
    
    print(f"Pre-trained vectors created: {len(pre_trained_vectors)} words")
    
    # ============================================================================
    # Step 4: Build embedding matrix
    # ============================================================================
    print("\n" + "-"*40)
    print("STEP 4: Building Embedding Matrix")
    print("-"*40)
    
    # Initialize matrix builder
    matrix_builder = EmbeddingMatrixBuilder(
        embedding_dim=300,    # Must match pre-trained vectors
        initialization='random'
    )
    
    # Build the matrix
    embedding_matrix = matrix_builder.build_matrix(vocab_builder, pre_trained_vectors)
    
    # ============================================================================
    # Step 5: Inspect results
    # ============================================================================
    print("\n" + "-"*40)
    print("STEP 5: Results Inspection")
    print("-"*40)
    
    print(f"\nMatrix shape: {embedding_matrix.shape}")
    print(f"  - {embedding_matrix.shape[0]} rows (words in vocabulary)")
    print(f"  - {embedding_matrix.shape[1]} columns (embedding dimensions)")
    
    # Show some example vectors
    print("\nExample vectors (first 5 dimensions):")
    example_words = ["natural", "[PAD]", "[UNK]", "xyzzy"]
    
    for word in example_words:
        idx = vocab_builder.word_to_idx(word)
        vector = embedding_matrix[idx]
        vector_str = ', '.join([f"{x:.4f}" for x in vector[:5]])
        status = "pre-trained" if word in pre_trained_vectors else "initialized"
        print(f"  '{word}' (index {idx}, {status}): [{vector_str} ...]")
    
    # Calculate average similarity between covered words
    print("\nCalculating similarity between covered words...")
    
    covered_indices = [vocab_builder.word_to_idx(word) 
                      for word in words_with_vectors 
                      if word in vocab_builder.word_to_index]
    
    if len(covered_indices) > 1:
        covered_vectors = embedding_matrix[covered_indices]
        
        # Normalize vectors (already normalized in build_matrix)
        # Calculate average pairwise cosine similarity
        similarities = []
        for i in range(len(covered_indices)):
            for j in range(i + 1, len(covered_indices)):
                sim = np.dot(covered_vectors[i], covered_vectors[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        print(f"  Average cosine similarity: {avg_similarity:.4f}")
    
    # ============================================================================
    # Step 6: Save results (optional)
    # ============================================================================
    print("\n" + "-"*40)
    print("STEP 6: Saving Results")
    print("-"*40)
    
    # Save vocabulary
    vocab_builder.save_vocabulary('vocabulary.pkl')
    
    # Save embedding matrix
    matrix_builder.save_matrix('embedding_matrix.npy')
    
    return vocab_builder, matrix_builder


def practical_tips():
    """
    Provides practical tips for real-world usage.
    """
    
    print("\n" + "="*70)
    print("PRACTICAL TIPS FOR REAL-WORLD USAGE")
    print("="*70)
    
    tips = [
        ("1. Corpus Size", 
         "For production models, use at least 1M+ tokens. Larger corpora capture "
         "better word distributions and rare word occurrences."),
        
        ("2. Minimum Frequency", 
         "Set min_frequency=5-10 to filter out noise and typos. This reduces "
         "vocabulary size and improves model performance."),
        
        ("3. Pre-trained Vectors", 
         "Use established models like Word2Vec, GloVe, or FastText. They capture "
         "semantic relationships from massive corpora."),
        
        ("4. Handling OOV Words", 
         "Consider using subword models (FastText) or character-level embeddings "
         "for better OOV handling."),
        
        ("5. Memory Considerations", 
         f"Embedding matrix size = vocab_size × embedding_dim × 4 bytes (float32). "
         f"100K words × 300 dims = 120MB. Plan accordingly."),
        
        ("6. Special Tokens", 
         "Always include [PAD] and [UNK]. Consider domain-specific special tokens "
         "like [NUM], [URL], or [MENTION] for social media text."),
        
        ("7. Text Preprocessing", 
         "Consistent preprocessing (lowercasing, punctuation handling) is crucial. "
         "It must match the pre-trained vectors' preprocessing."),
        
        ("8. Vocabulary Persistence", 
         "Save and reuse the same vocabulary for training and inference. "
         "Inconsistent mappings break the model.")
    ]
    
    for title, tip in tips:
        print(f"\n{title}:")
        print(f"  {tip}")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    """
    Main execution block.
    
    When run directly:
    1. Demonstrates the complete workflow
    2. Shows practical examples
    3. Provides educational explanations
    """
    
    print(__doc__)
    
    try:
        # Run the demonstration
        vocab_builder, matrix_builder = demonstrate_workflow()
        
        # Show practical tips
        practical_tips()
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETE!")
        print("="*70)
        print("\nYou've successfully:")
        print("1. Built a vocabulary from a text corpus")
        print("2. Created word↔index mappings")
        print("3. Constructed an embedding matrix")
        print("4. Aligned with pre-trained vectors")
        print("\nNext steps:")
        print("1. Use embedding_matrix as first layer in neural network")
        print("2. Use vocabulary to convert text to indices")
        print("3. Train/fine-tune your model")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("\nCommon issues:")
        print("1. Dimension mismatch between vocabulary and pre-trained vectors")
        print("2. Memory issues with large vocabularies")
        print("3. Inconsistent preprocessing between corpus and pre-trained vectors")


# ================================================================================
# ADDITIONAL RESOURCES
# ================================================================================
"""
Additional Learning Resources:

1. Vocabulary Building:
   - The importance of vocabulary in NLP: Vocabulary acts as the interface between
     human language and machine learning algorithms.

2. Embedding Matrices:
   - These are often called "embedding layers" in neural networks.
   - Can be static (frozen) or trainable (fine-tuned).

3. Pre-trained Models:
   - Word2Vec: Google's model, trained on Google News
   - GloVe: Stanford's model, combines matrix factorization with context windows
   - FastText: Facebook's model, includes subword information

4. Practical Considerations:
   - Always lowercase if your pre-trained vectors are lowercase
   - Consider lemmatization/stemming for morphological normalization
   - Handle numbers, URLs, and special symbols consistently

5. Advanced Topics:
   - Dynamic vocabularies for streaming data
   - Vocabulary compression techniques
   - Multi-lingual embedding spaces
"""