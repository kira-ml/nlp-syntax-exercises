#!/usr/bin/env python3
"""
06_fasttext_subword_vector_lookup.py

Purpose:
    Demonstrates how to load FastText word embeddings and handle Out-Of-Vocabulary (OOV)
    words using subword composition. FastText can generate vectors for unseen words
    by averaging the embeddings of their character n-grams.

Concepts:
    1. FastText Behavior - How FastText uses subword information (character n-grams)
    2. Fallback Embeddings - Generating vectors for OOV words via subword composition
    3. Gensim API - Using the gensim library to load and query FastText models

Key Features:
    - Load pre-trained FastText models
    - Handle OOV words via subword composition
    - Compare FastText with Word2Vec for OOV handling
    - Visualize subword contributions

Dependencies:
    - gensim: For loading FastText models
    - numpy: For vector operations
    - matplotlib: For visualization (optional)

Input: Pre-trained FastText model file (.bin or .vec format)
Output: Word vectors, including for OOV words
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Union

# Optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available. Visualizations disabled.")

class FastTextVectorLookup:
    """
    A class for loading and querying FastText embeddings with subword support.
    
    FastText differs from Word2Vec by representing words as the sum of their 
    character n-gram vectors. This allows it to:
    1. Generate embeddings for OOV words by composing subword vectors
    2. Capture morphological information through character-level patterns
    3. Handle misspellings and rare words more effectively
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the FastText model from a pre-trained file.
        
        Parameters:
        -----------
        model_path : str
            Path to the FastText model file (.bin or .vec format)
            
        Raises:
        -------
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the file format is unsupported
        """
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Determine file format from extension
        file_ext = os.path.splitext(model_path)[1].lower()
        
        # ========================================================================
        # Load model based on file format
        # ========================================================================
        try:
            from gensim.models import FastText, KeyedVectors
            
            if file_ext == '.bin':
                # .bin files contain the full FastText model with subword information
                # This format supports OOV word vector generation
                print(f"Loading FastText binary model from: {model_path}")
                print("(This may take a few minutes for large models)...")
                
                # FastText class maintains subword information for OOV handling
                self.model = FastText.load_fasttext_format(model_path)
                self.model_type = 'fasttext_binary'
                
            elif file_ext == '.vec':
                # .vec files contain only word vectors without subword information
                # These are essentially Word2Vec format and don't support OOV words
                print(f"Loading FastText vectors from text file: {model_path}")
                print("Note: .vec format lacks subword info - OOV words not supported")
                
                # KeyedVectors loads only the word vectors (no subword info)
                self.model = KeyedVectors.load_word2vec_format(model_path)
                self.model_type = 'fasttext_text'
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. "
                               f"Use .bin (with subwords) or .vec (vectors only)")
                
        except ImportError as e:
            print("Error: gensim library not found. Install with: pip install gensim")
            raise
            
        # ========================================================================
        # Model metadata
        # ========================================================================
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        # Different attributes based on model type
        if self.model_type == 'fasttext_binary':
            # Full FastText model with subword capabilities
            vocab_size = len(self.model.wv)
            vector_dim = self.model.wv.vector_size
            print(f"Model type: FastText (with subword support)")
            print(f"Vocabulary size: {vocab_size:,} words")
            print(f"Vector dimension: {vector_dim}")
            print(f"Minimum n-gram length: {self.model.wv.min_n}")
            print(f"Maximum n-gram length: {self.model.wv.max_n}")
            print(f"Bucket size (n-gram hash table): {self.model.wv.bucket}")
            
        else:  # fasttext_text
            # Just word vectors (like Word2Vec)
            vocab_size = len(self.model)
            vector_dim = self.model.vector_size
            print(f"Model type: FastText vectors (no subword support)")
            print(f"Vocabulary size: {vocab_size:,} words")
            print(f"Vector dimension: {vector_dim}")
            
        print("="*60)
        
    def get_vector(self, word: str, use_subwords: bool = True) -> np.ndarray:
        """
        Retrieve the vector for a word, handling OOV words with subword composition.
        
        FastText's key innovation: If a word is not in the vocabulary, it decomposes
        the word into character n-grams and averages their vectors.
        
        Parameters:
        -----------
        word : str
            The word to get the vector for
        use_subwords : bool
            If True and word is OOV, use subword composition
            If False, return None for OOV words
            
        Returns:
        --------
        np.ndarray
            Word vector (normalized to unit length)
        None
            If word is OOV and use_subwords=False
            
        Example:
        --------
        >>> ft = FastTextVectorLookup('cc.en.300.bin')
        >>> # In-vocabulary word
        >>> vec1 = ft.get_vector('computer')
        >>> # OOV word - will be constructed from subwords
        >>> vec2 = ft.get_vector('computationalization')
        """
        
        # ========================================================================
        # Handle different model types
        # ========================================================================
        
        if self.model_type == 'fasttext_binary':
            # Full FastText model with OOV support
            word = word.lower()  # FastText models are typically lowercase
            
            if use_subwords:
                # This will return a vector even for OOV words by using subwords
                # get_vector() method handles both in-vocab and OOV cases
                vector = self.model.wv[word]
                
                # Note: For OOV words, this returns the sum of character n-gram vectors
                # normalized to unit length
                return vector
                
            else:
                # Only return vector if word is in vocabulary
                if word in self.model.wv:
                    return self.model.wv[word]
                else:
                    return None
                    
        else:  # fasttext_text
            # .vec format - no subword support
            word = word.lower()
            
            if word in self.model:
                return self.model[word]
            elif use_subwords:
                print(f"Warning: '{word}' is OOV and model has no subword support")
                return None
            else:
                return None
    
    def get_similar_words(self, word: str, top_n: int = 10, 
                          use_subwords: bool = True) -> List[Tuple[str, float]]:
        """
        Find words most similar to the given word.
        
        For OOV words, this uses the subword-composed vector to find similar words
        in the vocabulary.
        
        Parameters:
        -----------
        word : str
            Input word (can be OOV)
        top_n : int
            Number of similar words to return
        use_subwords : bool
            Whether to use subwords for OOV words
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (word, similarity_score) pairs
        """
        
        # Get vector for the word (with subword fallback if enabled)
        word_vector = self.get_vector(word, use_subwords)
        
        if word_vector is None:
            print(f"Word '{word}' not found in vocabulary")
            return []
        
        # Find most similar words using cosine similarity
        if self.model_type == 'fasttext_binary':
            similar_words = self.model.wv.most_similar(
                positive=[word_vector], 
                topn=top_n
            )
        else:
            similar_words = self.model.most_similar(
                positive=[word_vector], 
                topn=top_n
            )
            
        return similar_words
    
    def get_subword_ngrams(self, word: str) -> List[str]:
        """
        Extract character n-grams for a word (if model supports it).
        
        FastText represents words as the sum of character n-gram vectors.
        Example: "apple" with n=3 generates: <ap, app, ppl, ple, le>
        
        Parameters:
        -----------
        word : str
            Input word
            
        Returns:
        --------
        List[str]
            List of character n-grams, or empty list if not supported
        """
        
        if self.model_type != 'fasttext_binary':
            print("Model doesn't support subword n-grams (.vec format)")
            return []
        
        # Extract n-grams using FastText's internal method
        # Note: This shows the actual n-grams used by the model
        word = word.lower()
        
        # Check min_n and max_n parameters
        min_n = self.model.wv.min_n
        max_n = self.model.wv.max_n
        
        ngrams = []
        
        # Add word boundary symbols (FastText uses '<' and '>')
        word_with_boundaries = f'<{word}>'
        
        # Generate all n-grams of lengths between min_n and max_n
        for n in range(min_n, max_n + 1):
            for i in range(len(word_with_boundaries) - n + 1):
                ngram = word_with_boundaries[i:i + n]
                ngrams.append(ngram)
                
        return ngrams
    
    def analyze_word(self, word: str) -> Dict:
        """
        Comprehensive analysis of a word's representation in FastText.
        
        Parameters:
        -----------
        word : str
            Word to analyze
            
        Returns:
        --------
        Dict with analysis results
        """
        
        word = word.lower()
        result = {
            'word': word,
            'in_vocabulary': False,
            'vector_norm': None,
            'subword_ngrams': [],
            'similar_words': []
        }
        
        # Check if word is in vocabulary
        if self.model_type == 'fasttext_binary':
            result['in_vocabulary'] = word in self.model.wv
        else:
            result['in_vocabulary'] = word in self.model
        
        # Get vector
        vector = self.get_vector(word, use_subwords=True)
        
        if vector is not None:
            result['vector_norm'] = np.linalg.norm(vector)
            result['vector_shape'] = vector.shape
            
            # Get similar words
            result['similar_words'] = self.get_similar_words(word, top_n=5)
            
            # Get subword n-grams (if supported)
            if self.model_type == 'fasttext_binary':
                result['subword_ngrams'] = self.get_subword_ngrams(word)
        
        return result
    
    def compare_oov_handling(self, words: List[str]) -> None:
        """
        Compare how different models handle OOV words.
        
        Demonstrates FastText's advantage over Word2Vec for rare/misspelled words.
        
        Parameters:
        -----------
        words : List[str]
            List of words to test (including some OOV words)
        """
        
        print("\n" + "="*60)
        print("OOV WORD HANDLING COMPARISON")
        print("="*60)
        
        for word in words:
            print(f"\nWord: '{word}'")
            
            # Try to get vector
            vector = self.get_vector(word, use_subwords=True)
            
            if vector is not None:
                # Word was found (either in vocab or via subwords)
                norm = np.linalg.norm(vector)
                print(f"  ✓ Vector obtained (norm: {norm:.4f})")
                
                # Check if it was in vocabulary
                if self.model_type == 'fasttext_binary':
                    in_vocab = word.lower() in self.model.wv
                else:
                    in_vocab = word.lower() in self.model
                    
                if in_vocab:
                    print("  ✓ Word was in vocabulary")
                else:
                    print("  → Word was OOV, vector created from subwords")
                    
                # Show similar words
                similar = self.get_similar_words(word, top_n=3)
                if similar:
                    sim_str = ', '.join([f'{w} ({s:.3f})' for w, s in similar])
                    print(f"  Similar words: {sim_str}")
            else:
                print(f"  ✗ No vector available (OOV with no subword support)")
    
    def visualize_word_similarities(self, words: List[str]) -> None:
        """
        Create a heatmap of cosine similarities between words.
        
        Parameters:
        -----------
        words : List[str]
            Words to compare
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for visualization")
            return
        
        # Get vectors for all words
        vectors = []
        valid_words = []
        
        for word in words:
            vec = self.get_vector(word, use_subwords=True)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if len(vectors) < 2:
            print("Need at least 2 valid words for visualization")
            return
        
        vectors = np.array(vectors)
        
        # Compute cosine similarity matrix
        # Cosine similarity = dot product of normalized vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(valid_words)))
        ax.set_yticks(range(len(valid_words)))
        ax.set_xticklabels(valid_words, rotation=45, ha='right')
        ax.set_yticklabels(valid_words)
        
        # Add similarity values as text
        for i in range(len(valid_words)):
            for j in range(len(valid_words)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                              ha='center', va='center', color='black',
                              fontsize=9)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Title and labels
        model_type = "with subwords" if self.model_type == 'fasttext_binary' else "vectors only"
        ax.set_title(f'Word Similarities (FastText {model_type})', fontsize=14, pad=20)
        ax.set_xlabel('Words', fontsize=12)
        ax.set_ylabel('Words', fontsize=12)
        
        plt.tight_layout()
        plt.show()

# ================================================================================
# DEMONSTRATION AND USAGE EXAMPLES
# ================================================================================

def demonstrate_fasttext_capabilities():
    """
    Demonstrates key FastText features with examples.
    
    Shows:
    1. OOV word handling
    2. Morphological similarity capture
    3. Misspelling robustness
    """
    
    print("\n" + "="*70)
    print("FASTTEXT DEMONSTRATION: Subword Vector Lookup")
    print("="*70)
    
    # Note: This demo requires a FastText model file
    # Download one from: https://fasttext.cc/docs/en/english-vectors.html
    # Example: 'cc.en.300.bin' (2.6GB) or 'crawl-300d-2M-subword.bin' (6.7GB)
    
    # ============================================================================
    # IMPORTANT: Replace with your actual model path
    # ============================================================================
    model_path = "cc.en.300.bin"  # Change this to your model file
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model file not found: {model_path}")
        print("\nTo run this demo, please:")
        print("1. Download a FastText model (e.g., from fasttext.cc)")
        print("2. Update the model_path variable in the code")
        print("\nExample download command:")
        print("wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")
        print("gunzip cc.en.300.bin.gz")
        return
    
    # ============================================================================
    # Initialize FastText lookup
    # ============================================================================
    print("\nInitializing FastText model...")
    ft_lookup = FastTextVectorLookup(model_path)
    
    # ============================================================================
    # Example 1: In-vocabulary vs OOV words
    # ============================================================================
    print("\n" + "-"*40)
    print("EXAMPLE 1: Vocabulary Lookup")
    print("-"*40)
    
    test_words = [
        "computer",          # Common word (in vocabulary)
        "computational",     # Less common (likely in vocabulary)
        "computationalization",  # Rare/OOV word
        "university",        # Common word
        "universalization",  # OOV word
    ]
    
    for word in test_words:
        result = ft_lookup.analyze_word(word)
        status = "✓ IN VOCAB" if result['in_vocabulary'] else "→ OOV (subwords)"
        print(f"{word:25} {status:20} norm: {result['vector_norm']:.4f}")
        
        # Show similar words if available
        if result['similar_words']:
            similar = ', '.join([w for w, _ in result['similar_words'][:3]])
            print(f"  Similar: {similar}")
    
    # ============================================================================
    # Example 2: Morphological patterns
    # ============================================================================
    print("\n" + "-"*40)
    print("EXAMPLE 2: Morphological Similarity")
    print("-"*40)
    
    # Words with similar morphemes
    word_families = [
        ["run", "running", "runner", "runned"],  # Base + inflections
        ["happy", "happiness", "unhappy", "happily"],  # With affixes
        ["nation", "national", "international", "nationality"],  # Morphology
    ]
    
    for family in word_families:
        print(f"\nWord family: {family}")
        
        # Check which are in vocabulary
        for word in family:
            in_vocab = word in ft_lookup.model.wv if ft_lookup.model_type == 'fasttext_binary' else word in ft_lookup.model
            status = "vocab" if in_vocab else "OOV"
            print(f"  {word:20} ({status})", end="")
            
            # Get vector if possible
            vec = ft_lookup.get_vector(word, use_subwords=True)
            if vec is not None:
                print(f" - vector obtained")
            else:
                print(f" - no vector")
    
    # ============================================================================
    # Example 3: Misspelling robustness
    # ============================================================================
    print("\n" + "-"*40)
    print("EXAMPLE 3: Misspelling Handling")
    print("-"*40)
    
    correct_word = "restaurant"
    misspellings = ["restarant", "resturant", "restaraunt", "restront"]
    
    print(f"\nCorrect spelling: '{correct_word}'")
    
    # Get vector for correct word
    correct_vec = ft_lookup.get_vector(correct_word)
    
    if correct_vec is not None:
        for misspelling in misspellings:
            misspelling_vec = ft_lookup.get_vector(misspelling, use_subwords=True)
            
            if misspelling_vec is not None:
                # Compute cosine similarity
                similarity = np.dot(correct_vec, misspelling_vec) / (
                    np.linalg.norm(correct_vec) * np.linalg.norm(misspelling_vec)
                )
                print(f"  '{misspelling:15}' → similarity: {similarity:.3f}")
    
    # ============================================================================
    # Example 4: Subword analysis
    # ============================================================================
    if ft_lookup.model_type == 'fasttext_binary':
        print("\n" + "-"*40)
        print("EXAMPLE 4: Subword N-gram Analysis")
        print("-"*40)
        
        analysis_words = ["cat", "cats", "dog", "running"]
        
        for word in analysis_words:
            ngrams = ft_lookup.get_subword_ngrams(word)
            print(f"\n{word}:")
            print(f"  {len(ngrams)} n-grams: {', '.join(ngrams[:10])}", end="")
            if len(ngrams) > 10:
                print(f" ... and {len(ngrams)-10} more")
            else:
                print()
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*70)
    print("SUMMARY: FastText Advantages")
    print("="*70)
    print("1. OOV Word Handling: Can generate vectors for unseen words")
    print("2. Morphological Awareness: Captures word structure via n-grams")
    print("3. Robustness: Handles misspellings and rare words")
    print("4. Consistency: Similar subwords → similar vectors")
    print("\nLimitations:")
    print("1. Larger model size (stores n-gram vectors)")
    print("2. Slower inference (computes n-gram hashes)")
    print("3. .vec format lacks subword information")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    """
    Main execution block.
    
    When run directly, demonstrates FastText capabilities.
    In production, use the FastTextVectorLookup class directly.
    """
    
    print(__doc__)
    
    # Demonstrate capabilities (with error handling)
    try:
        demonstrate_fasttext_capabilities()
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("\nCommon issues:")
        print("1. Model file not found or incorrect path")
        print("2. Insufficient memory for large models")
        print("3. Gensim version incompatibility")
        print("\nExample usage in your code:")
        print("""
# Initialize FastText lookup
ft = FastTextVectorLookup('path/to/model.bin')

# Get vector for any word (including OOV)
vector = ft.get_vector('supercalifragilistic')

# Find similar words
similar = ft.get_similar_words('computational', top_n=5)

# Analyze a word
analysis = ft.analyze_word('university')
        """)

# ================================================================================
# ADDITIONAL RESOURCES
# ================================================================================
"""
FastText Resources:
1. Official FastText website: https://fasttext.cc/
2. Pre-trained models: https://fasttext.cc/docs/en/english-vectors.html
3. Research paper: "Enriching Word Vectors with Subword Information" (Bojanowski et al., 2017)

File Formats:
1. .bin: Binary format with full FastText model (subword support)
2. .vec: Text format with only word vectors (no subword support)

Common Models:
1. wiki-news-300d-1M-subword.vec: 1M words, 300D, with subwords
2. crawl-300d-2M-subword.bin: 2M words, 300D, with subwords
3. cc.en.300.bin: Common Crawl, 2M words, 300D

Use Cases:
1. Handling rare or domain-specific terminology
2. Morphologically rich languages
3. Social media text with misspellings/abbreviations
4. Biomedical/legal domain with compound words
"""