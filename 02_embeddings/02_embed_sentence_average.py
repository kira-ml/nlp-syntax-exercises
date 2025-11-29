# 02_embed_sentence_average.py
"""
Compute average word embeddings for a sentence using pre-trained static embeddings, 
handling unknown words.

Concepts: Embedding aggregation, list filtering, numpy operations.
"""

import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from typing import List, Optional, Tuple
import re

class SentenceEmbedder:
    """
    A class to compute sentence embeddings by averaging word vectors.
    
    This demonstrates how to aggregate word-level embeddings into sentence-level
    representations while handling out-of-vocabulary words appropriately.
    """
    
    def __init__(self, word_vectors: KeyedVectors):
        """
        Initialize the SentenceEmbedder with pre-trained word vectors.
        
        Args:
            word_vectors (KeyedVectors): Pre-trained word embeddings from Gensim
        """
        self.word_vectors = word_vectors
        self.vector_size = word_vectors.vector_size
        print(f"Initialized SentenceEmbedder with vector size: {self.vector_size}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase and splitting into words.
        
        Args:
            text (str): Input sentence or text
            
        Returns:
            List[str]: List of preprocessed words
            
        Example:
            >>> preprocess_text("Hello World! This is a test.")
            ['hello', 'world', 'this', 'is', 'a', 'test']
        """
        # Convert to lowercase and split by non-alphanumeric characters
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get word vector if word exists in vocabulary, otherwise return None.
        
        Args:
            word (str): Word to get vector for
            
        Returns:
            Optional[np.ndarray]: Word vector or None if word not in vocabulary
            
        Example:
            >>> vector = embedder.get_word_vector("king")
            >>> vector.shape
            (300,)
        """
        try:
            return self.word_vectors[word]
        except KeyError:
            return None
    
    def average_embeddings(self, words: List[str], strategy: str = 'ignore') -> Tuple[Optional[np.ndarray], dict]:
        """
        Compute average embedding for a list of words.
        
        Args:
            words (List[str]): List of words to average
            strategy (str): How to handle unknown words. Options:
                          'ignore' - skip unknown words (default)
                          'zeros' - use zero vector for unknown words
                          'raise' - raise error for unknown words
                          
        Returns:
            Tuple[Optional[np.ndarray], dict]: 
                - Average vector or None if no valid words
                - Statistics about the embedding process
                
        Raises:
            ValueError: If strategy is 'raise' and unknown words are encountered
            
        Example:
            >>> avg_vector, stats = embedder.average_embeddings(["king", "queen", "xyz_unknown"])
            >>> print(f"Average vector shape: {avg_vector.shape}")
            >>> print(f"Words found: {stats['words_found']}")
        """
        valid_vectors = []
        unknown_words = []
        
        for word in words:
            vector = self.get_word_vector(word)
            if vector is not None:
                valid_vectors.append(vector)
            else:
                unknown_words.append(word)
        
        # Handle case where no valid words were found
        if not valid_vectors:
            if strategy == 'zeros':
                # Return zero vector
                avg_vector = np.zeros(self.vector_size)
            else:
                # Return None for no valid embeddings
                avg_vector = None
        else:
            # Convert list of vectors to 2D numpy array for efficient computation
            vectors_array = np.array(valid_vectors)
            
            # Compute average along axis 0 (average of each dimension across all words)
            avg_vector = np.mean(vectors_array, axis=0)
            
            # If using zeros strategy for unknown words, we need to adjust the average
            if strategy == 'zeros' and unknown_words:
                total_words = len(words)
                # Scale the average to account for zero-padded unknown words
                scaling_factor = len(valid_vectors) / total_words
                avg_vector = avg_vector * scaling_factor
        
        # Raise error if requested and unknown words found
        if strategy == 'raise' and unknown_words:
            raise ValueError(f"Unknown words encountered: {unknown_words}")
        
        # Compile statistics
        stats = {
            'total_words': len(words),
            'words_found': len(valid_vectors),
            'unknown_words': unknown_words,
            'coverage_rate': len(valid_vectors) / len(words) if words else 0.0
        }
        
        return avg_vector, stats
    
    def embed_sentence(self, sentence: str, strategy: str = 'ignore') -> Tuple[Optional[np.ndarray], dict]:
        """
        Compute average embedding for a complete sentence.
        
        Args:
            sentence (str): Input sentence to embed
            strategy (str): How to handle unknown words ('ignore', 'zeros', 'raise')
            
        Returns:
            Tuple[Optional[np.ndarray], dict]: 
                - Sentence embedding vector or None
                - Processing statistics
                
        Example:
            >>> embedding, stats = embedder.embed_sentence("The quick brown fox jumps")
            >>> print(f"Embedding shape: {embedding.shape}")
            >>> print(f"Coverage: {stats['coverage_rate']:.2%}")
        """
        # Preprocess the sentence into individual words
        words = self.preprocess_text(sentence)
        
        if not words:
            # Return zero vector for empty sentences
            zero_vector = np.zeros(self.vector_size)
            stats = {
                'total_words': 0,
                'words_found': 0,
                'unknown_words': [],
                'coverage_rate': 0.0
            }
            return zero_vector, stats
        
        # Compute average embeddings
        return self.average_embeddings(words, strategy)


def demonstrate_embedding_operations():
    """
    Demonstrate various embedding operations and numpy concepts.
    """
    print("=" * 60)
    print("DEMONSTRATING EMBEDDING OPERATIONS")
    print("=" * 60)
    
    # Create some dummy vectors to demonstrate numpy operations
    print("\n1. NUMPY OPERATIONS DEMONSTRATION")
    print("-" * 40)
    
    # Simulate word vectors (3 words, 4 dimensions each)
    vectors = np.array([
        [1.0, 2.0, 3.0, 4.0],  # Word 1
        [2.0, 3.0, 4.0, 5.0],  # Word 2  
        [3.0, 4.0, 5.0, 6.0]   # Word 3
    ])
    
    print(f"Word vectors shape: {vectors.shape}")  # (3, 4)
    print(f"Individual vector shape: {vectors[0].shape}")  # (4,)
    
    # Average along axis 0 - average of each dimension across all words
    average_vector = np.mean(vectors, axis=0)
    print(f"Average vector: {average_vector}")
    print(f"Average vector shape: {average_vector.shape}")  # (4,)


def main():
    """
    Main function to demonstrate sentence embedding capabilities.
    """
    # Load pre-trained word vectors
    print("Loading pre-trained word vectors...")
    try:
        # Using a smaller model for demonstration
        word_vectors = api.load('glove-wiki-gigaword-100')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using a mock implementation for demonstration...")
        return
    
    # Initialize sentence embedder
    embedder = SentenceEmbedder(word_vectors)
    
    # Demonstrate numpy operations
    demonstrate_embedding_operations()
    
    # Test sentences with different characteristics
    test_sentences = [
        "The king and queen ruled the kingdom",  # All words should be in vocabulary
        "Python programming is amazing",         # Mixed case, common words
        "xyz unknownword notinvocab test",       # Contains unknown words  
        "The quick brown fox jumps over the lazy dog",  # Classic example
        "",                                      # Empty sentence
        "!@#$%^&*()",                           # Only special characters
    ]
    
    print("\n\n2. SENTENCE EMBEDDING EXAMPLES")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Sentence: '{sentence}'")
        print("-" * 40)
        
        # Try different strategies
        for strategy in ['ignore', 'zeros']:
            try:
                embedding, stats = embedder.embed_sentence(sentence, strategy=strategy)
                
                print(f"\n  Strategy: {strategy}")
                print(f"  Total words: {stats['total_words']}")
                print(f"  Words found: {stats['words_found']}")
                print(f"  Unknown words: {stats['unknown_words']}")
                print(f"  Coverage rate: {stats['coverage_rate']:.2%}")
                
                if embedding is not None:
                    print(f"  Embedding shape: {embedding.shape}")
                    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
                    print(f"  First 5 dimensions: {embedding[:5]}")
                else:
                    print("  No valid embedding computed")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n\n3. COMPARING SENTENCE SIMILARITIES")
    print("=" * 60)
    
    # Compare similar and dissimilar sentences
    sentence_pairs = [
        ("I love programming", "I enjoy coding"),
        ("The weather is nice", "Programming is fun"),
        ("cat dog animal", "kitty puppy pet"),
    ]
    
    for sent1, sent2 in sentence_pairs:
        emb1, stats1 = embedder.embed_sentence(sent1)
        emb2, stats2 = embedder.embed_sentence(sent2)
        
        if emb1 is not None and emb2 is not None:
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"\n'{sent1}' vs '{sent2}'")
            print(f"Similarity: {similarity:.4f}")
            print(f"Coverage 1: {stats1['coverage_rate']:.2%}, Coverage 2: {stats2['coverage_rate']:.2%}")


if __name__ == "__main__":
    main()