# 03_find_most_similar_words.py
"""
Use cosine similarity to find top-N most similar words to a given query 
in a static embedding space.

Concepts: numpy, vector norms, dictionary iteration.
"""

import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from typing import List, Tuple, Optional
import time


class WordSimilarityFinder:
    """
    A class to find the most similar words using cosine similarity in embedding space.
    
    Cosine similarity measures the cosine of the angle between two vectors, 
    providing a value between -1 (completely dissimilar) and 1 (identical direction).
    This is commonly used in NLP to find semantically similar words.
    """
    
    def __init__(self, word_vectors: KeyedVectors):
        """
        Initialize the similarity finder with pre-trained word vectors.
        
        Args:
            word_vectors (KeyedVectors): Pre-trained word embeddings containing
                                       the vocabulary and vector representations
        """
        self.word_vectors = word_vectors
        self.vocab = word_vectors.key_to_index  # Dictionary mapping words to indices
        print(f"Initialized WordSimilarityFinder with {len(self.vocab):,} words")
        print(f"Vector dimensions: {word_vectors.vector_size}")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Formula: cos(θ) = (A·B) / (||A|| * ||B||)
        Where A·B is the dot product, and ||A|| is the Euclidean norm (magnitude).
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity between -1 and 1
            
        Example:
            >>> vec1 = np.array([1, 0])
            >>> vec2 = np.array([1, 1])
            >>> cosine_similarity(vec1, vec2)
            0.7071  # 45-degree angle
        """
        # Dot product: sum of element-wise multiplication
        dot_product = np.dot(vec1, vec2)
        
        # Euclidean norm (L2 norm): sqrt(sum of squares)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        # Cosine similarity calculation
        similarity = dot_product / (norm_vec1 * norm_vec2)
        
        # Due to floating point precision, values might slightly exceed [-1, 1]
        return np.clip(similarity, -1.0, 1.0)
    
    def get_query_vector(self, query: str) -> Optional[np.ndarray]:
        """
        Retrieve the vector for a query word if it exists in vocabulary.
        
        Args:
            query (str): Word to search for
            
        Returns:
            Optional[np.ndarray]: Word vector or None if not found
        """
        try:
            return self.word_vectors[query]
        except KeyError:
            return None
    
    def find_similar_words_naive(self, query_vector: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words using naive iteration through all vocabulary.
        
        This method demonstrates the basic concept but is computationally expensive
        for large vocabularies (O(V) where V is vocabulary size).
        
        Args:
            query_vector (np.ndarray): Vector of the query word
            top_n (int): Number of most similar words to return
            
        Returns:
            List[Tuple[str, float]]: List of (word, similarity) tuples, sorted by similarity
            
        Example:
            >>> similar_words = finder.find_similar_words_naive(query_vec, top_n=5)
            >>> for word, sim in similar_words:
            ...     print(f"{word}: {sim:.4f}")
        """
        similarities = []
        
        print(f"\nComputing similarities across {len(self.vocab):,} words...")
        start_time = time.time()
        
        # Iterate through all words in vocabulary - DEMONSTRATES DICTIONARY ITERATION
        for word, index in self.vocab.items():
            # Get vector for current word
            word_vector = self.word_vectors[word]
            
            # Compute cosine similarity
            similarity = self.cosine_similarity(query_vector, word_vector)
            
            # Store result
            similarities.append((word, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        print(f"Naive search completed in {end_time - start_time:.4f} seconds")
        
        # Return top N results (excluding the query word itself if present)
        results = []
        count = 0
        for word, similarity in similarities:
            # Skip if we've collected enough results
            if count >= top_n:
                break
            results.append((word, similarity))
            count += 1
            
        return results
    
    def find_similar_words_optimized(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words using Gensim's optimized method.
        
        This uses optimized linear algebra operations and is much faster than
        naive iteration. Demonstrates the practical approach for production use.
        
        Args:
            query (str): Query word
            top_n (int): Number of most similar words to return
            
        Returns:
            List[Tuple[str, float]]: List of (word, similarity) tuples
        """
        if query not in self.word_vectors:
            print(f"Word '{query}' not in vocabulary")
            return []
        
        print(f"\nUsing optimized similarity search...")
        start_time = time.time()
        
        # Gensim's built-in method for finding most similar words
        similar_words = self.word_vectors.most_similar(query, topn=top_n)
        
        end_time = time.time()
        print(f"Optimized search completed in {end_time - start_time:.4f} seconds")
        
        return similar_words
    
    def demonstrate_vector_operations(self, query: str):
        """
        Demonstrate the vector operations involved in similarity calculation.
        
        Args:
            query (str): Word to use for demonstration
        """
        if query not in self.word_vectors:
            print(f"Word '{query}' not in vocabulary for demonstration")
            return
        
        query_vector = self.word_vectors[query]
        
        print(f"\n{'='*60}")
        print(f"VECTOR OPERATIONS DEMONSTRATION FOR: '{query}'")
        print(f"{'='*60}")
        
        # Show query vector properties
        print(f"\n1. QUERY VECTOR PROPERTIES:")
        print(f"   Shape: {query_vector.shape}")
        print(f"   Magnitude (L2 norm): {np.linalg.norm(query_vector):.4f}")
        print(f"   First 5 dimensions: {query_vector[:5]}")
        
        # Demonstrate with a few example words
        demo_words = ['king', 'queen', 'computer', 'unrelated']
        
        print(f"\n2. SIMILARITY CALCULATIONS:")
        for word in demo_words:
            if word in self.word_vectors and word != query:
                word_vector = self.word_vectors[word]
                
                # Manual calculation steps
                dot_product = np.dot(query_vector, word_vector)
                norm_query = np.linalg.norm(query_vector)
                norm_word = np.linalg.norm(word_vector)
                manual_similarity = dot_product / (norm_query * norm_word)
                
                # Using our method
                our_similarity = self.cosine_similarity(query_vector, word_vector)
                
                print(f"\n   '{query}' vs '{word}':")
                print(f"   Dot product: {dot_product:.4f}")
                print(f"   Norm '{query}': {norm_query:.4f}")
                print(f"   Norm '{word}': {norm_word:.4f}")
                print(f"   Manual similarity: {manual_similarity:.4f}")
                print(f"   Our method similarity: {our_similarity:.4f}")


def demonstrate_numpy_concepts():
    """
    Demonstrate key numpy concepts used in similarity calculations.
    """
    print(f"\n{'='*60}")
    print("NUMPY CONCEPTS DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create sample vectors for demonstration
    vector_a = np.array([3, 4])    # 2D vector
    vector_b = np.array([4, 3])    # Same magnitude, different direction
    vector_c = np.array([6, 8])    # Same direction as A, twice magnitude
    vector_d = np.array([-3, -4])  # Opposite direction to A
    
    vectors = {
        'A': vector_a,
        'B': vector_b, 
        'C': vector_c,
        'D': vector_d
    }
    
    print("\n1. VECTOR NORMS (MAGNITUDES):")
    for name, vector in vectors.items():
        norm = np.linalg.norm(vector)
        print(f"   Vector {name}: {vector}, Norm: {norm:.2f}")
    
    print("\n2. DOT PRODUCTS:")
    for name1, vec1 in vectors.items():
        for name2, vec2 in vectors.items():
            if name1 < name2:  # Avoid duplicate pairs
                dot_product = np.dot(vec1, vec2)
                print(f"   {name1}·{name2} = {dot_product}")
    
    print("\n3. COSINE SIMILARITIES:")
    finder = WordSimilarityFinder.__new__(WordSimilarityFinder)  # Create instance without init
    
    for name1, vec1 in vectors.items():
        for name2, vec2 in vectors.items():
            if name1 < name2:
                similarity = finder.cosine_similarity(vec1, vec2)
                print(f"   cos({name1}, {name2}) = {similarity:.4f}")
    
    print("\n4. KEY OBSERVATIONS:")
    print("   - Same direction (A,C): similarity = 1.0")
    print("   - Orthogonal (perpendicular): similarity = 0.0")  
    print("   - Opposite direction (A,D): similarity = -1.0")
    print("   - Same magnitude, different angle (A,B): similarity = 0.96")


def main():
    """
    Main function to demonstrate word similarity finding capabilities.
    """
    # Load pre-trained word vectors
    print("Loading pre-trained word vectors...")
    try:
        # Using a medium-sized model for good performance and variety
        word_vectors = api.load('glove-wiki-gigaword-300')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize similarity finder
    finder = WordSimilarityFinder(word_vectors)
    
    # Demonstrate numpy concepts
    demonstrate_numpy_concepts()
    
    # Test queries
    test_queries = [
        "king",
        "computer", 
        "beautiful",
        "paris",
        "programming",
        "xyz_unknown_word"  # This should not be in vocabulary
    ]
    
    print(f"\n{'='*60}")
    print("WORD SIMILARITY SEARCH RESULTS")
    print(f"{'='*60}")
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"QUERY: '{query}'")
        print(f"{'='*50}")
        
        # Check if query exists
        if query not in finder.word_vectors:
            print(f"✗ Word '{query}' not in vocabulary")
            continue
        
        # Demonstrate vector operations for first query
        if query == test_queries[0]:
            finder.demonstrate_vector_operations(query)
        
        # Method 1: Naive implementation (educational)
        print(f"\n1. NAIVE IMPLEMENTATION (Educational):")
        query_vector = finder.get_query_vector(query)
        naive_results = finder.find_similar_words_naive(query_vector, top_n=5)
        
        for i, (word, similarity) in enumerate(naive_results, 1):
            print(f"   {i:2d}. {word:15s} {similarity:.4f}")
        
        # Method 2: Optimized implementation (practical)
        print(f"\n2. OPTIMIZED IMPLEMENTATION (Practical):")
        optimized_results = finder.find_similar_words_optimized(query, top_n=5)
        
        for i, (word, similarity) in enumerate(optimized_results, 1):
            print(f"   {i:2d}. {word:15s} {similarity:.4f}")
        
        # Verify both methods give same results
        if len(naive_results) == len(optimized_results):
            matches = sum(1 for (w1, s1), (w2, s2) in zip(naive_results, optimized_results) 
                         if w1 == w2 and abs(s1 - s2) < 0.001)
            print(f"\n   ✓ Results match: {matches}/{len(naive_results)} words identical")
        else:
            print(f"\n   ⚠ Result lists have different lengths")


if __name__ == "__main__":
    main()