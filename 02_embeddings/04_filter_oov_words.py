# 04_filter_oov_words.py
"""
Create a utility to filter out-of-vocabulary (OOV) words from tokenized input, 
a common preprocessing step in embedding-based models.

Concepts: Set operations, membership testing, list comprehensions.
"""

from typing import List, Tuple, Dict, Set, Union
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import time


class OOVFilter:
    """
    A utility class for filtering Out-Of-Vocabulary (OOV) words from text data.
    
    OOV words are tokens that don't exist in a pre-trained model's vocabulary.
    Filtering them is crucial because:
    1. They cannot be represented by the model's embeddings
    2. They can cause errors during model inference
    3. They add noise to downstream tasks
    
    This class demonstrates efficient OOV handling using set operations and list comprehensions.
    """
    
    def __init__(self, word_vectors: KeyedVectors):
        """
        Initialize the OOV filter with a pre-trained word embedding model.
        
        Args:
            word_vectors (KeyedVectors): Pre-trained word vectors that define our vocabulary
            
        Example:
            >>> word_vectors = api.load('glove-wiki-gigaword-100')
            >>> oov_filter = OOVFilter(word_vectors)
        """
        self.word_vectors = word_vectors
        
        # CONCEPT: Set Operations - Creating a set for O(1) membership testing
        # Sets provide constant-time lookups vs O(n) for lists
        self.vocabulary_set: Set[str] = set(word_vectors.key_to_index.keys())
        
        # Store vocabulary size for reporting
        self.vocab_size = len(self.vocabulary_set)
        self.vector_size = word_vectors.vector_size
        
        print(f"✅ OOV Filter initialized with {self.vocab_size:,} words")
        print(f"📊 Vocabulary coverage statistics ready")
    
    def demonstrate_set_operations(self):
        """
        Demonstrate the power of set operations for vocabulary management.
        
        Sets provide O(1) average case complexity for membership testing,
        making them ideal for OOV filtering in large vocabularies.
        """
        print(f"\n{'='*60}")
        print("SET OPERATIONS DEMONSTRATION")
        print(f"{'='*60}")
        
        # Create sample sets for demonstration
        vocabulary = {'apple', 'banana', 'cherry', 'date', 'elderberry'}
        input_tokens = ['apple', 'banana', 'unknown1', 'cherry', 'unknown2']
        
        print(f"Vocabulary set: {vocabulary}")
        print(f"Input tokens: {input_tokens}")
        
        # CONCEPT: Set Membership Testing
        print(f"\n1. MEMBERSHIP TESTING (O(1) average case):")
        for token in input_tokens:
            is_in_vocab = token in vocabulary  # This is O(1) for sets!
            print(f"   '{token}' in vocabulary: {is_in_vocab}")
        
        # CONCEPT: Set Intersection - Find common elements
        common_tokens = vocabulary.intersection(input_tokens)
        print(f"\n2. SET INTERSECTION (common tokens): {common_tokens}")
        
        # CONCEPT: Set Difference - Find OOV words
        oov_tokens = set(input_tokens) - vocabulary
        print(f"3. SET DIFFERENCE (OOV tokens): {oov_tokens}")
        
        # CONCEPT: Set Union - Combine vocabularies
        extended_vocab = vocabulary.union(['fig', 'grape'])
        print(f"4. SET UNION (extended vocabulary): {extended_vocab}")
    
    def filter_oov_naive(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter OOV words using naive list iteration.
        
        This method demonstrates the basic concept but is less efficient
        for large token lists due to O(n) membership testing in lists.
        
        Args:
            tokens (List[str]): List of input tokens to filter
            
        Returns:
            Tuple[List[str], List[str]]: 
                - Filtered tokens (in-vocabulary words)
                - OOV tokens that were filtered out
                
        Time Complexity: O(n * m) where n = tokens, m = vocabulary (inefficient!)
        """
        print(f"\n🔍 Naive OOV Filtering (Educational Demo)")
        print(f"   Input: {len(tokens)} tokens")
        
        start_time = time.time()
        
        filtered_tokens = []
        oov_tokens = []
        
        # CONCEPT: Naive Iteration - Demonstrates why sets are better
        for token in tokens:
            # INEFFICIENT: This checks every token against the entire vocabulary list
            if token in list(self.vocabulary_set)[:1000]:  # Limiting for demo
                filtered_tokens.append(token)
            else:
                oov_tokens.append(token)
        
        end_time = time.time()
        
        print(f"   ✅ Filtered: {len(filtered_tokens)} in-vocabulary tokens")
        print(f"   ❌ Removed: {len(oov_tokens)} OOV tokens")
        print(f"   ⏱️  Time: {(end_time - start_time)*1000:.2f} ms")
        
        return filtered_tokens, oov_tokens
    
    def filter_oov_efficient(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter OOV words using efficient set operations and list comprehensions.
        
        This method uses Python's set data structure for O(1) membership testing,
        making it much faster for large token lists.
        
        Args:
            tokens (List[str]): List of input tokens to filter
            
        Returns:
            Tuple[List[str], List[str]]: 
                - Filtered tokens (in-vocabulary words)
                - OOV tokens that were filtered out
                
        Time Complexity: O(n) where n = number of tokens (efficient!)
        """
        print(f"\n🚀 Efficient OOV Filtering (Production Ready)")
        print(f"   Input: {len(tokens)} tokens")
        
        start_time = time.time()
        
        # CONCEPT: List Comprehensions with Conditional Filtering
        # This is both readable and efficient
        filtered_tokens = [token for token in tokens if token in self.vocabulary_set]
        
        # Alternative approach using set operations for OOV detection
        oov_tokens = [token for token in tokens if token not in self.vocabulary_set]
        
        end_time = time.time()
        
        print(f"   ✅ Filtered: {len(filtered_tokens)} in-vocabulary tokens")
        print(f"   ❌ Removed: {len(oov_tokens)} OOV tokens") 
        print(f"   ⏱️  Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"   📈 Coverage: {len(filtered_tokens)/len(tokens)*100:.1f}%")
        
        return filtered_tokens, oov_tokens
    
    def filter_oov_batch(self, token_lists: List[List[str]]) -> List[Tuple[List[str], List[str]]]:
        """
        Filter OOV words from multiple token lists (batch processing).
        
        Demonstrates how to efficiently process multiple documents at once,
        a common requirement in NLP pipelines.
        
        Args:
            token_lists (List[List[str]]): List of tokenized documents
            
        Returns:
            List[Tuple[List[str], List[str]]]: 
                - For each document: (filtered_tokens, oov_tokens)
        """
        print(f"\n📚 Batch Processing {len(token_lists)} documents")
        
        # CONCEPT: List Comprehension for Batch Operations
        # Process each document using our efficient method
        results = [self.filter_oov_efficient(tokens) for tokens in token_lists]
        
        # Calculate batch statistics
        total_tokens = sum(len(tokens) for tokens in token_lists)
        total_filtered = sum(len(filtered) for filtered, _ in results)
        total_oov = sum(len(oov) for _, oov in results)
        
        print(f"   📊 Batch Statistics:")
        print(f"   Total tokens processed: {total_tokens:,}")
        print(f"   Total in-vocabulary: {total_filtered:,} ({total_filtered/total_tokens*100:.1f}%)")
        print(f"   Total OOV: {total_oov:,} ({total_oov/total_tokens*100:.1f}%)")
        
        return results
    
    def get_vocabulary_coverage(self, corpus_tokens: List[str]) -> Dict[str, float]:
        """
        Analyze vocabulary coverage statistics for a given corpus.
        
        Args:
            corpus_tokens (List[str]): All tokens from a corpus to analyze
            
        Returns:
            Dict[str, float]: Coverage statistics
        """
        unique_tokens = set(corpus_tokens)
        
        # CONCEPT: Set Operations for Coverage Analysis
        covered_tokens = unique_tokens.intersection(self.vocabulary_set)
        uncovered_tokens = unique_tokens - self.vocabulary_set
        
        stats = {
            'total_unique_tokens': len(unique_tokens),
            'covered_unique_tokens': len(covered_tokens),
            'coverage_rate': len(covered_tokens) / len(unique_tokens),
            'vocabulary_utilization': len(covered_tokens) / self.vocab_size
        }
        
        return stats
    
    def demonstrate_list_comprehensions(self):
        """
        Demonstrate the power and efficiency of list comprehensions.
        
        List comprehensions are more readable and often faster than
        traditional for-loops for simple transformations and filtering.
        """
        print(f"\n{'='*60}")
        print("LIST COMPREHENSIONS DEMONSTRATION")
        print(f"{'='*60}")
        
        sample_tokens = ['the', 'quick', 'brown', 'unknown_word', 'fox', 'another_unknown']
        
        print(f"Sample tokens: {sample_tokens}")
        
        # Traditional for-loop approach
        print(f"\n1. TRADITIONAL FOR-LOOP:")
        filtered_traditional = []
        for token in sample_tokens:
            if token in self.vocabulary_set:
                filtered_traditional.append(token)
        print(f"   Result: {filtered_traditional}")
        
        # List comprehension approach (more Pythonic)
        print(f"\n2. LIST COMPREHENSION:")
        filtered_comprehension = [token for token in sample_tokens 
                                 if token in self.vocabulary_set]
        print(f"   Result: {filtered_comprehension}")
        
        # Both methods produce the same result
        assert filtered_traditional == filtered_comprehension
        print(f"\n   ✅ Both methods produce identical results!")
        
        # Additional list comprehension examples
        print(f"\n3. ADVANCED LIST COMPREHENSIONS:")
        
        # With transformation
        lengths = [len(token) for token in sample_tokens]
        print(f"   Token lengths: {lengths}")
        
        # With conditional transformation
        filtered_lengths = [len(token) for token in sample_tokens 
                           if token in self.vocabulary_set]
        print(f"   Filtered token lengths: {filtered_lengths}")


def create_sample_corpus() -> List[List[str]]:
    """
    Create a sample corpus with known OOV words for demonstration.
    
    Returns:
        List[List[str]]: Sample tokenized documents
    """
    documents = [
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["python", "programming", "is", "fun", "and", "rewarding"],
        ["this", "contains", "unknown_word", "and", "another_unknown", "token"],
        ["artificial", "intelligence", "machine", "learning", "deep", "neural", "networks"],
        ["hello", "world", "test", "example", "demonstration"]
    ]
    
    # Add some intentional OOV words
    documents[0].append("xyz_unknown_animal")
    documents[2].extend(["not_in_vocab", "missing_token"])
    
    return documents


def main():
    """
    Main function to demonstrate OOV filtering capabilities.
    """
    # Load pre-trained word vectors
    print("Loading pre-trained word vectors...")
    try:
        # Using a smaller model for demonstration
        word_vectors = api.load('glove-wiki-gigaword-100')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize OOV filter
    oov_filter = OOVFilter(word_vectors)
    
    # Demonstrate core concepts
    oov_filter.demonstrate_set_operations()
    oov_filter.demonstrate_list_comprehensions()
    
    # Create test data
    sample_corpus = create_sample_corpus()
    
    print(f"\n{'='*60}")
    print("OOV FILTERING DEMONSTRATION")
    print(f"{'='*60}")
    
    # Test individual document filtering
    test_document = sample_corpus[2]  # Contains known OOV words
    print(f"\n📄 Testing single document:")
    print(f"   Original: {test_document}")
    
    # Compare both methods
    filtered_naive, oov_naive = oov_filter.filter_oov_naive(test_document)
    filtered_efficient, oov_efficient = oov_filter.filter_oov_efficient(test_document)
    
    print(f"\n   Naive method - Filtered: {filtered_naive}")
    print(f"   Naive method - OOV: {oov_naive}")
    print(f"   Efficient method - Filtered: {filtered_efficient}")
    print(f"   Efficient method - OOV: {oov_efficient}")
    
    # Test batch processing
    print(f"\n{'='*60}")
    print("BATCH PROCESSING DEMONSTRATION")
    print(f"{'='*60}")
    
    batch_results = oov_filter.filter_oov_batch(sample_corpus)
    
    # Show results for each document
    for i, (filtered_tokens, oov_tokens) in enumerate(batch_results):
        print(f"\n   Document {i+1}:")
        print(f"     ✅ In vocabulary ({len(filtered_tokens)}): {filtered_tokens}")
        print(f"     ❌ OOV ({len(oov_tokens)}): {oov_tokens}")
    
    # Vocabulary coverage analysis
    print(f"\n{'='*60}")
    print("VOCABULARY COVERAGE ANALYSIS")
    print(f"{'='*60}")
    
    # Flatten all tokens for coverage analysis
    all_tokens = [token for doc in sample_corpus for token in doc]
    coverage_stats = oov_filter.get_vocabulary_coverage(all_tokens)
    
    print(f"\n📊 Coverage Statistics:")
    for stat_name, value in coverage_stats.items():
        if 'rate' in stat_name or 'utilization' in stat_name:
            print(f"   {stat_name.replace('_', ' ').title()}: {value:.1%}")
        else:
            print(f"   {stat_name.replace('_', ' ').title()}: {value:,}")


if __name__ == "__main__":
    main()