"""
17_build_vocab_from_tokens.py

Build a vocabulary from a list of tokenized sentences, including frequency counting 
and optional minimum frequency thresholding.

Concepts: collections.Counter, vocabulary trimming.
"""

import collections
from typing import List, Dict, Set, Optional

def build_vocabulary_from_tokens(
    tokenized_sentences: List[List[str]], 
    min_frequency: Optional[int] = None,
    special_tokens: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Build a vocabulary dictionary from tokenized sentences.
    
    This function takes a list of tokenized sentences and creates a vocabulary mapping
    each unique token to a unique integer index. It can filter out rare tokens based
    on a minimum frequency threshold and include special tokens.
    
    Args:
        tokenized_sentences: A list of sentences, where each sentence is a list of string tokens.
                            Example: [["hello", "world"], ["good", "morning"]]
        min_frequency: Optional minimum frequency threshold. Tokens that appear fewer times
                      than this value will be excluded from the vocabulary. If None, all
                      tokens are included.
        special_tokens: Optional list of special tokens to include in the vocabulary
                       (e.g., ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]). These tokens will
                       be added to the beginning of the vocabulary.
    
    Returns:
        A dictionary mapping tokens to integer indices. Special tokens (if provided) 
        get the lowest indices, followed by the most frequent tokens.
    
    Example:
        >>> sentences = [["hello", "world"], ["hello", "python"], ["world", "!"]]
        >>> vocab = build_vocabulary_from_tokens(sentences, min_frequency=1)
        >>> print(vocab)
        {'<UNK>': 0, 'hello': 1, 'world': 2, 'python': 3, '!': 4}
    """
    
    # Step 1: Initialize special tokens if provided
    # Special tokens are typically used for padding, unknown words, sentence boundaries, etc.
    # They get the lowest indices (0, 1, 2, ...) in the vocabulary
    if special_tokens is None:
        special_tokens = ["<UNK>"]  # Default: only include unknown token
    
    # Step 2: Flatten the list of tokenized sentences and count token frequencies
    # We use collections.Counter which is optimized for counting hashable objects
    # The list comprehension flattens the 2D list of sentences into a 1D list of tokens
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    token_counter = collections.Counter(all_tokens)
    
    print(f"Original vocabulary size: {len(token_counter)}")
    print(f"Total tokens processed: {len(all_tokens)}")
    
    # Step 3: Apply minimum frequency filtering if specified
    # This helps reduce vocabulary size by removing rare tokens that might be noise
    if min_frequency is not None:
        # Dictionary comprehension that keeps only tokens meeting the frequency threshold
        filtered_tokens = {
            token: count for token, count in token_counter.items() 
            if count >= min_frequency
        }
        token_counter = filtered_tokens
        print(f"Vocabulary size after min_frequency={min_frequency} filtering: {len(token_counter)}")
        print(f"Tokens removed: {len(all_tokens) - len(token_counter)}")
    
    # Step 4: Sort tokens by frequency (descending) for consistent ordering
    # Most frequent tokens get lower indices, which can help with some model optimizations
    # sorted() returns a list of (token, count) tuples, sorted by count in descending order
    sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Step 5: Build the vocabulary dictionary
    # We start with special tokens, then add the sorted frequent tokens
    vocabulary = {}
    current_index = 0
    
    # Add special tokens first (they get indices 0, 1, 2, ...)
    for token in special_tokens:
        vocabulary[token] = current_index
        current_index += 1
    
    # Add the frequent tokens from our sorted list
    # We only need the token string, not the count, for the vocabulary mapping
    for token, count in sorted_tokens:
        vocabulary[token] = current_index
        current_index += 1
    
    return vocabulary

def get_token_frequencies(tokenized_sentences: List[List[str]]) -> Dict[str, int]:
    """
    Get raw frequency counts for all tokens in the tokenized sentences.
    
    This is a helper function that shows the frequency distribution before
    building the vocabulary.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        
    Returns:
        Dictionary mapping each token to its frequency count
    """
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    return dict(collections.Counter(all_tokens))

def analyze_vocabulary_coverage(
    vocabulary: Dict[str, int], 
    tokenized_sentences: List[List[str]]
) -> Dict[str, float]:
    """
    Analyze what percentage of tokens in the corpus are covered by the vocabulary.
    
    This is useful for understanding how much data might be lost to unknown tokens
    when using the vocabulary.
    
    Args:
        vocabulary: The vocabulary dictionary to test
        tokenized_sentences: The original tokenized sentences
        
    Returns:
        Dictionary with coverage statistics
    """
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    total_tokens = len(all_tokens)
    
    # Count how many tokens are in vocabulary (excluding special tokens from count)
    # We assume special tokens are those that weren't in the original corpus
    covered_tokens = [token for token in all_tokens if token in vocabulary]
    covered_count = len(covered_tokens)
    
    coverage_percentage = (covered_count / total_tokens) * 100
    
    return {
        "total_tokens": total_tokens,
        "covered_tokens": covered_count,
        "coverage_percentage": coverage_percentage,
        "oov_percentage": 100 - coverage_percentage  # Out Of Vocabulary percentage
    }

def main():
    """
    Demonstrate the vocabulary building process with example data.
    """
    # Example tokenized sentences - in practice, this would come from your data
    example_sentences = [
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["hello", "world", "this", "is", "a", "test"],
        ["the", "cat", "in", "the", "hat"],
        ["python", "programming", "is", "fun"],
        ["the", "the", "the", "test", "test"]  # Intentional repetition for frequency demo
    ]
    
    print("=" * 60)
    print("VOCABULARY BUILDER DEMONSTRATION")
    print("=" * 60)
    
    # Show raw frequency counts
    print("\n1. RAW FREQUENCY COUNTS:")
    print("-" * 30)
    frequencies = get_token_frequencies(example_sentences)
    for token, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {count}")
    
    # Build vocabulary without frequency thresholding
    print("\n2. VOCABULARY WITHOUT FREQUENCY THRESHOLD:")
    print("-" * 40)
    vocab_all = build_vocabulary_from_tokens(
        example_sentences, 
        min_frequency=None,
        special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    )
    
    # Display the vocabulary
    for token, idx in vocab_all.items():
        print(f"  {idx:2d}: {token}")
    
    # Analyze coverage
    coverage = analyze_vocabulary_coverage(vocab_all, example_sentences)
    print(f"\n  Coverage: {coverage['coverage_percentage']:.1f}% "
          f"({coverage['covered_tokens']}/{coverage['total_tokens']} tokens)")
    
    # Build vocabulary with frequency thresholding
    print("\n3. VOCABULARY WITH min_frequency=2:")
    print("-" * 35)
    vocab_filtered = build_vocabulary_from_tokens(
        example_sentences, 
        min_frequency=2,
        special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    )
    
    # Display the filtered vocabulary
    for token, idx in vocab_filtered.items():
        print(f"  {idx:2d}: {token}")
    
    # Analyze coverage of filtered vocabulary
    coverage_filtered = analyze_vocabulary_coverage(vocab_filtered, example_sentences)
    print(f"\n  Coverage: {coverage_filtered['coverage_percentage']:.1f}% "
          f"({coverage_filtered['covered_tokens']}/{coverage_filtered['total_tokens']} tokens)")
    print(f"  OOV (Out Of Vocabulary) rate: {coverage_filtered['oov_percentage']:.1f}%")
    
    print("\n4. KEY CONCEPTS DEMONSTRATED:")
    print("-" * 35)
    print("  • collections.Counter: Efficient frequency counting")
    print("  • List flattening: Converting 2D sentences to 1D tokens")
    print("  • Dictionary comprehensions: Filtering tokens by frequency")
    print("  • Vocabulary trimming: Removing rare tokens to reduce noise")
    print("  • Special tokens: Handling reserved tokens for ML models")
    print("  • Coverage analysis: Understanding vocabulary effectiveness")

if __name__ == "__main__":
    main()