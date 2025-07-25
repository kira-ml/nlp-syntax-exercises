"""
02_remove_stopwords_dict.py

In this implementation, I demonstrate a foundational NLP preprocessing technique: stopword removal using a dictionary-based approach. This script illustrates how to efficiently filter out common, low-information words from text data, a crucial step in many text analysis pipelines.

The implementation uses Python's built-in set data structure for O(1) lookup performance and employs list comprehension for clean, idiomatic filtering. This approach balances code readability with computational efficiency, making it suitable for both educational purposes and production environments.

The code demonstrates several key NLP concepts:
- Stopword filtering using hash-based lookups
- Case-insensitive token matching
- Simple word boundary tokenization
- Modular design for pipeline integration
"""

import re

def get_stopwords_set() -> set:
    """
    Create a set of common English stopwords for text filtering.

    This function provides a basic stopword list focused on common English articles,
    prepositions, and conjunctions. In production systems, consider using more
    comprehensive lists from NLTK or spaCy, or domain-specific stopwords.

    Returns
    -------
    set
        A set of lowercase stopwords for efficient lookup operations.

    Notes
    -----
    - Using a set ensures O(1) lookup time complexity
    - The stopword list can be extended based on specific requirements
    """
    # Core set of common English stopwords
    # Set literal syntax provides better readability than set() constructor
    stopwords = {
        "a", "an", "and", "the", "is", "are", "in", "on", "of", 
        "to", "it", "this", "that"
    }
    return stopwords

def remove_stopwords(tokens: list, stopword_set: set) -> list:
    """
    Filter out stopwords from a list of tokens.

    This function demonstrates a Pythonic approach to token filtering using
    list comprehension and case-insensitive comparison.

    Parameters
    ----------
    tokens : list
        Input tokens to be filtered
    stopword_set : set
        Set of stopwords to remove

    Returns
    -------
    list
        Filtered list with stopwords removed
    """
    # Use list comprehension for efficient filtering
    # Convert tokens to lowercase during comparison for case-insensitivity
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_set]
    return filtered_tokens


def simple_tokenizer(text: str) -> list:
    """
    Tokenize text using word boundaries.
    
    This function implements basic word tokenization using regex word boundaries.
    While simple, this approach handles most common cases in English text.
    
    Parameters
    ----------
    text : str
        Input text to tokenize

    Returns
    -------
    list
        List of lowercase tokens
    
    Notes
    -----
    - Uses \b word boundaries for basic word splitting
    - Converts to lowercase for consistency
    - For more complex needs, consider NLTK's word_tokenize or spaCy
    """
    pattern = r'\b\w+\b'  # Match word characters between word boundaries
    tokens = re.findall(pattern, text.lower())
    return tokens


def preprocess_text(text: str, stopword_set: set) -> list:
    """
    Complete text preprocessing pipeline combining tokenization and stopword removal.
    
    This function demonstrates how to chain multiple preprocessing steps in a
    clear, maintainable way. It serves as an example of modular NLP pipeline design.
    
    Parameters
    ----------
    text : str
        Raw input text
    stopword_set : set
        Set of stopwords to filter out
    
    Returns
    -------
    list
        Processed list of tokens with stopwords removed
    """
    tokens = simple_tokenizer(text)
    filtered = remove_stopwords(tokens, stopword_set)
    return filtered


if __name__ == "__main__":
    # Demonstrate the pipeline with a sample text
    sample_tokens = ["This", "is", "a", "sample", "sentence", "in", "Python"]
    stops = get_stopwords_set()
    
    # Show stopword removal on pre-tokenized text
    filtered = remove_stopwords(sample_tokens, stops)
    
    # Demonstrate the complete pipeline on raw text
    text = "This is a sample sentence written in Python"
    tokens = simple_tokenizer(text)
    result = preprocess_text(text, stops)
    
    # Display results of each processing stage
    print("Filtered pre-tokenized text:", filtered)
    print("Raw tokenization:", tokens)
    print("Complete pipeline result:", result)



