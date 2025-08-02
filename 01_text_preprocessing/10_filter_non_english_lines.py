"""Language Detection and Text Filtering for English Content

This module demonstrates robust language detection techniques for filtering English text
from multilingual datasets - a critical preprocessing step in many NLP pipelines.
The implementation showcases multiple complementary approaches:

1. Unicode Analysis:
   - Character script detection using Unicode character names
   - ASCII ratio analysis for Western text identification
   - Handles edge cases like mixed-script content

2. Linguistic Heuristics:
   - English letter frequency patterns
   - Common foreign word detection
   - Statistical threshold-based classification

Key Applications:
   - Preprocessing multilingual web-scraped data
   - Cleaning social media datasets
   - Preparing monolingual training corpora
   - Content moderation pipelines

This approach is particularly valuable when working with noisy, real-world text data
where simple language detection libraries may struggle with short texts, code-switching,
or transliterated content.

Author: kira-ml
Date: August 3, 2025
"""

import unicodedata
from typing import List

def filter_english_lines(lines: List[str]) -> List[str]:
    """Filter a list of text lines to retain only those likely to be English.
    
    This function serves as the main entry point for batch text filtering,
    applying sophisticated language detection to each line independently.
    In production pipelines, this is typically used for:
    - Dataset cleaning and preparation
    - Multilingual content separation
    - Quality control in text processing workflows
    
    Args:
        lines (List[str]): List of text lines to filter.
    
    Returns:
        List[str]: Filtered list containing only lines classified as English.
    
    Example:
        >>> mixed_lines = ["Hello world", "你好世界", "Bonjour monde"]
        >>> filter_english_lines(mixed_lines)
        ['Hello world']
    """
    english_lines = []
    for line in lines:
        # Normalize whitespace and skip empty lines
        cleaned = line.strip()
        if not cleaned:
            continue
        
        # Apply multi-criteria English detection
        if is_probably_english(cleaned):
            english_lines.append(cleaned)
    
    return english_lines



def is_probably_english(text: str) -> bool:
    """Determine if text is likely English using multi-criteria analysis.
    
    This function combines several complementary approaches for robust language detection:
    1. Unicode script analysis (Latin character detection)
    2. ASCII ratio assessment (Western script preference)
    3. English letter frequency patterns
    4. Foreign language keyword exclusion
    
    The multi-criteria approach handles edge cases that single-method detection
    might miss, such as:
    - Mixed-script content (URLs, usernames in text)
    - Transliterated text
    - Code-switched content
    - Very short text snippets
    
    Args:
        text (str): Input text to analyze.
    
    Returns:
        bool: True if text is likely English, False otherwise.
    
    Example:
        >>> is_probably_english("Hello, how are you?")
        True
        >>> is_probably_english("你好，你在做什么？")
        False
    """
    latin_count = 0
    total_letters = 0
    ascii_count = 0
    total_chars = len(text)

    # Analyze character composition
    for char in text:
        if char.isascii():
            ascii_count += 1

        if char.isalpha():
            total_letters += 1

            try:
                # Check if character belongs to Latin script family
                # This covers extended Latin (accented characters, etc.)
                if "LATIN" in unicodedata.name(char):
                    latin_count += 1
            except ValueError:
                # Some characters may not have Unicode names
                continue

    # Handle edge cases: texts with no letters or empty strings
    if total_letters == 0 or total_chars == 0:
        return False
    
    # Calculate ratios for threshold-based classification
    latin_ratio = latin_count / total_letters
    ascii_ratio = ascii_count / total_letters
    
    # Apply combined criteria for robust classification
    return (
        latin_ratio > 0.8              # Predominantly Latin script
        and ascii_ratio > 0.7          # High ASCII content (Western text)
        and has_english_letter_profile(text)    # English letter patterns
        and not contains_common_foreign_words(text)  # Exclude obvious non-English
    )



def has_english_letter_profile(text: str) -> bool:
    """Check if text exhibits typical English letter frequency patterns.
    
    This function applies a simplified version of frequency analysis used in
    cryptography and linguistics. English text typically contains high frequencies
    of common letters (E, T, A, O, I, N), making this a useful discriminative feature.
    
    While not as sophisticated as full statistical models, this heuristic provides
    a lightweight check that's particularly effective for distinguishing English
    from languages with markedly different letter distributions.
    
    Args:
        text (str): Text to analyze for English letter patterns.
    
    Returns:
        bool: True if text shows English-like letter distribution.
    
    Note:
        This is a heuristic method - very short texts may not exhibit
        clear patterns, and some English texts may fail this test.
    """
    text = text.lower()
    # Count occurrences of most frequent English letters
    # Based on standard English letter frequency: E(12.7%), T(9.1%), A(8.2%), etc.
    common_letters_count = sum(text.count(c) for c in ['e', 't', 'a', 'o', 'i', 'n'])
    
    # Threshold chosen empirically - typical English text should have
    # at least a few instances of these common letters
    return common_letters_count >= 3


def contains_common_foreign_words(text: str) -> bool:
    """Detect presence of common non-English words as exclusion criteria.
    
    This function implements a simple but effective exclusion filter by checking
    for commonly occurring words in major European languages. While not exhaustive,
    this approach catches many obvious cases of non-English content.
    
    In production systems, this word list would typically be:
    - Much more comprehensive
    - Dynamically loaded from external resources
    - Language-specific with confidence scores
    
    Args:
        text (str): Text to check for foreign language indicators.
    
    Returns:
        bool: True if common foreign words are detected.
    
    Example:
        >>> contains_common_foreign_words("Bonjour, comment allez-vous?")
        True
        >>> contains_common_foreign_words("Hello, how are you?")
        False
    """
    # Curated list of high-frequency words from major European languages
    # These are words that rarely appear in English contexts
    foreign_keywords = [
        "bonjour", "monde", "merci",      # French
        "gracias", "por", "qué", "que",   # Spanish  
        "danke", "und", "nicht",          # German
    ]

    # Tokenize and normalize for case-insensitive matching
    words = text.lower().split()
    
    # Return True if any foreign keyword is found
    return any(word in foreign_keywords for word in words)


if __name__ == "__main__":
    """Demonstration of multilingual text filtering capabilities.
    
    This example showcases the filter's ability to distinguish English content
    from various non-English languages and edge cases commonly encountered
    in real-world text processing scenarios.
    """
    # Sample dataset representing typical multilingual content
    sample_lines = [
        "Hello, how are you?",                    # English - should pass
        "你好，你在做什么？",                        # Chinese - should be filtered
        "Bonjour tout le monde",                  # French - should be filtered  
        "12345 @#$%",                            # Numbers/symbols - should be filtered
        "The quick brown fox jumps over the lazy dog.",  # English - should pass
        "Это предложение на русском языке.",       # Russian - should be filtered
    ]

    # Apply the filtering algorithm
    english_only = filter_english_lines(sample_lines)

    # Display results for educational purposes
    print("Filtered English lines:")
    for line in english_only:
        print("-", line)