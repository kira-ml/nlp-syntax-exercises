"""Text Preprocessing: Emoji and Symbol Processing in Natural Language Processing
    
This module demonstrates professional-grade text preprocessing techniques focusing on emoji
and special symbol handling - a critical component in modern NLP pipelines. The implementation
showcases how to effectively manage Unicode ranges for accurate emoji and symbol detection,
which is particularly important for social media text analysis and sentiment detection tasks.

The module provides utilities for:
    - Extracting emojis from text (useful for emoji-based sentiment analysis)
    - Removing emojis (for traditional NLP tasks requiring clean text)
    - Removing technical symbols (for specialized text normalization)
    - Combined emoji and symbol removal (for comprehensive text cleaning)

Key Technical Concepts:
    - Unicode range-based pattern matching
    - Regular expression compilation for performance
    - Proper handling of supplemental Unicode planes
    
Example:
    >>> text = "Hello! 😊 → World 🌍"
    >>> extract_emojis(text)
    ['😊', '🌍']
    >>> remove_symbols_and_emojis(text)
    'Hello! World'

Author: kira-ml
Date: July 28, 2025
"""

import re


def extract_emojis(text: str) -> list:
    """Extract all emojis and emoji-like symbols from the input text.
    
    This function identifies and extracts both standard emoji characters and
    commonly used symbolic emoji (like ☀, ☕) that are treated as emojis in
    modern digital communication. The Unicode ranges are carefully selected
    to capture:
        - Standard emoji sets (emoticons, objects, flags)
        - Weather and celestial symbols (☀, ⭐, etc.)
        - Common symbolic emoji (☕, ♥, etc.)
        - Supplemental pictographic symbols
    
    Args:
        text (str): The input text containing potential emoji characters.
    
    Returns:
        list: A list of all emoji characters found in the text.
    
    Example:
        >>> extract_emojis("Good morning! ☕ 🌞")
        ['☕', '🌞']
    """
    # Define comprehensive emoji pattern including modern symbolic emoji
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"   # Basic emoticons
        "\U0001F300-\U0001F5FF"   # Symbols & pictographs
        "\U0001F680-\U0001F6FF"   # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"   # Flags
        "\U00002700-\U000027BF"   # Dingbats
        "\U0001F900-\U0001F9FF"   # Supplemental symbols
        "\u2600-\u26FF"           # Weather & commonly used symbols (☀, ☕, etc.)
        "]", flags=re.UNICODE)
    
    return emoji_pattern.findall(text)



def remove_emojis(text: str) -> str:
    """Remove all emoji characters from the input text.
    
    This function uses the same Unicode ranges as extract_emojis() but applies them
    for removal rather than extraction. In production NLP pipelines, this is particularly
    useful for:
        - Preparing text for traditional NLP models not trained on emoji
        - Standardizing text input across different sources
        - Cleaning text for formal document processing
    
    Args:
        text (str): The input text containing potential emoji characters.
    
    Returns:
        str: The input text with all emoji characters removed.
    
    Example:
        >>> remove_emojis("Hello 👋 World 🌍!")
        'Hello  World !'
    """
    # Define comprehensive emoji pattern using Unicode ranges
    # Note: The pattern is compiled for performance as it's likely
    # to be used repeatedly in text processing pipelines
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Basic emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental symbols
        "\u2600-\u26FF"          # Weather & other symbols (☀, ☕, etc.)
        "]", flags=re.UNICODE)
    
    # Use sub() for efficient replacement of all matches with empty string
    return emoji_pattern.sub('', text)




def remove_symbol(text: str) -> str:
    """Remove technical and miscellaneous symbols from the input text.
    
    This function focuses specifically on non-emoji symbols that commonly appear
    in technical or formatted text, including:
        - Arrows and mathematical symbols
        - Technical and drawing symbols
        - Miscellaneous dingbats
    
    In production systems, this function is particularly useful for:
        - Cleaning technical documentation
        - Processing markdown or formatted text
        - Standardizing input for downstream NLP tasks
    
    Args:
        text (str): The input text containing potential symbol characters.
    
    Returns:
        str: The input text with all specified symbols removed.
    
    Example:
        >>> remove_symbol("A → B ⌚ C")
        'A  B  C'
    """
    # Define pattern for technical and decorative symbols
    # These ranges are carefully chosen to avoid removing
    # common punctuation or mathematical operators
    symbol_pattern = re.compile(
        r'['
        '\u2190-\u21FF'  # Arrows (e.g., ←, →)
        '\u2300-\u23FF'  # Misc Technical (e.g., ⌚, ⌛)
        '\u2700-\u27BF'  # Dingbats (e.g., ✂, ✎)
        # Note: \u2600-\u26FF range is handled by emoji functions as it contains
        # modern symbolic emoji like ☀ and ☕
        ']', flags=re.UNICODE
    )

    return symbol_pattern.sub('', text)


def remove_symbols_and_emojis(text: str) -> str:
    """Remove all emojis, emoji-like symbols, and technical symbols from the input text.
    
    This function provides comprehensive text cleaning by combining:
        - Standard emoji removal (emoticons, objects, flags)
        - Symbolic emoji removal (☀, ☕, etc.)
        - Technical symbol removal (arrows, technical symbols)
    
    It's optimized for production use where complete text cleaning is needed in a
    single pass, making it more efficient than calling separate functions.
    
    Use Cases:
        - Preparing text for traditional NLP models
        - Cleaning technical documentation
        - Standardizing text for analysis
    
    Args:
        text (str): The input text containing potential emojis and symbols.
    
    Returns:
        str: The input text with all emojis and symbols removed.
    
    Example:
        >>> remove_symbols_and_emojis("Hello ☕👋 → World 🌍!")
        'Hello  World !'
    """
    # Comprehensive pattern for all emoji types and technical symbols
    pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\u2600-\u26FF"          # Misc symbols
        "\u2300-\u23FF"          # Misc technical
        "\u2190-\u21FF"          # Arrows
        "]", flags=re.UNICODE)

    

    return pattern.sub('', text)


def main():
    """Demonstrate the emoji and symbol processing capabilities.
    
    This function provides a comprehensive example of all text processing
    utilities in this module. It uses a realistic social media-style text
    that contains various types of emojis and symbols.
    """
    # Sample text combining casual language, emojis, and technical symbols
    sample_text = "Let's grab a ☕ and chat 😊! #MondayMotivation 😎✌️ ← → ⌚ ✂ ☀"

    # Demonstrate each text processing capability
    print("Original text:")
    print(sample_text)

    print("\nExtracted Emojis:")
    print(extract_emojis(sample_text))  # Shows emoji extraction capability

    print("\nText without emojis:")
    print(remove_emojis(sample_text))   # Demonstrates emoji removal

    print("\nText without symbols:")
    print(remove_symbol(sample_text))    # Shows symbol-specific removal

    print("\nText without symbols and emojis:")
    print(remove_symbols_and_emojis(sample_text))  # Shows combined cleaning


if __name__ == "__main__":
    # Execute the demonstration if run as a script
    main()

