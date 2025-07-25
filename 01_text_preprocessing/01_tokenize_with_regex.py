
"""
01_tokenize_with_regex.py

In this implementation, I demonstrate a flexible and extensible approach to tokenizing text using regular expressions—a foundational technique in natural language processing (NLP) pipelines. This script is designed to capture URLs, time expressions, contractions, and general word tokens as distinct units, reflecting real-world text preprocessing needs for both academic research and production ML systems.

The code is structured for clarity and modularity, making it suitable for integration into larger data cleaning or feature extraction workflows. I include input normalization and pattern prioritization to ensure robust tokenization across diverse text sources.
"""

import re

def tokenize_text(text: str) -> list:
    """
    Tokenize input text into URLs, time expressions, contractions, and words using prioritized regex patterns.

    This function is intended for use in NLP preprocessing pipelines where granular tokenization is required for downstream tasks such as entity recognition, information extraction, or language modeling.

    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    list of str
        List of tokens extracted from the input text, preserving URLs, times, and contractions as single units.

    Notes
    -----
    - Pattern order is critical: more specific patterns (URLs, times, contractions) precede general word tokens to avoid premature splitting.
    - For multilingual or domain-specific corpora, consider extending the regex patterns or integrating with more advanced tokenizers.
    """
    text = clean_text(text)

    # Regex patterns for different token types
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'  # Capture URLs
    time_pattern = r'\d{1,2}:\d{2}(?:\s?[ap]\.m\.?|\s?[AP]\.M\.?)?'  # Capture times like '12:30 p.m.'
    contraction_pattern = r"\b\w+['’]\w+\b"  # Capture contractions like "Don't"
    word_pattern = r'\w+'  # General word tokens

    # Prioritize specific patterns to avoid splitting complex tokens
    pattern = f"{url_pattern}|{time_pattern}|{contraction_pattern}|{word_pattern}"
    tokens = re.findall(pattern, text)
    return tokens


def clean_text(text: str) -> str:
    """
    Normalize input text by standardizing quotes and trimming whitespace.

    This function prepares text for tokenization by replacing curly quotes with straight quotes and removing leading/trailing spaces. Such normalization is essential for consistent regex matching and downstream NLP tasks.

    Parameters
    ----------
    text : str
        The input string to be cleaned.

    Returns
    -------
    str
        The cleaned string, ready for tokenization.
    """
    # Replace curly quotes with straight quotes for regex compatibility
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text.strip()


if __name__ == "__main__":
    # Example usage: tokenizing a sample sentence with mixed token types
    sample = "This is a test: visit https://youtube.com at 12:30 p.m. Don't miss it!"
    tokens = tokenize_text(sample)

    print(tokens)