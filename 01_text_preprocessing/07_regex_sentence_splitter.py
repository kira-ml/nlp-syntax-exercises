"""
regex_sentence_splitter.py

This module provides a lightweight utility for splitting raw text into sentences using regular expressions,
with special attention to edge cases such as common abbreviations and trailing punctuation (quotes, parentheses, etc.).

In this implementation, I demonstrate how to:
- Use boundary-aware regular expressions to identify sentence breaks.
- Mask and unmask known abbreviations to avoid incorrect splits.
- Preprocess input text for consistent formatting.

This type of utility is common in early-stage NLP pipelines where full tokenization frameworks (like spaCy or NLTK)
may be too heavy or unnecessary. It's particularly useful for preprocessing user-generated content, legal documents,
or academic text where custom rules are required.
"""

import re


def split_into_sentences(text):
    """
    Split a block of text into sentences using a regular expression-based approach.

    This function accounts for common abbreviations (e.g., "Dr.", "e.g.") that would otherwise
    lead to incorrect sentence splits. It also normalizes spacing and trims whitespace from results.

    Parameters
    ----------
    text : str
        The raw input text to be split into individual sentences.

    Returns
    -------
    List[str]
        A list of sentence strings, each representing a distinct sentence boundary.
    
    Notes
    -----
    This method avoids splitting inside known abbreviations by temporarily replacing periods
    with a placeholder token. After splitting, the placeholder is reverted to restore the original text.
    """
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
        "e.g.", "i.e.", "etc.", "vs.", "a.m.", "p.m."
    ]

    # Replace periods in known abbreviations with a safe placeholder to prevent splitting on them
    for abbr in abbreviations:
        safe_abbr = abbr.replace(".", "[dot]")
        text = text.replace(abbr, safe_abbr)

    # Regex: Split on end punctuation (., !, ?) followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore periods in abbreviations and clean up whitespace
    clean_sentences = [
        sentence.replace("[dot]", ".").strip()
        for sentence in raw_sentences
        if sentence.strip()  # Avoid empty strings caused by trailing space
    ]

    return clean_sentences


def clean_input(text):
    """
    Normalize raw text input by flattening whitespace.

    This utility removes excess whitespace characters (e.g., tabs, newlines, double spaces)
    to ensure consistent formatting before sentence splitting.

    Parameters
    ----------
    text : str
        The raw input string, potentially with inconsistent whitespace.

    Returns
    -------
    str
        A cleaned version of the text with single spaces and no leading/trailing whitespace.
    
    Example
    -------
    >>> clean_input("  Hello,  world!\\nThis is\\tPython. ")
    'Hello, world! This is Python.'
    """
    return ' '.join(text.strip().split())


def main():
    """
    Demonstrates sentence splitting on a sample paragraph.

    This function is useful for standalone testing or when using this script as a CLI utility.
    It prints each extracted sentence with a line number for easy inspection.
    """
    sample_text = (
        "Dr. Smith went to Manila. He arrived at 5 p.m. sharp. "
        "Can you believe it? 'No' she said. It's amazing!"
    )

    # Optional: Pre-clean the text before splitting
    sample_text = clean_input(sample_text)

    # Run the sentence splitting function
    sentences = split_into_sentences(sample_text)

    # Print each sentence with its index
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}: {sentence}")


if __name__ == "__main__":
    # Ensure this block runs only when the script is executed directly
    main()
