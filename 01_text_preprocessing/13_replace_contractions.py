"""
Text Preprocessing Utilities for Natural Language Processing

This module provides utilities for normalizing text data by expanding common
English contractions. Contraction expansion is a crucial preprocessing step
in NLP pipelines as it standardizes text representation, which can improve
the performance of downstream tasks such as tokenization, parsing, and
machine learning model training.

The implementation uses regular expressions for efficient pattern matching
and maintains proper capitalization during expansion, making it suitable
for integration into production ML pipelines.

Example:
    >>> text = "I can't believe it's working!"
    >>> expanded = replace_contractions(text)
    >>> print(expanded)
    "I cannot believe it is working!"
"""

import re
from typing import Dict, Match


# Mapping of common English contractions to their expanded forms
# This dictionary serves as the lookup table for contraction replacement
# and can be extended to include additional contractions as needed
CONTRACTIONS_MAP: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "isn't": "is not",
    "aren't": "are not",
    "didn't": "did not",
    "hasn't": "has not",
    "couldn't": "could not",
    "shouldn't": "should not"
}


def replace_contractions(text: str) -> str:
    """
    Replace common English contractions in text with their expanded forms.
    
    This function uses regular expressions to identify contractions and replace
    them while preserving original capitalization. The regex pattern is
    dynamically constructed from the CONTRACTIONS_MAP keys to ensure
    comprehensive coverage.
    
    Parameters
    ----------
    text : str
        Input text containing contractions to be expanded
        
    Returns
    -------
    str
        Text with contractions replaced by their expanded forms
        
    Examples
    --------
    >>> replace_contractions("I can't do this")
    'I cannot do this'
    
    >>> replace_contractions("They're going to the store")
    'They are going to the store'
    
    Notes
    -----
    The function preserves sentence capitalization - if a contraction starts
    a sentence and is capitalized, the expanded form will also be capitalized.
    This is important for maintaining proper grammar in downstream NLP tasks.
    """
    
    # Construct regex pattern that matches any contraction from our map
    # The \b word boundaries ensure we match complete words only
    # re.escape() protects against special regex characters in contractions
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(key) for key in CONTRACTIONS_MAP.keys()) + r')\b', 
        flags=re.IGNORECASE
    )

    def expand_match(match: Match) -> str:
        """
        Expand a single contraction match while preserving capitalization.
        
        This inner function handles the case-sensitive expansion of each
        matched contraction. It checks if the original text was capitalized
        and applies the same capitalization to the expanded form.
        
        Parameters
        ----------
        match : re.Match
            Regex match object containing the contraction to expand
            
        Returns
        -------
        str
            Expanded contraction with preserved capitalization
        """
        # Extract the matched text (e.g., "can't", "I'm")
        matched_text = match.group(0)
        
        # Convert to lowercase for dictionary lookup
        lower_matched = matched_text.lower()
        expanded = CONTRACTIONS_MAP.get(lower_matched)

        # If no expansion found, return original text unchanged
        if expanded is None:
            return matched_text
        
        # Preserve capitalization: if original started with uppercase,
        # capitalize the first letter of the expanded form
        if matched_text[0].isupper():
            return expanded.capitalize()
        
        return expanded

    # Apply the expansion function to all matches in the text
    return pattern.sub(expand_match, text)


if __name__ == "__main__":
    # Test cases demonstrating various contraction scenarios
    # Including edge cases like sentence-initial capitalization
    test_sentences = [
        "I can't believe it's already Friday!",  # Multiple contractions
        "You're going to love this movie, it's great!",  # Sentence-medial contractions
        "They didn't know what was going on.",  # Past tense contraction
        "I'm sure they won't mind.",  # Multiple contractions in one sentence
        "That's what I'm talking about!"  # Mixed positions
    ]

    # Process each test sentence and display results
    for sentence in test_sentences:
        print("Original:", sentence)
        print("Expanded:", replace_contractions(sentence))
        print("-" * 40)