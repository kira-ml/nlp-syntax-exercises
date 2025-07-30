"""
08_stem_with_nltk.py

In this script, I demonstrate how to apply two popular stemming algorithms—
PorterStemmer and LancasterStemmer—provided by the NLTK library to a tokenized
text corpus. Stemming is a fundamental preprocessing step in many NLP pipelines,
particularly for search engines, text classification, and feature standardization.

The goal here is to observe how different stemmers reduce words to their base forms.
This script is structured for educational purposes and can be extended or embedded
into larger NLP workflows for batch text preprocessing.

Author: kira-ml
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer

# Download required tokenizer model from NLTK (used once; cached afterward)
nltk.download('punkt')


def get_tokenized_text():
    """
    Tokenizes a sample sentence using NLTK's word tokenizer.

    Returns
    -------
    list of str
        A list of word-level tokens extracted from the sample sentence.
    """
    text = "I was running faster than usual, hoping they wouldn't catch me easily"
    tokens = word_tokenize(text)

    return tokens


def apply_porter_stemming(tokens):
    """
    Applies the Porter stemming algorithm to a list of tokens.

    Parameters
    ----------
    tokens : list of str
        The input list of tokenized words.

    Returns
    -------
    list of str
        A list where each word is reduced to its Porter stem.
    """
    porter = PorterStemmer()  # Instantiate once to avoid redundant object creation
    return [porter.stem(word) for word in tokens]


def apply_lancaster_stemming(tokens):
    """
    Applies the Lancaster stemming algorithm to a list of tokens.

    Parameters
    ----------
    tokens : list of str
        The input list of tokenized words.

    Returns
    -------
    list of str
        A list where each word is reduced to its Lancaster stem.
    """
    lancaster = LancasterStemmer()  # More aggressive stemmer than Porter
    return [lancaster.stem(word) for word in tokens]


if __name__ == "__main__":
    # Tokenize the input sentence
    tokens = get_tokenized_text()

    # Apply two different stemming algorithms to compare outputs
    porter_output = apply_porter_stemming(tokens)
    lancaster_output = apply_lancaster_stemming(tokens)

    # Display results in a tabular format for easy comparison
    print(f"{'Original':<15} {'Porter':<15} {'Lancaster'}")
    print("-" * 45)
    for original, porter_w, lancaster_w in zip(tokens, porter_output, lancaster_output):
        print(f"{original:<15} {porter_w:<15} {lancaster_w}")

    # For debugging or inspection purposes
    print("\nTokenized words:", tokens)
