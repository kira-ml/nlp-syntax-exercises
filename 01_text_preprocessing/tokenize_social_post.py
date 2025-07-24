"""
tokenize_social_post.py

This module demonstrates a clean, idiomatic approach to word tokenization using NLTK's `word_tokenize`.
In this implementation, I showcase a reusable function designed for tokenizing social media content 
into word-level tokens, including punctuation and hashtags as separate units.

The code is structured to support future integration into larger NLP pipelines, such as preprocessing
modules for sentiment analysis, topic modeling, or LLM fine-tuning on informal text sources.

Dependencies:
    - nltk: Natural Language Toolkit (https://www.nltk.org/)
      Install via: `pip install nltk`

This script assumes the 'punkt' tokenizer model is available and will download it automatically if not.
"""

import nltk
from nltk.tokenize import word_tokenize

# Ensure the Punkt tokenizer model is available for sentence and word tokenization.
# This model provides rules learned from a large corpus and is required by `word_tokenize`.
nltk.download('punkt')


def tokenize_sentence(sentence):
    """
    Tokenize a single sentence into words and punctuation using NLTK's pretrained Punkt tokenizer.

    This function is robust to edge cases such as empty strings or incorrect input types,
    and it ensures output cleanliness suitable for downstream NLP tasks.

    Parameters
    ----------
    sentence : str
        The input sentence to be tokenized. Typically a single sentence or short text,
        such as a tweet or social media post.

    Returns
    -------
    List[str]
        A list of tokens representing words and punctuation marks from the input string.

    Raises
    ------
    ValueError
        If the input is not a string.

    Notes
    -----
    - The tokenizer is pre-trained and rule-based, making it suitable for informal text.
    - For multilingual or domain-specific corpora, you may need to customize tokenization further.
    """
    if not isinstance(sentence, str):
        raise ValueError("Input must be string.")

    # Skip tokenization if input is an empty or whitespace-only string
    if not sentence.strip():
        return []

    # Use NLTK's word_tokenize which preserves punctuation and handles contractions
    return word_tokenize(sentence)


if __name__ == "__main__":
    # Example use-case: tokenizing informal, hashtag-laden text from a social media post.
    social_post = "Just checked out the new cafe downtown! #foodie #Manila"

    tokens = tokenize_sentence(social_post)

    print("Original post:", social_post)
    print("Tokenized output:", tokens)
