"""Text Normalization: Comparing Lemmatization and Stemming Approaches

In this implementation, I demonstrate and compare different text normalization techniques
commonly used in NLP pipelines. The focus is on two main approaches:

1. Lemmatization (using spaCy):
   - Context-aware reduction to base dictionary form
   - Morphological analysis for accurate word reduction
   - Computationally more expensive but linguistically accurate

2. Stemming (using NLTK):
   - Rule-based suffix stripping (Porter and Snowball algorithms)
   - Faster but potentially less accurate
   - Language-specific rules for English

The comparison helps understand when to use each approach:
- Lemmatization: Best for tasks requiring semantic precision
- Stemming: Suitable for search engines and large-scale text processing

Author: kira-ml
Date: July 31, 2025
"""

import spacy 
from nltk.stem import PorterStemmer, SnowballStemmer
from typing import List

# Initialize spaCy with the small English model
# Note: Requires running 'python -m spacy download en_core_web_sm' first
nlp = spacy.load("en_core_web_sm")


def lemmatize_with_spacy(text: str) -> List[str]:
    """Extract lemmas from text using spaCy's contextual lemmatization.
    
    This function leverages spaCy's advanced NLP pipeline to perform lemmatization,
    which considers the word's context and part-of-speech when reducing to base form.
    Unlike stemming, lemmatization ensures the result is a valid dictionary word.
    
    Args:
        text (str): Input text to lemmatize.
    
    Returns:
        List[str]: List of lemmatized words, maintaining their contextual meaning.
    
    Example:
        >>> lemmatize_with_spacy("The children were playing")
        ['the', 'child', 'be', 'play']  # Note: 'children' -> 'child'
    """
    # Process text through spaCy's pipeline
    doc = nlp(text)
    # Extract lemma for each token using spaCy's linguistic analysis
    return [token.lemma_ for token in doc]


# Sample text showcasing various morphological forms
example_text = "The children were playing games and running happily"

# Demonstrate lemmatization
lemmas = lemmatize_with_spacy(example_text)
print("Lemmas:", lemmas)  # Shows context-aware base forms



def stem_with_porter(text: str) -> List[str]:
    """Apply Porter stemming algorithm to input text.
    
    The Porter stemmer uses a set of rules to strip suffixes, making it:
    - Faster than lemmatization
    - Language-specific (designed for English)
    - Sometimes producing non-dictionary words
    
    Args:
        text (str): Input text to stem.
    
    Returns:
        List[str]: List of stemmed words.
    
    Example:
        >>> stem_with_porter("running happily")
        ['run', 'happili']  # Note: 'happily' -> 'happili'
    """
    # Initialize Porter stemmer (one-time cost)
    stemmer = PorterStemmer()
    # Apply stemming to each word independently
    return [stemmer.stem(word) for word in text.split()]


def stem_with_snowball(text: str) -> list[str]:
    """Apply Snowball (Porter2) stemming algorithm to input text.
    
    Snowball is an improved version of the Porter stemmer:
    - More conservative in its rules
    - Better handling of English vocabulary
    - Often produces more natural stems
    
    Args:
        text (str): Input text to stem.
    
    Returns:
        List[str]: List of stemmed words.
    
    Example:
        >>> stem_with_snowball("running happily")
        ['run', 'happili']  # Similar to Porter but with improvements
    """
    # Initialize Snowball stemmer with English language rules
    stemmer = SnowballStemmer("english")
    # Apply stemming to each word independently
    return [stemmer.stem(word) for word in text.split()]



porter_stems = stem_with_porter(example_text)
snowball_stems = stem_with_snowball(example_text)

print("Porter Stem :", porter_stems)
print("Snowball Stems :", snowball_stems)



def compare_normalization_methods(text: str) -> None:
    """Compare different text normalization approaches side by side.
    
    This function provides a visual comparison of how different normalization
    techniques handle the same text. It's particularly useful for understanding:
    - How lemmatization preserves word meaning
    - How stemmers can produce non-dictionary words
    - When each approach might be more appropriate
    
    Args:
        text (str): Input text to normalize using different methods.
    
    Example output:
        Original     Lemmatized    Porter Stem   Snowball Stem
        ---------------------------------------------------
        running      run           run           run
        happily      happily       happili       happili
    """
    print("\nInput text:")
    print(text)
    
    print("\nNormalization comparison:")
    
    # Split text into words for comparison
    words = text.split()
    
    # Apply each normalization method
    lemmas = lemmatize_with_spacy(text)     # Context-aware
    porter = stem_with_porter(text)          # Basic stemming
    snowball = stem_with_snowball(text)      # Improved stemming
    
    # Format and display results in aligned columns
    print(f"{'Original':<15}{'Lemmatized':<15}{'Porter Stem':<15}{'Snowball Stem':<15}")
    print("-" * 60)
    for o, l, p, s in zip(words, lemmas, porter, snowball):
        print(f"{o:<15}{l:<15}{p:<15}{s:<15}")


# Demonstrate normalization methods with a comprehensive example
compare_normalization_methods("The children were playing games and running happily")