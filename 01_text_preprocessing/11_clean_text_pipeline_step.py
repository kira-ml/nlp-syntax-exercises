"""Text Cleaning Pipeline: Modular Preprocessing for NLP

In this implementation, I demonstrate a modular and extensible text cleaning pipeline
for natural language processing (NLP) tasks. The pipeline showcases the composition
pattern commonly used in production ML systems, where preprocessing steps are applied
as a sequence of transformations.

Key Design Principles:
    - Modularity: Each function handles a single responsibility
    - Composability: Functions can be combined in different orders
    - Testability: Individual components can be tested in isolation
    - Extensibility: New cleaning steps can be easily added

Pipeline Components:
    1. Text normalization (lowercasing)
    2. Punctuation removal
    3. Emoji and special character removal
    4. Flexible pipeline composition

Production Applications:
    - Social media text preprocessing
    - Document cleaning for information retrieval
    - Feature engineering for text classification
    - Data preparation for language models

This modular approach enables both research experimentation and production deployment,
where different combinations of preprocessing steps may be required for different
downstream tasks.

Author: kira-ml
Date: August 4, 2025
"""

import string
import re
from typing import List, Callable



def to_lowercase(text: str) -> str:
    """Convert text to lowercase for consistent text normalization.
    
    Lowercasing is a fundamental preprocessing step that ensures consistent
    treatment of words regardless of their original capitalization. This is
    particularly important for:
    - Reducing vocabulary size in NLP models
    - Improving feature consistency in text classification
    - Normalizing user-generated content (social media, reviews)
    
    Args:
        text (str): Input text to normalize.
        
    Returns:
        str: Lowercased version of the input text.
        
    Example:
        >>> to_lowercase("Hello World!")
        'hello world!'
    """
    return text.lower()


# Demonstrate basic text normalization
sample = "Hello World!"
print(to_lowercase(sample))


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters from text.
    
    Punctuation removal is often necessary for:
    - Text classification tasks where punctuation adds noise
    - Search and information retrieval systems
    - Bag-of-words models where punctuation doesn't add semantic value
    - Social media text analysis where punctuation usage varies widely
    
    Note: This function removes ALL punctuation. For some applications,
    preserving certain punctuation (like periods for sentence boundaries)
    may be more appropriate.
    
    Args:
        text (str): Input text containing punctuation.
        
    Returns:
        str: Text with all punctuation characters removed.
        
    Example:
        >>> remove_punctuation("Hello, world! NLP is fun :)")
        'Hello world NLP is fun '
    """
    # Use generator expression for memory efficiency with large texts
    return ''.join(char for char in text if char not in string.punctuation)


# Demonstrate punctuation removal with varied punctuation
sample = "Hello, world! NLP is fun :)"
print(remove_punctuation(sample))



def remove_emojis(text: str) -> str:
    """Remove emojis and emoji-related Unicode sequences from text.
    
    This function handles comprehensive emoji removal including:
    - Standard emoji ranges (emoticons, symbols, transport, flags)
    - Variation selectors (U+FE0F) that modify emoji appearance
    - Zero-width joiners (U+200D) used in composite emoji
    - Miscellaneous symbols often treated as emoji (☀, ⭐, etc.)
    
    Emoji removal is essential for:
    - Traditional NLP models not trained on emoji
    - Text analysis requiring standardized character sets
    - Cross-platform consistency (emoji rendering varies)
    - Academic text processing where emoji are considered noise
    
    Args:
        text (str): Input text potentially containing emoji.
        
    Returns:
        str: Text with all emoji and related sequences removed.
        
    Example:
        >>> remove_emojis("Good morning ☀️! Let's build NLP pipelines 🚀")
        'Good morning ! Lets build NLP pipelines '
    """
    # Comprehensive emoji pattern covering major Unicode blocks
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons (😀-🙏)
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs (🌀-🗿)
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols (🚀-🛿)
        u"\U0001F1E0-\U0001F1FF"  # Flags (🇠-🇿)
        u"\u2600-\u26FF"          # Miscellaneous symbols (☀-⛿)
        "]+"                      # Match sequences of emoji
        "|"                       # OR
        u"[\uFE0F\u200D]",        # Variation selector & zero-width joiner
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', text)


# Demonstrate emoji removal with various emoji types
sample = "Good morning ☀️! Let's build NLP pipelines 🚀"
print(remove_emojis(sample))



def clean_text(text: str) -> str:
    """Apply a comprehensive text cleaning pipeline in a fixed sequence.
    
    This function demonstrates a standard cleaning workflow commonly used
    in text preprocessing. The order of operations is important:
    1. Lowercase first (preserves word boundaries before punctuation removal)
    2. Remove punctuation (eliminates noise characters)
    3. Remove emojis (final cleanup of special characters)
    
    Use this function when you need a consistent, repeatable cleaning process.
    For more flexibility in step ordering, see apply_pipeline().
    
    Args:
        text (str): Raw input text to clean.
        
    Returns:
        str: Cleaned text ready for downstream NLP tasks.
        
    Example:
        >>> clean_text("Let's build NLP tools 🚀 TODAY!!!")
        'lets build nlp tools  today'
    """
    # Apply transformations in sequence, each building on the previous
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_emojis(text)
    
    return text


# Demonstrate complete text cleaning with mixed content
sample = "Let's build NLP tools 🚀 TODAY!!!"
print(clean_text(sample))



def apply_pipeline(text: str, pipeline_steps: List[Callable[[str], str]]) -> str:
    """Apply a configurable sequence of text processing functions.
    
    This function implements the pipeline pattern, a fundamental design pattern
    in data processing systems. It enables:
    - Dynamic composition of preprocessing steps
    - Easy experimentation with different step combinations
    - Reusable preprocessing logic across different use cases
    - A/B testing of different cleaning strategies
    
    In production ML systems, this pattern allows for:
    - Configuration-driven preprocessing pipelines
    - Step-by-step debugging and validation
    - Conditional step application based on data characteristics
    
    Args:
        text (str): Input text to process.
        pipeline_steps (List[Callable[[str], str]]): Ordered list of functions
            to apply. Each function should take a string and return a string.
            
    Returns:
        str: Text after all pipeline steps have been applied.
        
    Example:
        >>> steps = [to_lowercase, remove_punctuation, remove_emojis]
        >>> apply_pipeline("Wow 🤩! Pipelines rock!!", steps)
        'wow  pipelines rock'
    """
    # Apply each transformation function in sequence
    for step in pipeline_steps:
        text = step(text)
    
    return text


# Demonstrate flexible pipeline composition
pipeline = [to_lowercase, remove_punctuation, remove_emojis]
sample = "Wow 🤩! Pipelines make everything clean & modular!!"

print(apply_pipeline(sample, pipeline))
