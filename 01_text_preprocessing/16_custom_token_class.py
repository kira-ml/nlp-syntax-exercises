"""
16_custom_token_class.py

Educational Exercise: Custom Token Class for NLP

This module demonstrates how to create a lightweight, structured Token class
to represent linguistic units in natural language processing pipelines.

Learning Objectives:
- Understand object-oriented programming (OOP) principles in NLP contexts
- Learn to use Python dataclasses for clean, boilerplate-free data containers
- Explore attribute access patterns and metadata management
- Simulate structured token representations used in real NLP libraries

Key Concepts:
- Token: The fundamental unit of text processing, representing a word, punctuation, or other linguistic unit
- POS Tagging: Part-of-speech tagging categorizes tokens by grammatical role (noun, verb, adjective, etc.)
- Lemmatization: Reducing tokens to their base or dictionary form
- Metadata: Additional linguistic annotations that enrich token representation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class POS(Enum):
    """
    Part-of-Speech (POS) tags using Universal Dependencies taxonomy.
    
    This enum defines common grammatical categories that help in understanding
    the syntactic role of each token in a sentence.
    
    Example usage:
        >>> token.pos = POS.NOUN
        >>> token.pos == POS.VERB
        False
    """
    NOUN = "NOUN"           # People, places, things, ideas
    VERB = "VERB"           # Actions, states, occurrences
    ADJ = "ADJ"             # Describes nouns
    ADV = "ADV"             # Modifies verbs, adjectives, other adverbs
    PRON = "PRON"           # Replaces nouns (he, she, it)
    DET = "DET"             # Articles and determiners (the, a, this)
    ADP = "ADP"             # Prepositions and postpositions (in, on, of)
    CONJ = "CONJ"           # Connecting words (and, but, or)
    NUM = "NUM"             # Numerals
    PUNCT = "PUNCT"         # Punctuation marks
    X = "X"                 # Other, undefinable
    SPACE = "SPACE"         # Whitespace characters


class Dependency(Enum):
    """
    Syntactic dependency relations between tokens.
    
    These labels describe how tokens relate to each other in a sentence's
    syntactic structure, forming a dependency parse tree.
    
    Example:
        In "The cat sat on the mat", "cat" is the nsubj (nominal subject) of "sat"
    """
    ROOT = "root"           # The main predicate of the sentence
    NSUBJ = "nsubj"         # Nominal subject
    DOBJ = "dobj"           # Direct object
    PREP = "prep"           # Prepositional modifier
    DET = "det"             # Determiner
    AMOD = "amod"           # Adjectival modifier


@dataclass
class Token:
    """
    A structured representation of a linguistic token with linguistic annotations.
    
    This class encapsulates all the information NLP systems typically store about
    each token (word, punctuation, etc.) in a text. It uses dataclasses to
    automatically generate common methods like __init__, __repr__, and __eq__.
    
    Attributes:
        text (str): The original surface form of the token as it appears in text
        pos (Optional[POS]): Part-of-speech tag indicating grammatical category
        lemma (Optional[str]): Base or dictionary form of the token
        is_alpha (bool): Whether the token consists of alphabetic characters
        is_stop (bool): Whether the token is a stop word (common, low-meaning word)
        dependency (Optional[Dependency]): Syntactic dependency relation
        head (Optional['Token']): The governing token in dependency parse
        metadata (Dict[str, Any]): Additional linguistic features and annotations
    
    Example:
        >>> token = Token(
        ...     text="running",
        ...     pos=POS.VERB,
        ...     lemma="run",
        ...     is_alpha=True,
        ...     is_stop=False
        ... )
        >>> print(f"'{token.text}' -> '{token.lemma}' ({token.pos.value})")
        'running' -> 'run' (VERB)
    """
    
    # Required attribute: the actual text of the token
    text: str
    
    # Linguistic annotations (optional - may be None if not yet processed)
    pos: Optional[POS] = None
    lemma: Optional[str] = None
    dependency: Optional[Dependency] = None
    head: Optional['Token'] = None
    
    # Boolean flags for quick token filtering and analysis
    is_alpha: bool = field(default=False)
    is_stop: bool = field(default=False)
    is_punct: bool = field(default=False)
    is_space: bool = field(default=False)
    
    # Additional metadata storage for extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Automatically called after __init__ to set derived properties.
        
        This method ensures that boolean flags are consistent with the token's
        text content and POS tag. It demonstrates how to maintain data integrity
        in dataclasses.
        """
        # Auto-detect punctuation if not explicitly set and POS is PUNCT
        if self.pos == POS.PUNCT and not self.is_punct:
            self.is_punct = True
            
        # Auto-detect alphabetic tokens based on text content
        if not self.is_alpha and self.text.isalpha():
            self.is_alpha = True
            
        # Auto-detect space tokens
        if not self.is_space and self.text.isspace():
            self.is_space = True
            self.is_alpha = False  # Spaces aren't alphabetic
    
    @property
    def normalized_text(self) -> str:
        """
        Return a normalized version of the token text.
        
        This property demonstrates computed attributes that derive information
        from existing data. Normalization typically includes lowercasing and
        is useful for case-insensitive comparisons.
        
        Returns:
            str: Lowercase version of the token text
            
        Example:
            >>> token = Token(text="Apple")
            >>> token.normalized_text
            'apple'
        """
        return self.text.lower()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata to the token.
        
        This method shows how to extend token information dynamically.
        Metadata can store any additional linguistic features like:
        - Named Entity Recognition tags
        - Sentiment scores  
        - Custom domain-specific annotations
        
        Args:
            key (str): The metadata key/name
            value (Any): The value to store
            
        Example:
            >>> token.add_metadata("ner", "PERSON")
            >>> token.add_metadata("sentiment", 0.8)
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieve metadata with a default value if key doesn't exist.
        
        This method demonstrates defensive programming practices by
        providing a safe way to access potentially missing metadata.
        
        Args:
            key (str): The metadata key to retrieve
            default (Any): Default value if key not found
            
        Returns:
            Any: The metadata value or default if not found
            
        Example:
            >>> token.get_metadata("ner", "UNKNOWN")
            'PERSON'
            >>> token.get_metadata("nonexistent", "default_value")
            'default_value'
        """
        return self.metadata.get(key, default)
    
    def has_annotation(self, annotation_type: str) -> bool:
        """
        Check if the token has a specific type of linguistic annotation.
        
        This method provides a unified way to check for various types of
        annotations, making client code cleaner and more maintainable.
        
        Args:
            annotation_type (str): Type of annotation to check for.
                Supported values: 'pos', 'lemma', 'dependency'
                
        Returns:
            bool: True if the specified annotation is present
            
        Example:
            >>> token = Token(text="run", pos=POS.VERB)
            >>> token.has_annotation('pos')
            True
            >>> token.has_annotation('lemma')  
            False
        """
        annotation_map = {
            'pos': self.pos is not None,
            'lemma': self.lemma is not None, 
            'dependency': self.dependency is not None
        }
        return annotation_map.get(annotation_type, False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the token to a dictionary representation.
        
        This is useful for serialization, debugging, or integration with
        systems that expect dictionary data. It demonstrates how to create
        comprehensive data export functionality.
        
        Returns:
            Dict[str, Any]: Dictionary containing all token attributes
            
        Example:
            >>> token = Token(text="running", lemma="run", pos=POS.VERB)
            >>> token_dict = token.to_dict()
            >>> print(token_dict['text'], token_dict['lemma'])
            running run
        """
        return {
            'text': self.text,
            'pos': self.pos.value if self.pos else None,
            'lemma': self.lemma,
            'is_alpha': self.is_alpha,
            'is_stop': self.is_stop,
            'is_punct': self.is_punct,
            'is_space': self.is_space,
            'dependency': self.dependency.value if self.dependency else None,
            'head_text': self.head.text if self.head else None,
            'metadata': self.metadata.copy()  # Return a copy to prevent external modification
        }
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the token.
        
        This method is called by print() and str() conversions. It provides
        a concise, informative summary of the token's key attributes.
        
        Returns:
            str: Formatted string showing text, lemma, and POS
            
        Example:
            >>> token = Token(text="books", lemma="book", pos=POS.NOUN)
            >>> print(token)
            Token('books' -> 'book' / NOUN)
        """
        lemma_info = f" -> '{self.lemma}'" if self.lemma else ""
        pos_info = f" / {self.pos.value}" if self.pos else ""
        return f"Token('{self.text}'{lemma_info}{pos_info})"
    
    def __repr__(self) -> str:
        """
        Return a detailed, unambiguous string representation.
        
        This method is used for debugging and should contain enough information
        to recreate the object. It's more detailed than __str__.
        
        Returns:
            str: Detailed representation including all key fields
        """
        return (f"Token(text='{self.text}', pos={self.pos}, lemma='{self.lemma}', "
                f"is_alpha={self.is_alpha}, is_stop={self.is_stop})")


def demonstrate_token_class():
    """
    Comprehensive demonstration of the Token class functionality.
    
    This function showcases various features of the Token class through
    practical examples, illustrating real-world NLP usage patterns.
    """
    print("=" * 70)
    print("CUSTOM TOKEN CLASS DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. BASIC TOKEN CREATION")
    print("-" * 40)
    
    # Create a simple token with just text
    simple_token = Token(text="hello")
    print(f"Simple token: {simple_token}")
    print(f"  is_alpha: {simple_token.is_alpha}")
    print(f"  normalized: '{simple_token.normalized_text}'")
    
    # Create a fully annotated token
    annotated_token = Token(
        text="running",
        pos=POS.VERB,
        lemma="run", 
        is_alpha=True,
        is_stop=False
    )
    print(f"\nAnnotated token: {annotated_token}")
    print(f"  Has POS annotation: {annotated_token.has_annotation('pos')}")
    print(f"  Has lemma annotation: {annotated_token.has_annotation('lemma')}")
    
    print("\n2. PART-OF-SPEECH TAGGING EXAMPLES")
    print("-" * 40)
    
    # Demonstrate different POS tags
    pos_examples = [
        Token(text="cat", pos=POS.NOUN),
        Token(text="quickly", pos=POS.ADV),
        Token(text="beautiful", pos=POS.ADJ),
        Token text="the", pos=POS.DET, is_stop=True),
        Token(text="!", pos=POS.PUNCT, is_punct=True)
    ]
    
    for token in pos_examples:
        print(f"  {token.text:12} -> {token.pos.value if token.pos else 'None':8} "
              f"(stop: {token.is_stop}, alpha: {token.is_alpha})")
    
    print("\n3. METADATA MANAGEMENT")
    print("-" * 40)
    
    # Create a token and add custom metadata
    entity_token = Token(text="Microsoft", pos=POS.NOUN, is_alpha=True)
    entity_token.add_metadata("ner", "ORGANIZATION")
    entity_token.add_metadata("sentiment", 0.1)
    entity_token.add_metadata("frequency", "high")
    
    print(f"Token: {entity_token.text}")
    print(f"  NER: {entity_token.get_metadata('ner', 'UNKNOWN')}")
    print(f"  Sentiment: {entity_token.get_metadata('sentiment', 0.0)}")
    print(f"  Custom field: {entity_token.get_metadata('custom', 'not_found')}")
    
    print("\n4. DEPENDENCY PARSING SIMULATION")
    print("-" * 40)
    
    # Simulate a dependency parse for "The cat sat"
    the_token = Token(text="The", pos=POS.DET, is_stop=True)
    cat_token = Token(text="cat", pos=POS.NOUN, is_alpha=True)
    sat_token = Token(text="sat", pos=POS.VERB, lemma="sit", is_alpha=True)
    
    # Set up dependency relations
    the_token.dependency = Dependency.DET
    the_token.head = cat_token
    
    cat_token.dependency = Dependency.NSUBJ  
    cat_token.head = sat_token
    
    sat_token.dependency = Dependency.ROOT
    sat_token.head = None  # Root has no head
    
    # Display dependency information
    tokens = [the_token, cat_token, sat_token]
    for token in tokens:
        head_text = token.head.text if token.head else "ROOT"
        dep_type = token.dependency.value if token.dependency else "None"
        print(f"  '{token.text}' --{dep_type:6}--> '{head_text}'")
    
    print("\n5. SERIALIZATION AND DATA EXPORT")
    print("-" * 40)
    
    # Show dictionary representation
    sample_token = Token(
        text="transformers",
        pos=POS.NOUN,
        lemma="transformer",
        is_alpha=True,
        is_stop=False
    )
    sample_token.add_metadata("domain", "AI/ML")
    
    token_dict = sample_token.to_dict()
    print("Dictionary representation:")
    for key, value in token_dict.items():
        print(f"  {key:15}: {value}")
    
    print("\n6. ADVANCED USAGE PATTERNS")
    print("-" * 40)
    
    # Demonstrate token filtering and analysis
    sentence_tokens = [
        Token(text="The", pos=POS.DET, is_stop=True),
        Token(text="quick", pos=POS.ADJ, is_alpha=True),
        Token(text="brown", pos=POS.ADJ, is_alpha=True),
        Token(text="fox", pos=POS.NOUN, is_alpha=True),
        Token(text="jumps", pos=POS.VERB, lemma="jump", is_alpha=True),
        Token(text="over", pos=POS.ADP, is_stop=True),
        Token(text="the", pos=POS.DET, is_stop=True),
        Token(text="lazy", pos=POS.ADJ, is_alpha=True),
        Token(text="dog", pos=POS.NOUN, is_alpha=True),
        Token(text=".", pos=POS.PUNCT, is_punct=True)
    ]
    
    # Filter out stop words and punctuation
    content_tokens = [t for t in sentence_tokens if not t.is_stop and not t.is_punct]
    print("Content words (no stop words or punctuation):")
    for token in content_tokens:
        print(f"  {token}")
    
    # Group by POS
    from collections import defaultdict
    pos_groups = defaultdict(list)
    for token in sentence_tokens:
        if token.pos:
            pos_groups[token.pos].append(token.text)
    
    print("\nPOS distribution:")
    for pos, words in pos_groups.items():
        print(f"  {pos.value:8}: {', '.join(words)}")


if __name__ == "__main__":
    """
    Main execution block that runs the demonstration when script is executed directly.
    
    This pattern allows the file to be both imported as a module and run as a script,
    making it versatile for both educational and practical use.
    """
    demonstrate_token_class()