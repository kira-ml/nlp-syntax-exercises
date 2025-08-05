"""
Language Detection Utility Using langid

This module provides robust language detection capabilities for short text snippets
using the langid library. It implements confidence-based filtering to ensure
reliable predictions and handles common edge cases that can lead to false positives.

The implementation demonstrates production-grade practices including:
- Proper error handling for malformed or edge-case inputs
- Confidence thresholding using langid's log-probability scoring
- Batch processing for efficient handling of multiple texts
- Type hints for improved code clarity and IDE support

Author: kira-ml (https://github.com/kira-ml)
Example:
    >>> from langid_detector import detect_languages_batch
    >>> texts = ["Hello world", "Bonjour le monde"]
    >>> results = detect_languages_batch(texts)
    >>> print(results[0]['language'])  # 'en'
"""

import langid
from typing import List, Dict, Any


def detect_language_with_threshold(text: str, threshold: float = -150.0) -> Dict[str, Any]:
    """
    Detect language with confidence threshold using langid library.
    
    This function implements robust language detection with multiple layers of
    validation to prevent false positives on edge cases. The langid library
    returns log-probabilities (negative values) where higher values (closer to 0)
    indicate higher confidence.
    
    Design rationale:
    - Uses -999.0 as sentinel value for invalid inputs to distinguish from
      legitimate low-confidence predictions
    - Implements early returns for common edge cases to avoid unnecessary
      computation
    - Wraps langid.classify in try/except to handle unexpected library errors
    
    Args:
        text: Input text string to analyze for language detection
        threshold: Log-probability threshold for reliable detection.
                  Range: -300 (very uncertain) to 0 (very certain).
                  Default -150.0 allows most real language detections while
                  filtering out gibberish.
    
    Returns:
        Dictionary containing:
            - 'language': ISO 639-1 language code or 'unknown'
            - 'confidence': Log-probability score or -999.0 for invalid inputs
            - 'is_reliable': Boolean indicating if confidence meets threshold
    
    Example:
        >>> result = detect_language_with_threshold("Hello world")
        >>> print(result['language'], result['is_reliable'])
        'en' True
    """
    # Validate input type - prevent TypeErrors from langid.classify
    if not isinstance(text, str):
        return {
            'language': 'unknown',
            'confidence': -999.0,
            'is_reliable': False
        }
    
    # Handle empty or extremely short text that cannot provide meaningful signals
    stripped_text = text.strip()
    if not stripped_text or len(stripped_text) <= 3:
        return {
            'language': 'unknown',
            'confidence': -999.0,
            'is_reliable': False
        }
    
    # Filter common trivial patterns that mimic language structure but aren't meaningful
    trivial_patterns = ['....', '...', '??', '!!', '???', '!!!']
    if stripped_text in trivial_patterns:
        return {
            'language': 'unknown',
            'confidence': -999.0,
            'is_reliable': False
        }
    
    try:
        # Perform core language detection using langid's trained model
        lang_code, confidence = langid.classify(stripped_text)
        
        # Apply confidence threshold to ensure prediction reliability
        # This prevents false positives on random characters or gibberish
        is_reliable = confidence >= threshold
        
        return {
            'language': lang_code if is_reliable else 'unknown',
            'confidence': confidence,
            'is_reliable': is_reliable
        }
        
    except Exception:
        # Defensive programming: catch any unexpected errors from langid
        # This ensures the function never crashes the calling pipeline
        return {
            'language': 'unknown',
            'confidence': -999.0,
            'is_reliable': False
        }


def detect_languages_batch(text_list: List[str], threshold: float = -150.0) -> List[Dict[str, Any]]:
    """
    Process multiple texts for language detection using list comprehension.
    
    This batch processing approach is more efficient than iterative loops for
    moderate-sized datasets and maintains consistent error handling across all
    inputs. In production systems, this could be extended with parallel
    processing for large-scale applications.
    
    Args:
        text_list: List of text strings to analyze
        threshold: Confidence threshold passed to individual detection calls
    
    Returns:
        List of detection results maintaining order with input texts
    
    Example:
        >>> texts = ["Hello", "Bonjour"]
        >>> results = detect_languages_batch(texts)
        >>> len(results) == len(texts)
        True
    """
    # List comprehension provides optimal performance for this operation
    # while maintaining readability and consistent error handling
    return [detect_language_with_threshold(text, threshold) for text in text_list]


def print_detection_results(text_list: List[str], results: List[Dict[str, Any]]) -> None:
    """
    Display formatted language detection results in tabular format.
    
    Provides human-readable output suitable for debugging, testing, or
    command-line interfaces. In production systems, this would typically
    be replaced with structured logging or JSON output for machine consumption.
    
    Args:
        text_list: Original input texts (for display context)
        results: Detection results from detect_languages_batch()
    """
    # Header with consistent column alignment for readability
    print(f"{'Text':<30} {'Language':<10} {'Confidence':<12} {'Reliable':<10}")
    print("-" * 65)
    
    for text, result in zip(text_list, results):
        # Truncate long texts to maintain table formatting
        display_text = (text[:27] + "...") if len(text) > 30 else text
        lang = result['language']
        
        # Format confidence score or show N/A for invalid inputs
        conf = f"{result['confidence']:.2f}" if result['confidence'] != -999.0 else "N/A"
        reliable = "Yes" if result['is_reliable'] else "No"
        
        print(f"{display_text:<30} {lang:<10} {conf:<12} {reliable:<10}")


if __name__ == "__main__":
    """
    Demonstration and testing entry point for the language detection utility.
    
    This section showcases the module's capabilities with a diverse set of
    test cases including multiple languages and common edge cases. In a
    production environment, this would typically be replaced with unit tests
    or moved to a separate test module.
    """
    # Comprehensive test dataset covering various languages and edge cases
    # Selected to demonstrate robustness across different script types and lengths
    sample_texts = [
        "Kumain kami ng kanin at isda ngayong umaga.",  # Filipino (Tagalog)
        "We are going to the market tomorrow.",         # English
        "¿Dónde está la biblioteca?",                   # Spanish
        "C'est une belle journée.",                     # French
        "Guten Tag! Wie geht es Ihnen?",                # German
        "....",                                         # Edge case: repeated punctuation
        "???",                                          # Edge case: question marks
        "",                                             # Edge case: empty string
        "a",                                            # Edge case: single character
        "Magandang umaga po!",                          # Filipino
        "Je vais au marché aujourd'hui.",               # French
        "Das ist ein schöner Tag.",                     # German
        "Saya makan nasi dan ikan pagi ini."            # Indonesian
    ]

    # Execute batch language detection with moderate confidence threshold
    # The -150.0 threshold balances sensitivity with reliability for short texts
    detection_results = detect_languages_batch(sample_texts, threshold=-150.0)
    
    # Display results in user-friendly tabular format
    print_detection_results(sample_texts, detection_results)
    
    # Provide detailed breakdown for deeper inspection of results
    print("\n" + "="*50)
    print("Detailed Results:")
    print("="*50)
    
    for i, (text, result) in enumerate(zip(sample_texts, detection_results)):
        print(f"{i+1}. Text: '{text}'")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reliable: {result['is_reliable']}")
        print()