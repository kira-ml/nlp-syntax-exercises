"""
04_unicode_normalization.py

In this implementation, I demonstrate robust Unicode normalization and ASCII transliteration for NLP preprocessing. These techniques are essential for standardizing text from diverse sources, ensuring consistent model input, and improving downstream performance in multilingual and noisy data scenarios.

The code is modular, with clear separation between normalization, transliteration, and combined standardization logic. This structure supports integration into larger ML pipelines and highlights best practices for maintainable, production-ready text processing.
"""

import unicodedata
from unidecode import unidecode


def normalize_unicode(text: str, form: str = 'NFC') -> str:
    """
    Normalize Unicode text to a specified form.

    This function uses Python's built-in unicodedata.normalize to convert text
    to one of the standard Unicode normalization forms (NFC, NFD, NFKC, NFKD).

    Parameters
    ----------
    text : str
        Input string to be normalized.
    form : str, optional
        Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD'). Default is 'NFC'.

    Returns
    -------
    str
        Normalized Unicode string.

    Raises
    ------
    TypeError
        If input is not a string.
    ValueError
        If an invalid normalization form is provided.

    Notes
    -----
    - Unicode normalization is critical for consistent text comparison and tokenization.
    - NFC is recommended for most NLP tasks; NFKC is useful for compatibility and symbol folding.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be string")
    
    if form not in {'NFC', 'NFD', 'NFKC', 'NFKD'}:
        raise ValueError("Invalid normalization form")
    

    return unicodedata.normalize(form, text)


def transliterate(text: str) -> str:
    """
    Transliterate Unicode text to closest ASCII representation.

    This function uses the unidecode library to convert accented and non-Latin
    characters to their nearest ASCII equivalents.

    Parameters
    ----------
    text : str
        Input string to be transliterated.

    Returns
    -------
    str
        ASCII-transliterated string.

    Notes
    -----
    - Useful for downstream models that require ASCII-only input.
    - May lose information for languages with complex scripts.
    """

    return unidecode(text)



def standardize_text(
        

    text: str,
    normalize_form: str = 'NFC',
    transliterate_chars: bool = False

) -> str:
    """
    Standardize text by applying Unicode normalization and optional transliteration.

    This function demonstrates how to chain normalization and transliteration
    for robust text preprocessing in multilingual and noisy data contexts.

    Parameters
    ----------
    text : str
        Input string to be standardized.
    normalize_form : str, optional
        Unicode normalization form. Default is 'NFC'.
    transliterate_chars : bool, optional
        Whether to apply ASCII transliteration after normalization. Default is False.

    Returns
    -------
    str
        Standardized string.

    Notes
    -----
    - This function is suitable for integration into larger NLP pipelines.
    - For production, consider logging or handling edge cases for specific languages.
    """

    normalized = normalize_unicode(text, form=normalize_form)
    if transliterate_chars:
        return transliterate(normalized)
    return normalized



if __name__ == "__main__":
    # Example cases demonstrating normalization and transliteration
    text_cases = [
        ('café', 'NFD'),
        ('𝓬𝓾𝓻𝓼𝓲𝓿𝓮', 'NFKC'),
        ('ṩẖạȵȇ', 'NFC'),
        ('ﬁﬂ', 'NFKD')
    ]


    print("Unicode Normalization demo:")
    print("-" * 40)
    for text, form in text_cases:
        print(f"Original: {text}")
        print(f"Normalized ({form}): {normalize_unicode(text, form)}")
        print(f"Transliterated: {transliterate(text)}")
        print(f"Standardized: {standardize_text(text, form, True)}")
        print("-" * 40)
