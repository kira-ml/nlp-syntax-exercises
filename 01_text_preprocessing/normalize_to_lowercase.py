
"""
normalize_to_lowercase.py

In this implementation, I demonstrate a robust, reusable function for normalizing text to lowercase—a foundational preprocessing step in most NLP and machine learning pipelines. This script is designed for clarity and extensibility, making it suitable for both educational purposes and integration into larger production systems.

Lowercasing is a common normalization technique that helps reduce vocabulary size and mitigate case-related variance in downstream models. The function includes input validation to ensure reliability when used in automated data pipelines.

Example usage is provided in the main block, illustrating how this function can be applied to informal, user-generated content such as tweets.
"""

def normalize_to_lowercase(text):
    """
    Convert input text to lowercase in a safe and robust manner.

    This function is intended for use as a preprocessing step in NLP pipelines, where consistent casing is critical for reducing feature sparsity and improving model generalization.

    Parameters
    ----------
    text : str
        The input string to be normalized.

    Returns
    -------
    str
        The normalized, lowercased string. Returns an empty string if input is empty or whitespace.

    Raises
    ------
    ValueError
        If the input is not a string. This guards against silent failures in data pipelines.

    Notes
    -----
    - This function does not perform Unicode normalization or remove diacritics; such steps may be added for specific use cases.
    - For batch processing, consider vectorizing this function or applying it with pandas.Series.str.lower.
    """
    # Validate input type to prevent subtle bugs in larger pipelines
    if not isinstance(text, str):
        raise ValueError("Input must be string.")

    # Return empty string for empty or whitespace-only input, ensuring predictable output
    if not text.strip():
        return ""

    # Use Python's built-in lower() for Unicode-aware lowercasing
    return text.lower()


def main():
    """
    Example usage of the normalize_to_lowercase function on a sample tweet.

    This demonstrates how the function can be integrated into a preprocessing pipeline for social media data.
    """
    tweet = "OMG! This NEW IPhone is CRAZY good 😍📱 #AppleEvent"

    # Normalize the tweet to lowercase for consistent downstream processing
    normalized_tweet = normalize_to_lowercase(tweet)

    print("Original tweet:", tweet)
    print("Normalized tweet:", normalized_tweet)


if __name__ == "__main__":
    main()