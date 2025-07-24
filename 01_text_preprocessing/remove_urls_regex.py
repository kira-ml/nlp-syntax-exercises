"""
remove_urls_regex.py

In this implementation, I demonstrate a robust and reusable approach for removing URLs from text using regular expressions—a common preprocessing step in NLP pipelines, especially when working with user-generated content such as chat logs, tweets, or forum posts.

Removing URLs is essential for tasks where links are irrelevant or may introduce noise, such as sentiment analysis, topic modeling, or training language models on clean corpora. The function is designed for clarity, extensibility, and safe integration into larger data processing workflows.
"""

import re

def remove_url_from_text(text):
    """
    Remove all URLs from the input text using a regular expression.

    This function is intended for use in NLP preprocessing pipelines where URLs are considered noise or irrelevant features. It supports both HTTP(S) and 'www.' style URLs.

    Parameters
    ----------
    text : str
        The input string potentially containing URLs.

    Returns
    -------
    str
        The input string with all URLs removed. If no URLs are present, the original string is returned unchanged.

    Raises
    ------
    ValueError
        If the input is not a string. This guards against silent failures in automated pipelines.

    Notes
    -----
    - The regular expression is designed to match most common URL patterns, but may require extension for edge cases (e.g., FTP links, custom TLDs).
    - For large-scale or multilingual data, consider using more advanced URL detection libraries or normalization steps.
    """
    # Validate input type to ensure robustness in production pipelines
    if not isinstance(text, str):
        raise ValueError("Input must be string.")

    # Regex pattern matches both http(s) URLs and 'www.' prefixed domains
    url_patterns = r'https?://\S+|www\.\S+'

    # Substitute all detected URLs with an empty string
    return re.sub(url_patterns, '', text)


def main():
    """
    Example usage of the remove_url_from_text function on a chatbot log.

    This demonstrates how the function can be integrated into a preprocessing pipeline for conversational AI or social media analytics.
    """
    chatbot_log = (
        "Hi there! Check out our pricing page at https://example.com/pricing 😊 or visit www.coolstuff.ai for more info."
    )

    # Remove URLs to clean the text for downstream NLP tasks
    cleaned_log = remove_url_from_text(chatbot_log)

    print("Original Message:")
    print(chatbot_log)
    print("\nCleaned Message:")
    print(cleaned_log)


if __name__ == "__main__":
    main()