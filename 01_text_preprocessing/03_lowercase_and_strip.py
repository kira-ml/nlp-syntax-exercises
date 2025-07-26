"""
03_lowercase_and_strip.py

In this implementation, I demonstrate a foundational text normalization technique for NLP pipelines: lowercasing and whitespace trimming. These operations are essential for reducing feature sparsity and ensuring consistent input to downstream models.

The code is modular, with clear separation between data generation, cleaning logic, and batch processing. This structure supports easy integration into larger ML workflows and highlights best practices for maintainable preprocessing code.
"""

def get_sample_web_text():
    """
    Generate a sample dataset simulating raw web text.

    Returns
    -------
    list of str
        List of unnormalized text strings, including mixed casing and extra whitespace.

    Notes
    -----
    - In production, this function would be replaced by actual data loading routines.
    """
    return [
        "   This is SOME Text!   ",
        "Here's another LINE\t",
        "   Mixed    Case   &   Spaces   ",
        "clean-Me    QUICK!   ",
        "123 NUMBERS and CAPS   "
    ]

def clean_text(text):
    """
    Normalize a single text string by trimming whitespace and converting to lowercase.

    Parameters
    ----------
    text : str
        The input string to be cleaned.

    Returns
    -------
    str
        The cleaned string, with leading/trailing whitespace removed and all characters lowercased.

    Notes
    -----
    - This function is suitable for basic normalization in most English NLP tasks.
    - For more advanced normalization, consider Unicode normalization or custom rules.
    """
    # Use built-in strip() and lower() for efficient normalization
    return text.strip().lower()

def clean_dataset(dataset):
    """
    Apply text normalization to an entire dataset.

    Parameters
    ----------
    dataset : list of str
        List of raw text strings to be cleaned.

    Returns
    -------
    list of str
        List of cleaned text strings.

    Notes
    -----
    - Demonstrates functional programming with list comprehensions for batch processing.
    - In production, consider using pandas.Series.str methods for large datasets.
    """
    return [clean_text(text) for text in dataset]

def main():
    """
    Example usage of the normalization pipeline.

    This function demonstrates how to apply lowercasing and whitespace trimming to a batch of raw text, and how to display results for inspection.
    """
    raw_data = get_sample_web_text()
    cleaned_data = clean_dataset(raw_data)

    print("Raw data:")
    for line in raw_data:
        print(f"- {repr(line)}")

    print("Cleaned data:")
    for line in cleaned_data:
        print(f"- {repr(line)}")

if __name__ == "__main__":
    main()