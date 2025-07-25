import re




def get_stopwords_set() -> set:

    stopwords = {
        "a", "an", "and", "the", "is", "are", "in", "on", "of", "to", "it", "this", "that"
    }

    return stopwords

def remove_stopwords(tokens: list, stopword_set: set) -> list:

    filtered_tokens = [token for token in tokens if token.lower() not in stopword_set]

    return filtered_tokens


def simple_tokenizer(text: str) -> list:
    pattern = r'\b\w+\b'
    tokens = re.findall(pattern, text.lower())
    return tokens

def preprocess_text(text: str, stopword_set: set) -> list:
    tokens = simple_tokenizer(text)
    filtered = remove_stopwords(tokens, stopword_set)
    return filtered


if __name__ == "__main__":
    sample_tokens = ["This", "is", "a", "sample", "sentence", "in", "Python"]
    stops = get_stopwords_set()
    filtered = remove_stopwords(sample_tokens, stops)
    stopwords = get_stopwords_set()
    text = "This is a sample sentence written in Python"
    tokens = simple_tokenizer(text)
    result = preprocess_text(text, stopwords)
    print(filtered)
    print(tokens)
    print(result)



