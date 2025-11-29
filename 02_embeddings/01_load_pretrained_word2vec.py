# 01_load_pretrained_word2vec.py
"""
Load pre-trained Word2Vec vectors in binary or text format using gensim.
Retrieve word vectors for inspection or similarity queries.

Concepts: File I/O, dictionary-style access, KeyedVectors.
"""

import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import os
import argparse

def load_word2vec_binary(file_path):
    """
    Load Word2Vec vectors from binary format.
    
    Args:
        file_path (str): Path to the binary Word2Vec file
    
    Returns:
        KeyedVectors: Loaded word vectors
    """
    print(f"Loading Word2Vec vectors from binary file: {file_path}")
    try:
        # Load binary format
        model = KeyedVectors.load_word2vec_format(file_path, binary=True)
        print(f"Successfully loaded binary model with {len(model)} words")
        return model
    except Exception as e:
        print(f"Error loading binary file: {e}")
        return None

def load_word2vec_text(file_path):
    """
    Load Word2Vec vectors from text format.
    
    Args:
        file_path (str): Path to the text Word2Vec file
    
    Returns:
        KeyedVectors: Loaded word vectors
    """
    print(f"Loading Word2Vec vectors from text file: {file_path}")
    try:
        # Load text format
        model = KeyedVectors.load_word2vec_format(file_path, binary=False)
        print(f"Successfully loaded text model with {len(model)} words")
        return model
    except Exception as e:
        print(f"Error loading text file: {e}")
        return None

def download_pretrained_model(model_name='word2vec-google-news-300'):
    """
    Download a pre-trained Word2Vec model using gensim's downloader.
    
    Args:
        model_name (str): Name of the model to download
    
    Returns:
        KeyedVectors: Downloaded word vectors
    """
    print(f"Downloading pre-trained model: {model_name}")
    try:
        model = api.load(model_name)
        print(f"Successfully downloaded {model_name} with {len(model)} words")
        return model
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def inspect_word_vectors(model, words):
    """
    Inspect word vectors for given words.
    
    Args:
        model (KeyedVectors): Loaded word vectors
        words (list): List of words to inspect
    """
    print("\n" + "="*50)
    print("WORD VECTOR INSPECTION")
    print("="*50)
    
    for word in words:
        if word in model:
            vector = model[word]
            print(f"\nWord: '{word}'")
            print(f"Vector shape: {vector.shape}")
            print(f"First 10 dimensions: {vector[:10]}")
            print(f"Vector norm: {np.linalg.norm(vector):.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary")

def perform_similarity_queries(model, words, topn=5):
    """
    Perform similarity queries for given words.
    
    Args:
        model (KeyedVectors): Loaded word vectors
        words (list): List of words to query
        topn (int): Number of similar words to return
    """
    print("\n" + "="*50)
    print("SIMILARITY QUERIES")
    print("="*50)
    
    for word in words:
        if word in model:
            print(f"\nMost similar to '{word}':")
            similar_words = model.most_similar(word, topn=topn)
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary, cannot find similar words")

def word_analogy(model, positive, negative, topn=5):
    """
    Perform word analogy tasks (e.g., king - man + woman = queen).
    
    Args:
        model (KeyedVectors): Loaded word vectors
        positive (list): Positive words to add
        negative (list): Negative words to subtract
        topn (int): Number of results to return
    """
    print(f"\nAnalogy: {' + '.join(positive)} - {' - '.join(negative)}")
    try:
        results = model.most_similar(positive=positive, negative=negative, topn=topn)
        for word, similarity in results:
            print(f"  {word}: {similarity:.4f}")
    except Exception as e:
        print(f"Error performing analogy: {e}")

def get_vocabulary_info(model):
    """
    Get basic information about the vocabulary.
    
    Args:
        model (KeyedVectors): Loaded word vectors
    """
    print("\n" + "="*50)
    print("VOCABULARY INFORMATION")
    print("="*50)
    print(f"Vocabulary size: {len(model):,}")
    print(f"Vector dimensions: {model.vector_size}")
    
    # Show some sample words
    sample_words = list(model.key_to_index.keys())[:10]
    print(f"Sample words: {sample_words}")

def main():
    parser = argparse.ArgumentParser(description='Load pre-trained Word2Vec vectors')
    parser.add_argument('--file', type=str, help='Path to Word2Vec file')
    parser.add_argument('--format', choices=['binary', 'text'], help='File format')
    parser.add_argument('--download', action='store_true', help='Download pre-trained model')
    parser.add_argument('--model_name', type=str, default='word2vec-google-news-300', 
                       help='Model name for download')
    
    args = parser.parse_args()
    
    model = None
    
    # Load model based on arguments
    if args.download:
        model = download_pretrained_model(args.model_name)
    elif args.file and args.format:
        if args.format == 'binary':
            model = load_word2vec_binary(args.file)
        elif args.format == 'text':
            model = load_word2vec_text(args.file)
    else:
        # Default: try to download a model
        print("No file specified, downloading pre-trained model...")
        model = download_pretrained_model()
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    # Demonstrate various operations
    test_words = ['king', 'queen', 'man', 'woman', 'computer', 'python', 'apple']
    
    # Basic information
    get_vocabulary_info(model)
    
    # Inspect vectors
    inspect_word_vectors(model, test_words[:3])
    
    # Similarity queries
    perform_similarity_queries(model, test_words[:3])
    
    # Word analogies
    print("\n" + "="*50)
    print("WORD ANALOGIES")
    print("="*50)
    word_analogy(model, ['king', 'woman'], ['man'])  # king - man + woman = queen
    word_analogy(model, ['paris', 'germany'], ['france'])  # paris - france + germany = berlin
    word_analogy(model, ['walked', 'swim'], ['walk'])  # walked - walk + swim = swam
    
    # Similarity between words
    print("\n" + "="*50)
    print("WORD SIMILARITIES")
    print("="*50)
    word_pairs = [('king', 'queen'), ('man', 'woman'), ('computer', 'laptop')]
    for word1, word2 in word_pairs:
        if word1 in model and word2 in model:
            similarity = model.similarity(word1, word2)
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        else:
            print(f"Cannot compute similarity: one of '{word1}', '{word2}' not in vocabulary")

if __name__ == "__main__":
    main()