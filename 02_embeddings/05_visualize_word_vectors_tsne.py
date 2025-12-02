#!/usr/bin/env python3
"""
05_visualize_word_vectors_tsne.py

Purpose:
    Visualizes high-dimensional word vectors in 2D space using t-SNE (t-distributed 
    Stochastic Neighbor Embedding). This allows us to see semantic relationships 
    between words that are encoded in the vector space.

Concepts:
    1. Dimensionality Reduction - Transforming high-dimensional data (300D vectors)
       into 2D/3D while preserving relative distances between points
    2. Dictionary Slicing - Selecting specific words and their vectors for visualization
    3. Matplotlib Visualization - Creating scatter plots with annotations

Dependencies:
    - scikit-learn: For t-SNE implementation
    - matplotlib: For creating visualizations
    - numpy: For numerical operations

Input: Word vectors (dictionary format: {word: numpy array})
Output: 2D scatter plot with word labels
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_word_vectors_tsne(word_vectors, words_to_visualize=None, 
                                n_words=50, random_state=42, figsize=(12, 10)):
    """
    Reduces word vector dimensions from 300D to 2D using t-SNE and visualizes them.
    
    t-SNE (t-distributed Stochastic Neighbor Embedding) is particularly effective
    for visualizing high-dimensional data by modeling pairwise similarities in
    both high and low-dimensional spaces.
    
    Parameters:
    -----------
    word_vectors : dict
        Dictionary mapping words to their vector representations (numpy arrays)
    words_to_visualize : list, optional
        Specific list of words to visualize. If None, selects random sample
    n_words : int, default=50
        Number of words to visualize if words_to_visualize is not provided
    random_state : int, default=42
        Random seed for reproducibility of t-SNE and word selection
    figsize : tuple, default=(12, 10)
        Figure dimensions for matplotlib plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated visualization figure
    numpy.ndarray
        2D coordinates of word vectors after t-SNE transformation
    """
    
    # ============================================================================
    # STEP 1: Select words for visualization
    # ============================================================================
    
    # If specific words are provided, use them; otherwise, select random sample
    if words_to_visualize is not None:
        # Filter to only include words that exist in our vocabulary
        selected_words = [word for word in words_to_visualize 
                         if word in word_vectors]
        if len(selected_words) < len(words_to_visualize):
            print(f"Warning: {len(words_to_visualize) - len(selected_words)} "
                  f"words not found in vocabulary")
    else:
        # Select random subset of words from the vocabulary
        # This prevents overcrowding in visualization
        all_words = list(word_vectors.keys())
        if n_words > len(all_words):
            n_words = len(all_words)
            print(f"Warning: n_words reduced to {len(all_words)} (vocabulary size)")
        
        np.random.seed(random_state)  # For reproducible random selection
        selected_words = np.random.choice(all_words, size=n_words, replace=False)
    
    # ============================================================================
    # STEP 2: Extract vectors for selected words
    # ============================================================================
    
    # Create arrays for words and their corresponding vectors
    # Each vector is a 300-dimensional representation (for Word2Vec/GloVe)
    vectors = []
    valid_words = []
    
    for word in selected_words:
        vectors.append(word_vectors[word])  # Get vector for this word
        valid_words.append(word)            # Keep track of the word label
    
    # Convert list of vectors to 2D numpy array
    # Shape: (n_words, 300) - each row is a word vector
    vectors_array = np.array(vectors)
    
    print(f"Visualizing {len(valid_words)} words")
    print(f"Vector shape: {vectors_array.shape} (words × dimensions)")
    
    # ============================================================================
    # STEP 3: Dimensionality reduction with t-SNE
    # ============================================================================
    
    # Initialize t-SNE model
    # Parameters:
    # - n_components: Output dimension (2D for visualization)
    # - random_state: For reproducible results
    # - perplexity: Roughly balances local/global structure (typical: 5-50)
    # - n_iter: Number of optimization iterations
    # - init: Initialization method ('pca' often works better than 'random')
    tsne = TSNE(n_components=2, random_state=random_state, 
                perplexity=min(30, len(vectors_array)-1), 
                n_iter=1000, init='pca')
    
    print("Running t-SNE dimensionality reduction...")
    
    # Transform 300D vectors to 2D space
    # This is the core operation: reducing dimensions while preserving
    # the relative distances between word vectors as much as possible
    vectors_2d = tsne.fit_transform(vectors_array)
    
    print("t-SNE transformation complete!")
    print(f"Reduced shape: {vectors_2d.shape} (words × 2D coordinates)")
    
    # ============================================================================
    # STEP 4: Create visualization
    # ============================================================================
    
    # Set up the figure with specified size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract x and y coordinates from t-SNE output
    x_coords = vectors_2d[:, 0]  # First dimension
    y_coords = vectors_2d[:, 1]  # Second dimension
    
    # Create scatter plot
    # Each point represents a word in the 2D reduced space
    scatter = ax.scatter(x_coords, y_coords, alpha=0.6, c='steelblue', edgecolors='black')
    
    # Add word labels to each point
    # This helps interpret the semantic clusters
    for i, word in enumerate(valid_words):
        # Add text annotation slightly offset from the point
        ax.annotate(word, 
                   xy=(x_coords[i], y_coords[i]),  # Point coordinates
                   xytext=(5, 2),                  # Text offset from point
                   textcoords='offset points',     # Offset coordinate system
                   fontsize=9,
                   alpha=0.8)
    
    # Set plot title and labels
    ax.set_title(f't-SNE Visualization of Word Vectors ({len(valid_words)} words)', 
                fontsize=14, pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure equal aspect ratio for proper distance representation
    ax.set_aspect('equal', adjustable='datalim')
    
    # Add informative text box
    info_text = f"""t-SNE Parameters:
    Perplexity: {tsne.perplexity}
    Iterations: {tsne.n_iter}
    Initialization: {tsne.init}
    Original dimensions: {vectors_array.shape[1]}D
    Reduced to: {tsne.n_components}D"""
    
    # Place text box in upper right corner
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig, vectors_2d

# ================================================================================
# EXAMPLE USAGE
# ================================================================================

def example_usage():
    """
    Demonstrates how to use the visualize_word_vectors_tsne function.
    
    This example shows:
    1. Creating synthetic word vectors for demonstration
    2. Visualizing semantic categories (animals, countries, emotions)
    3. Saving the visualization to a file
    """
    
    # Create synthetic word vectors for demonstration
    # In practice, you would load real word vectors (Word2Vec, GloVe, etc.)
    np.random.seed(42)
    
    # Define words from different semantic categories
    animals = ['cat', 'dog', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey']
    countries = ['france', 'germany', 'italy', 'spain', 'japan', 'china', 'india', 'brazil']
    emotions = ['happy', 'sad', 'angry', 'excited', 'calm', 'anxious', 'joyful', 'fearful']
    
    # Combine all words
    demo_words = animals + countries + emotions
    
    # Create synthetic vectors (300-dimensional like typical word embeddings)
    # For demonstration, we'll make words in the same category somewhat similar
    word_vectors = {}
    
    # Generate vectors with some structure
    for i, word in enumerate(demo_words):
        # Base vector with random values
        vector = np.random.randn(300) * 0.5
        
        # Add category-specific signal
        if word in animals:
            vector += np.random.randn(300) * 0.3 + 0.5  # Animal cluster
        elif word in countries:
            vector += np.random.randn(300) * 0.3 - 0.5  # Country cluster
        elif word in emotions:
            vector += np.random.randn(300) * 0.3 + 1.0  # Emotion cluster
        
        word_vectors[word] = vector
    
    print("=" * 70)
    print("DEMONSTRATION: t-SNE Visualization of Word Vectors")
    print("=" * 70)
    
    # Example 1: Visualize specific semantic categories
    print("\nExample 1: Visualizing animals, countries, and emotions")
    fig1, coords1 = visualize_word_vectors_tsne(
        word_vectors=word_vectors,
        words_to_visualize=demo_words,  # Specify which words to visualize
        random_state=42
    )
    
    # Example 2: Visualize random sample of words
    print("\nExample 2: Visualizing random sample of words")
    fig2, coords2 = visualize_word_vectors_tsne(
        word_vectors=word_vectors,
        words_to_visualize=None,  # Will select random words
        n_words=20,               # Number of words to visualize
        random_state=123
    )
    
    # Save the figures (uncomment to use)
    # fig1.savefig('tsne_word_vectors.png', dpi=300, bbox_inches='tight')
    # fig2.savefig('tsne_random_sample.png', dpi=300, bbox_inches='tight')
    
    return fig1, fig2

# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    """
    Main execution block.
    
    When this script is run directly:
    1. If real word vectors are available, they can be loaded and visualized
    2. Otherwise, the example usage demonstrates the functionality
    """
    
    # In practice, you would load real word vectors here:
    # Example:
    # from gensim.models import KeyedVectors
    # word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # visualize_word_vectors_tsne(word_vectors, n_words=100)
    
    print(__doc__)  # Print module documentation
    
    # Run demonstration with synthetic data
    example_usage()
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("Interpretation guide:")
    print("- Words close together have similar meanings in the vector space")
    print("- Clusters represent semantic categories")
    print("- Distances in 2D approximate distances in original 300D space")
    print("=" * 70)

# ================================================================================
# NOTES AND BEST PRACTICES
# ================================================================================
"""
Notes:
1. t-SNE is stochastic - results vary with random_state
2. Perplexity should be less than number of samples
3. t-SNE preserves local structure better than global structure
4. For better results with real data:
   - Use more words (100-1000)
   - Experiment with perplexity (5-50)
   - Increase n_iter for convergence
   - Consider using PCA initialization

Limitations:
1. t-SNE is computationally expensive for large datasets
2. 2D distances don't perfectly represent high-dimensional distances
3. Different runs with different seeds yield different layouts
4. Not suitable for words not in the original vocabulary

Alternative approaches:
1. PCA: Faster but may not preserve non-linear relationships
2. UMAP: Often preserves both local and global structure better
3. MDS: Preserves distances but computationally expensive
"""