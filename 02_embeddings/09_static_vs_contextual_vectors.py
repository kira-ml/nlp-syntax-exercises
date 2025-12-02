"""
09_static_vs_contextual_vectors.py
→ Compare cosine similarity between static (Word2Vec) and contextual (BERT) vectors 
  for polysemous words like "bank".

Educational Objectives:
1. Understand the fundamental difference between static and contextual embeddings
2. Learn how to calculate cosine similarity for word vectors
3. Observe how contextual embeddings capture semantic shifts in polysemy
4. Compare the limitations and strengths of both approaches

Concepts Covered:
- Static embeddings (Word2Vec/FastText): One vector per word type
- Contextual embeddings (BERT): Different vectors per word token
- Cosine similarity as a measure of semantic relatedness
- Polysemy: Words with multiple meanings (e.g., "bank", "bat", "rock")
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: SIMULATED STATIC EMBEDDINGS (Word2Vec-like)
# ============================================================================

class StaticWordEmbeddings:
    """
    Simulates static word embeddings like Word2Vec or GloVe.
    
    Educational Note:
    Static embeddings assign ONE fixed vector to each word regardless of context.
    This is efficient but cannot distinguish between different meanings of polysemous words.
    """
    
    def __init__(self):
        # Create a simple vocabulary with simulated 5-dimensional embeddings
        # In real Word2Vec, dimensions are typically 100-300
        self.vocab = {
            # Financial meanings
            'bank': np.array([0.9, 0.1, 0.3, 0.1, 0.8]),  # Financial institution
            'money': np.array([0.8, 0.2, 0.4, 0.1, 0.7]),
            'loan': np.array([0.7, 0.1, 0.3, 0.2, 0.9]),
            
            # River meanings
            'river': np.array([0.1, 0.9, 0.2, 0.8, 0.1]),
            'water': np.array([0.2, 0.8, 0.1, 0.9, 0.2]),
            'shore': np.array([0.1, 0.7, 0.1, 0.8, 0.3]),
            
            # Neutral/ambiguous
            'deposit': np.array([0.5, 0.5, 0.3, 0.4, 0.5]),  # Both financial and geological
            
            # Same vector for "bank" regardless of context!
            # This is the key limitation of static embeddings
        }
    
    def get_vector(self, word):
        """Retrieve static vector for a word."""
        return self.vocab.get(word.lower(), np.zeros(5))

# ============================================================================
# PART 2: SIMULATED CONTEXTUAL EMBEDDINGS (BERT-like)
# ============================================================================

class ContextualEmbeddings:
    """
    Simulates contextual embeddings like BERT or ELMo.
    
    Educational Note:
    Contextual embeddings generate DIFFERENT vectors for the same word 
    based on surrounding context. This allows them to capture polysemy.
    """
    
    def __init__(self):
        # Base vectors that get modified by context
        self.base_vectors = {
            'bank': np.array([0.5, 0.5, 0.3, 0.4, 0.5]),  # Ambiguous base
            'money': np.array([0.8, 0.2, 0.4, 0.1, 0.7]),
            'loan': np.array([0.7, 0.1, 0.3, 0.2, 0.9]),
            'river': np.array([0.1, 0.9, 0.2, 0.8, 0.1]),
            'water': np.array([0.2, 0.8, 0.1, 0.9, 0.2]),
            'deposit': np.array([0.5, 0.5, 0.3, 0.4, 0.5]),
        }
    
    def get_contextual_vector(self, word, context_words):
        """
        Generate contextual vector based on surrounding words.
        
        Educational Note:
        In real BERT, this is done through transformer architecture that 
        attends to all words in the sentence simultaneously.
        """
        base = self.base_vectors.get(word.lower(), np.zeros(5))
        
        # Simulate context influence (simplified version of attention)
        context_vector = np.zeros(5)
        for ctx_word in context_words:
            if ctx_word in self.base_vectors:
                # Context words influence the target word's representation
                context_vector += 0.2 * self.base_vectors[ctx_word]
        
        # Combine base meaning with context
        contextual = base + 0.3 * context_vector
        return contextual / np.linalg.norm(contextual)  # Normalize

# ============================================================================
# PART 3: SIMILARITY CALCULATION AND VISUALIZATION
# ============================================================================

def calculate_cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Mathematical Formula:
    cos_sim(A, B) = (A · B) / (||A|| * ||B||)
    
    Educational Insight:
    Cosine similarity ranges from -1 (opposite) to 1 (identical).
    For normalized word vectors, values typically range from 0 to 1.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def print_similarity_matrix(words, vectors, title):
    """Display similarity matrix for educational comparison."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"{'Word':<10} {'Vector (first 3 dims)':<25} Similarities")
    print("-"*60)
    
    for i, (word, vec) in enumerate(zip(words, vectors)):
        # Show truncated vector for readability
        vec_str = f"[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}, ...]"
        
        # Calculate similarities with all other words
        sims = []
        for j, other_vec in enumerate(vectors):
            if i != j:
                sim = calculate_cosine_similarity(vec, other_vec)
                sims.append(f"{words[j]}: {sim:.3f}")
        
        print(f"{word:<10} {vec_str:<25} {', '.join(sims[:3])}")

def demonstrate_polysemy():
    """Main educational demonstration."""
    
    # Initialize embedding systems
    print("Initializing embedding systems...")
    static_model = StaticWordEmbeddings()
    contextual_model = ContextualEmbeddings()
    
    # ========================================================================
    # DEMONSTRATION 1: Static Embeddings (Word2Vec)
    # ========================================================================
    print("\n" + "="*60)
    print("DEMONSTRATION 1: STATIC EMBEDDINGS (Word2Vec-like)")
    print("="*60)
    
    print("\n❓ Key Question: Does 'bank' have the same vector in all contexts?")
    
    # Get static vectors
    bank_financial = static_model.get_vector('bank')
    bank_river = static_model.get_vector('bank')  # Same vector!
    
    # Compare similarities
    money_vec = static_model.get_vector('money')
    river_vec = static_model.get_vector('river')
    
    sim_bank_money = calculate_cosine_similarity(bank_financial, money_vec)
    sim_bank_river = calculate_cosine_similarity(bank_river, river_vec)
    
    print(f"\n✅ In static embeddings, 'bank' ALWAYS has the same vector:")
    print(f"   Vector for 'bank' (any context): {bank_financial.round(2)}")
    print(f"\n📊 Similarities:")
    print(f"   bank–money: {sim_bank_money:.3f} (should be high for financial context)")
    print(f"   bank–river: {sim_bank_river:.3f} (should be high for river context)")
    
    # ========================================================================
    # DEMONSTRATION 2: Contextual Embeddings (BERT)
    # ========================================================================
    print("\n\n" + "="*60)
    print("DEMONSTRATION 2: CONTEXTUAL EMBEDDINGS (BERT-like)")
    print("="*60)
    
    print("\n❓ Key Question: Can we get different vectors for 'bank' in different contexts?")
    
    # Create different contexts for "bank"
    financial_context = ["bank", "withdraw", "money", "account", "loan"]
    river_context = ["bank", "river", "water", "fish", "shore"]
    
    # Get contextual vectors
    bank_financial_ctx = contextual_model.get_contextual_vector('bank', financial_context)
    bank_river_ctx = contextual_model.get_contextual_vector('bank', river_context)
    
    # Get context words for comparison
    money_vec_ctx = contextual_model.get_contextual_vector('money', financial_context)
    river_vec_ctx = contextual_model.get_contextual_vector('river', river_context)
    
    print(f"\n✅ In contextual embeddings, 'bank' gets DIFFERENT vectors:")
    print(f"   'bank' in financial context: {bank_financial_ctx.round(2)}")
    print(f"   'bank' in river context:     {bank_river_ctx.round(2)}")
    
    # Calculate similarities
    sim_financial = calculate_cosine_similarity(bank_financial_ctx, money_vec_ctx)
    sim_river = calculate_cosine_similarity(bank_river_ctx, river_vec_ctx)
    sim_cross_context = calculate_cosine_similarity(bank_financial_ctx, bank_river_ctx)
    
    print(f"\n📊 Similarities with contextual embeddings:")
    print(f"   bank(financial)–money: {sim_financial:.3f}")
    print(f"   bank(river)–river:     {sim_river:.3f}")
    print(f"   bank(financial)–bank(river): {sim_cross_context:.3f}")
    
    # ========================================================================
    # DEMONSTRATION 3: Quantitative Comparison
    # ========================================================================
    print("\n\n" + "="*60)
    print("DEMONSTRATION 3: QUANTITATIVE COMPARISON")
    print("="*60)
    
    # Prepare comparison table
    comparisons = [
        ("bank–money", "bank", "money", financial_context, financial_context),
        ("bank–river", "bank", "river", river_context, river_context),
        ("bank(fin)–bank(riv)", "bank", "bank", financial_context, river_context),
    ]
    
    print("\n📈 Similarity Comparison Table:")
    print(f"{'Pair':<20} {'Static':<10} {'Contextual':<10} {'Difference':<10}")
    print("-"*50)
    
    for name, word1, word2, ctx1, ctx2 in comparisons:
        # Static similarity
        vec1_static = static_model.get_vector(word1)
        vec2_static = static_model.get_vector(word2)
        static_sim = calculate_cosine_similarity(vec1_static, vec2_static)
        
        # Contextual similarity
        vec1_ctx = contextual_model.get_contextual_vector(word1, ctx1)
        vec2_ctx = contextual_model.get_contextual_vector(word2, ctx2)
        ctx_sim = calculate_cosine_similarity(vec1_ctx, vec2_ctx)
        
        diff = ctx_sim - static_sim
        
        print(f"{name:<20} {static_sim:<10.3f} {ctx_sim:<10.3f} {diff:>+8.3f}")
    
    # ========================================================================
    # EDUCATIONAL INSIGHTS
    # ========================================================================
    print("\n\n" + "="*60)
    print("KEY EDUCATIONAL INSIGHTS")
    print("="*60)
    
    insights = [
        "1. STATIC EMBEDDINGS (Word2Vec/GloVe):",
        "   • One vector per word type",
        "   • Efficient for storage and computation",
        "   • Cannot handle polysemy well",
        "   • 'Bank' is equally similar to 'money' and 'river'",
        "",
        "2. CONTEXTUAL EMBEDDINGS (BERT/ELMo):",
        "   • Different vectors per word token",
        "   • Captures semantic shifts in context",
        "   • Computationally more expensive",
        "   • 'Bank' in financial context ≠ 'bank' in river context",
        "",
        "3. PRACTICAL IMPLICATIONS:",
        "   • Use static embeddings for:",
        "     - Large-scale similarity tasks",
        "     - Computational efficiency needed",
        "     - When polysemy is not critical",
        "   • Use contextual embeddings for:",
        "     - Understanding nuanced meaning",
        "     - Downstream NLP tasks",
        "     - When context matters",
    ]
    
    for line in insights:
        print(line)

# ============================================================================
# PART 4: EXERCISES FOR STUDENTS
# ============================================================================

def student_exercises():
    """Suggested exercises for deeper understanding."""
    
    print("\n" + "="*60)
    print("STUDENT EXERCISES")
    print("="*60)
    
    exercises = [
        "1. EXTEND THE VOCABULARY:",
        "   Add more polysemous words like 'bat', 'rock', 'light', 'crane'",
        "",
        "2. EXPERIMENT WITH DIMENSIONALITY:",
        "   Modify the vector dimensions from 5 to higher values",
        "   Observe how it affects similarity scores",
        "",
        "3. CREATE NEW CONTEXTS:",
        "   Test 'deposit' in both financial and geological contexts",
        "   Compare similarity with 'bank' and 'sediment'",
        "",
        "4. IMPLEMENT ACTUAL MODELS:",
        "   Replace simulations with actual Word2Vec and BERT models",
        "   Use libraries: gensim for Word2Vec, transformers for BERT",
        "",
        "5. QUANTITATIVE ANALYSIS:",
        "   Calculate the average similarity difference across multiple words",
        "   Create visualization of vector spaces using PCA",
    ]
    
    for line in exercises:
        print(line)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    # Run the main demonstration
    demonstrate_polysemy()
    
    # Show student exercises
    student_exercises()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
This demonstration illustrates the fundamental difference between static and 
contextual word embeddings. Static embeddings are computationally efficient 
but cannot handle polysemy, while contextual embeddings capture nuanced 
meanings but require more resources.

The choice between them depends on your specific NLP task, available 
computational resources, and the importance of contextual understanding 
in your application.
    """)