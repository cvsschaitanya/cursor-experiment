import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from transformer.src.attention import create_qkv, compute_attention_scores, apply_softmax, compute_attention_output

def test_attention():
    # Simple test case with 2 words and dimension 3
    input_embeddings = np.array([
        [1, 0, 1],  # Word 1
        [0, 1, 1]   # Word 2
    ])
    
    # Different weight matrices for Q, K, V
    W_q = np.array([
        [2, 0, 0],  # Amplify first dimension
        [0, 1, 1],  # Mix second and third dimensions
        [0, 1, 1]   # Mix second and third dimensions
    ])
    
    W_k = np.array([
        [1, 1, 0],  # Mix first and second dimensions
        [1, 1, 0],  # Mix first and second dimensions
        [0, 0, 2]   # Amplify third dimension
    ])
    
    W_v = np.array([
        [1, -1, 0], # Contrast first and second dimensions
        [-1, 1, 0], # Opposite contrast
        [0, 0, 1]   # Keep third dimension as is
    ])
    
    # Compute Q, K, V
    Q, K, V = create_qkv(input_embeddings, W_q, W_k, W_v)
    
    print("Input embeddings:")
    print(input_embeddings)
    print("\nWeight matrix for Q:")
    print(W_q)
    print("\nQ matrix (what each word is looking for):")
    print(Q)
    print("\nWeight matrix for K:")
    print(W_k)
    print("\nK matrix (what each word contains):")
    print(K)
    print("\nWeight matrix for V:")
    print(W_v)
    print("\nV matrix (what each word offers):")
    print(V)
    
    # Test attention scores
    d_model = 3
    scores = compute_attention_scores(Q, K, d_model)
    print("\nAttention scores:")
    print(scores)
    
    # Test softmax
    weights = apply_softmax(scores)
    print("\nAttention weights (after softmax):")
    print(weights)
    print("\nRow sums (should be 1):")
    print(np.sum(weights, axis=1))
    
    # Test final attention output
    output = compute_attention_output(weights, V)
    print("\nFinal attention output:")
    print(output)
    print("\nOutput shape:", output.shape)

if __name__ == "__main__":
    test_attention() 