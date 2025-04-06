import numpy as np

def create_qkv(input_embeddings, W_q, W_k, W_v):
    """
    Creates Query, Key, Value matrices from input embeddings using provided weight matrices.
    
    Args:
        input_embeddings: numpy array of shape (seq_len, d_model)
        W_q: Query weight matrix of shape (d_model, d_model)
        W_k: Key weight matrix of shape (d_model, d_model)
        W_v: Value weight matrix of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices of shape (seq_len, d_model)
    """
    return np.dot(input_embeddings, W_q), np.dot(input_embeddings, W_k), np.dot(input_embeddings, W_v)

def compute_attention_scores(Q, K, d_model):
    """
    Computes attention scores between all pairs of words.
    
    Args:
        Q: Query matrix of shape (seq_len, d_model)
        K: Key matrix of shape (seq_len, d_model)
        d_model: dimension of the model (for scaling)
    
    Returns:
        Attention scores matrix of shape (seq_len, seq_len)
    """
    # TODO: Your implementation here
    # This gives a square matrix of dim word length x word length
    # (i, j) element in the matrix is the dot product of ith row in Q and jth row in K
    # Then scaled by 1/sqrt(d_model)
    return np.dot(Q, K.T) / np.sqrt(d_model)

def apply_softmax(scores):
    """
    Apply softmax to attention scores to get attention weights.
    
    Args:
        scores: Attention scores matrix of shape (sequence_length, sequence_length)
        
    Returns:
        weights: Attention weights matrix of shape (sequence_length, sequence_length)
    """
    # Subtract max for numerical stability
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    # Normalize
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return weights

def compute_attention_output(weights, V):
    """
    Compute the final attention output by combining values according to attention weights.
    
    Args:
        weights: Attention weights matrix of shape (sequence_length, sequence_length)
        V: Value matrix of shape (sequence_length, d_model)
        
    Returns:
        output: Attention output matrix of shape (sequence_length, d_model)
    """
    return np.matmul(weights, V)