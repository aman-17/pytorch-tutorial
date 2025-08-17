"""
Question: Attention Mechanism Gradients and Optimization (Medium-Hard)

Implement the attention mechanism with focus on gradient computation, numerical stability,
and optimization techniques. This question dives deep into the mathematical details of
how gradients flow through the attention mechanism.

Key concepts covered:
1. Scaled dot-product attention forward and backward pass
2. Softmax gradient computation and numerical stability
3. Gradient accumulation in multi-head attention
4. Attention pattern analysis and gradient-based interpretation
5. Memory-efficient attention implementations

Your task: Implement attention with detailed gradient tracking and analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List

class AttentionMechanism:
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize attention parameters"""
        # TODO: Initialize weight matrices with proper scaling
        # W_q, W_k, W_v: (d_model, d_model)
        # W_o: (d_model, d_model)
        # Use Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        pass
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Implement scaled dot-product attention with gradient tracking.
        
        Q, K, V: (batch_size, n_heads, seq_len, d_k)
        mask: (batch_size, n_heads, seq_len, seq_len) or None
        
        Returns: 
        - attention_output: (batch_size, n_heads, seq_len, d_k)
        - attention_weights: (batch_size, n_heads, seq_len, seq_len)
        - cache: dict with intermediate values for backward pass
        """
        batch_size, n_heads, seq_len, d_k = Q.shape
        
        # TODO: Compute attention scores
        # scores = Q @ K^T / sqrt(d_k)
        # Be careful about the matrix multiplication dimensions
        
        # TODO: Apply mask if provided
        # masked_scores = scores + mask (where mask has -inf for masked positions)
        
        # TODO: Apply softmax with numerical stability
        # Use the numerically stable version:
        # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        # TODO: Apply attention weights to values
        # attention_output = attention_weights @ V
        
        cache = {
            'Q': Q,
            'K': K, 
            'V': V,
            'scores': None,  # TODO: store computed values
            'masked_scores': None,
            'attention_weights': None,
            'mask': mask,
            'd_k': d_k
        }
        
        return None, None, cache  # TODO: return (attention_output, attention_weights, cache)
    
    def scaled_dot_product_attention_backward(self, dout: np.ndarray, 
                                            attention_weights: np.ndarray, 
                                            cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for scaled dot-product attention.
        
        dout: gradient w.r.t. attention output (batch_size, n_heads, seq_len, d_k)
        attention_weights: (batch_size, n_heads, seq_len, seq_len)
        
        Returns: dQ, dK, dV gradients
        """
        Q, K, V = cache['Q'], cache['K'], cache['V']
        d_k = cache['d_k']
        
        # TODO: Gradient w.r.t. V
        # dV = attention_weights^T @ dout
        
        # TODO: Gradient w.r.t. attention_weights
        # d_attention_weights = dout @ V^T
        
        # TODO: Gradient through softmax (this is the tricky part!)
        # For softmax gradient: d_softmax = softmax * (d_upstream - sum(d_upstream * softmax, axis=-1, keepdims=True))
        
        # TODO: Gradient w.r.t. scores (before softmax)
        # d_scores = softmax_gradient
        
        # TODO: Apply mask gradient if mask was used
        # Masked positions should have zero gradient
        
        # TODO: Gradient w.r.t. Q and K
        # dQ = d_scores @ K / sqrt(d_k)
        # dK = d_scores^T @ Q / sqrt(d_k)
        
        return None, None, None  # TODO: return (dQ, dK, dV)
    
    def multi_head_attention_forward(self, x: np.ndarray, 
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention forward pass.
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # TODO: Compute Q, K, V projections
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        
        # TODO: Reshape for multi-head attention
        # From (batch_size, seq_len, d_model) to (batch_size, n_heads, seq_len, d_k)
        # Hint: use reshape and transpose operations
        
        # TODO: Apply scaled dot-product attention
        # attention_output, attention_weights, attention_cache = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # TODO: Concatenate heads and apply output projection
        # Reshape back to (batch_size, seq_len, d_model)
        # Apply final linear transformation: output @ W_o
        
        cache = {
            'x': x,
            'Q_proj': None,  # TODO: store projected Q, K, V
            'K_proj': None,
            'V_proj': None,
            'Q_heads': None,  # TODO: store reshaped Q, K, V for heads
            'K_heads': None,
            'V_heads': None,
            'attention_output': None,
            'attention_weights': None,
            'attention_cache': None,
            'concat_output': None
        }
        
        return None, cache  # TODO: return (output, cache)
    
    def multi_head_attention_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention backward pass.
        """
        # TODO: Implement backward pass through multi-head attention
        # Work backwards through:
        # 1. Output projection (W_o)
        # 2. Concatenation of heads
        # 3. Scaled dot-product attention for each head
        # 4. Head reshaping
        # 5. Q, K, V projections
        
        grads = {
            'dW_q': None,  # TODO: compute parameter gradients
            'dW_k': None,
            'dW_v': None,
            'dW_o': None
        }
        
        return None, grads  # TODO: return (dx, grads)
    
    def analyze_attention_patterns(self, x: np.ndarray, 
                                 layer_names: List[str] = None) -> Dict:
        """
        Analyze attention patterns and their gradients.
        """
        # TODO: Forward pass to get attention weights
        output, cache = self.multi_head_attention_forward(x)
        
        if output is None:
            return {'error': 'Forward pass not implemented'}
            
        attention_weights = cache['attention_weights']
        
        analysis = {
            'attention_weights': attention_weights,
            'entropy': None,  # TODO: compute attention entropy
            'max_attention': None,  # TODO: find max attention values
            'attention_distance': None,  # TODO: average attention distance
        }
        
        # TODO: Compute attention entropy for each head
        # entropy = -sum(p * log(p)) where p is attention weights
        
        # TODO: Compute average attention distance
        # How far on average does each position attend to?
        
        # TODO: Find positions with highest attention values
        
        return analysis
    
    def compute_gradient_norms(self, x: np.ndarray, target: np.ndarray) -> Dict:
        """
        Compute gradient norms for analysis of training dynamics.
        """
        # TODO: Forward pass
        output, cache = self.multi_head_attention_forward(x)
        
        if output is None:
            return {'error': 'Forward pass not implemented'}
        
        # TODO: Compute loss (e.g., MSE with target)
        # loss = 0.5 * sum((output - target)^2)
        
        # TODO: Backward pass
        # dout = output - target
        dx, grads = self.multi_head_attention_backward(None, cache)  # TODO: pass correct dout
        
        # TODO: Compute gradient norms
        grad_norms = {
            'input_grad_norm': None,  # TODO: ||dx||
            'W_q_grad_norm': None,    # TODO: ||dW_q||
            'W_k_grad_norm': None,    # TODO: ||dW_k||
            'W_v_grad_norm': None,    # TODO: ||dW_v||
            'W_o_grad_norm': None,    # TODO: ||dW_o||
            'total_grad_norm': None   # TODO: sum of all parameter grad norms
        }
        
        return grad_norms

def visualize_attention_patterns(attention_weights: np.ndarray, tokens: List[str] = None):
    """
    Visualize attention patterns as heatmaps.
    attention_weights: (n_heads, seq_len, seq_len)
    """
    # TODO: Create heatmap visualization of attention patterns
    # Show attention weights for each head
    # If tokens are provided, use them as axis labels
    
    n_heads = attention_weights.shape[0]
    
    # TODO: Create subplot for each attention head
    # Use matplotlib to create heatmaps
    # Add colorbar and proper labels
    
    pass

def gradient_flow_analysis(model: AttentionMechanism, x: np.ndarray, 
                         num_steps: int = 10) -> Dict:
    """
    Analyze how gradients flow through attention during training.
    """
    # TODO: Simulate training steps and track gradient statistics
    
    results = {
        'step': [],
        'grad_norms': [],
        'attention_entropy': [],
        'loss': []
    }
    
    # TODO: For each training step:
    # 1. Forward pass
    # 2. Compute loss (random target for simulation)
    # 3. Backward pass  
    # 4. Track gradient norms and attention statistics
    # 5. Update parameters (simple SGD)
    
    return results

def test_gradient_correctness(model: AttentionMechanism, x: np.ndarray, epsilon: float = 1e-5):
    """
    Test gradient correctness using numerical differentiation.
    """
    print("=== Testing Gradient Correctness ===")
    
    # TODO: Implement numerical gradient checking
    # 1. Compute analytical gradients using backward pass
    # 2. Compute numerical gradients using finite differences
    # 3. Compare relative error for each parameter
    
    # For each parameter W:
    # numerical_grad[i,j] = (f(W + eps*e_ij) - f(W - eps*e_ij)) / (2*eps)
    # where e_ij is unit vector with 1 at position (i,j)
    
    pass

def attention_memory_analysis(d_model: int, seq_lengths: List[int], n_heads: int = 8):
    """
    Analyze memory usage of attention mechanism for different sequence lengths.
    """
    print("=== Attention Memory Analysis ===")
    
    # TODO: For each sequence length, compute:
    # 1. Memory for storing Q, K, V matrices
    # 2. Memory for attention score matrix (seq_len^2)
    # 3. Memory for gradients
    # 4. Total memory usage
    
    results = {
        'seq_lengths': seq_lengths,
        'qkv_memory': [],      # Memory for Q, K, V
        'scores_memory': [],   # Memory for attention scores  
        'gradients_memory': [], # Memory for gradients
        'total_memory': []     # Total memory
    }
    
    for seq_len in seq_lengths:
        # TODO: Calculate memory requirements
        # Assume float32 (4 bytes per parameter)
        
        # QKV memory: 3 * batch_size * seq_len * d_model * 4 bytes
        # Scores memory: batch_size * n_heads * seq_len^2 * 4 bytes
        # etc.
        
        pass
    
    return results

# Test your implementation
if __name__ == "__main__":
    print("=== Attention Mechanism Gradients Test ===")
    
    # Model parameters
    d_model = 64
    n_heads = 4
    batch_size = 2
    seq_len = 8
    
    # Create model and test data
    model = AttentionMechanism(d_model, n_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Model: d_model={d_model}, n_heads={n_heads}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    output, cache = model.multi_head_attention_forward(x)
    
    if output is not None:
        print(f"✓ Forward pass successful")
        print(f"Output shape: {output.shape}")
        
        # Test backward pass
        print("\n=== Testing Backward Pass ===")
        dout = np.random.randn(*output.shape)
        dx, grads = model.multi_head_attention_backward(dout, cache)
        
        if dx is not None:
            print(f"✓ Backward pass successful")
            print(f"Input gradient shape: {dx.shape}")
            
            # Gradient correctness test
            test_gradient_correctness(model, x)
            
            # Attention analysis
            print("\n=== Attention Pattern Analysis ===")
            analysis = model.analyze_attention_patterns(x)
            if 'error' not in analysis:
                print(f"✓ Attention analysis completed")
                
            # Gradient flow analysis
            print("\n=== Gradient Flow Analysis ===")
            flow_results = gradient_flow_analysis(model, x)
            print(f"✓ Gradient flow analysis completed")
            
        else:
            print("✗ Backward pass not implemented")
    else:
        print("✗ Forward pass not implemented")
    
    # Memory analysis
    print("\n=== Memory Analysis ===")
    seq_lengths = [64, 128, 256, 512, 1024]
    memory_results = attention_memory_analysis(d_model, seq_lengths, n_heads)
    print(f"Memory analysis for sequence lengths: {seq_lengths}")
    
    print("\n=== Implementation Tips ===")
    print("1. Pay special attention to softmax gradient computation")
    print("2. Use numerically stable softmax (subtract max before exp)")
    print("3. Be careful with matrix dimension ordering in multi-head attention")
    print("4. Test with numerical gradient checking")
    print("5. Consider memory-efficient implementations for long sequences")