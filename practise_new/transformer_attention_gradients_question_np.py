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
        # Use Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        std = np.sqrt(2.0 / (self.d_model + self.d_model))
        
        self.W_q = np.random.randn(self.d_model, self.d_model) * std
        self.W_k = np.random.randn(self.d_model, self.d_model) * std
        self.W_v = np.random.randn(self.d_model, self.d_model) * std
        self.W_o = np.random.randn(self.d_model, self.d_model) * std
    
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
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask  # mask has -inf for masked positions
        masked_scores = scores.copy()
        
        # Apply softmax with numerical stability
        # Subtract max for numerical stability
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention weights to values
        attention_output = np.matmul(attention_weights, V)
        
        cache = {
            'Q': Q,
            'K': K, 
            'V': V,
            'scores': scores,
            'masked_scores': masked_scores,
            'attention_weights': attention_weights,
            'mask': mask,
            'd_k': d_k
        }
        
        return attention_output, attention_weights, cache
    
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
        
        # Gradient w.r.t. V: dV = attention_weights^T @ dout
        dV = np.matmul(np.swapaxes(attention_weights, -2, -1), dout)
        
        # Gradient w.r.t. attention_weights: d_attention_weights = dout @ V^T
        d_attention_weights = np.matmul(dout, np.swapaxes(V, -2, -1))
        
        # Gradient through softmax (this is the tricky part!)
        # For softmax gradient: d_softmax = softmax * (d_upstream - sum(d_upstream * softmax, axis=-1, keepdims=True))
        softmax_sum = np.sum(d_attention_weights * attention_weights, axis=-1, keepdims=True)
        d_scores = attention_weights * (d_attention_weights - softmax_sum)
        
        # Apply mask gradient if mask was used
        if cache['mask'] is not None:
            # Masked positions should have zero gradient
            mask_zero = (cache['mask'] == -np.inf)
            d_scores[mask_zero] = 0.0
        
        # Gradient w.r.t. Q and K
        # dQ = d_scores @ K / sqrt(d_k)
        # dK = d_scores^T @ Q / sqrt(d_k)
        dQ = np.matmul(d_scores, K) / np.sqrt(d_k)
        dK = np.matmul(np.swapaxes(d_scores, -2, -1), Q) / np.sqrt(d_k)
        
        return dQ, dK, dV
    
    def multi_head_attention_forward(self, x: np.ndarray, 
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention forward pass.
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V projections
        Q_proj = np.matmul(x, self.W_q)  # (batch_size, seq_len, d_model)
        K_proj = np.matmul(x, self.W_k)
        V_proj = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        # From (batch_size, seq_len, d_model) to (batch_size, n_heads, seq_len, d_k)
        Q_heads = Q_proj.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights, attention_cache = self.scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        
        # Concatenate heads and apply output projection
        # Reshape back to (batch_size, seq_len, d_model)
        concat_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        output = np.matmul(concat_output, self.W_o)
        
        cache = {
            'x': x,
            'Q_proj': Q_proj,
            'K_proj': K_proj,
            'V_proj': V_proj,
            'Q_heads': Q_heads,
            'K_heads': K_heads,
            'V_heads': V_heads,
            'attention_output': attention_output,
            'attention_weights': attention_weights,
            'attention_cache': attention_cache,
            'concat_output': concat_output
        }
        
        return output, cache
    
    def multi_head_attention_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention backward pass.
        """
        # Implement backward pass through multi-head attention
        # Work backwards through:
        
        # 1. Output projection (W_o): dW_o = concat_output^T @ dout
        dW_o = np.matmul(cache['concat_output'].swapaxes(-2, -1), dout)
        d_concat = np.matmul(dout, self.W_o.T)
        
        # 2. Concatenation of heads
        batch_size, seq_len, d_model = d_concat.shape
        d_attention_output = d_concat.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 3. Scaled dot-product attention for each head
        dQ_heads, dK_heads, dV_heads = self.scaled_dot_product_attention_backward(
            d_attention_output, cache['attention_weights'], cache['attention_cache'])
        
        # 4. Head reshaping - transpose back to projection shape
        dQ_proj = dQ_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        dK_proj = dK_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        dV_proj = dV_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # 5. Q, K, V projections
        dW_q = np.matmul(cache['x'].swapaxes(-2, -1), dQ_proj)
        dW_k = np.matmul(cache['x'].swapaxes(-2, -1), dK_proj)
        dW_v = np.matmul(cache['x'].swapaxes(-2, -1), dV_proj)
        
        # Input gradients
        dx_q = np.matmul(dQ_proj, self.W_q.T)
        dx_k = np.matmul(dK_proj, self.W_k.T)
        dx_v = np.matmul(dV_proj, self.W_v.T)
        dx = dx_q + dx_k + dx_v
        
        grads = {
            'dW_q': dW_q,
            'dW_k': dW_k,
            'dW_v': dW_v,
            'dW_o': dW_o
        }
        
        return dx, grads
    
    def analyze_attention_patterns(self, x: np.ndarray, 
                                 layer_names: List[str] = None) -> Dict:
        """
        Analyze attention patterns and their gradients.
        """
        # Forward pass to get attention weights
        output, cache = self.multi_head_attention_forward(x)
        
        if output is None:
            return {'error': 'Forward pass not implemented'}
            
        attention_weights = cache['attention_weights']
        
        # Compute attention entropy for each head
        # entropy = -sum(p * log(p)) where p is attention weights
        eps = 1e-8
        entropy = -np.sum(attention_weights * np.log(attention_weights + eps), axis=-1)
        entropy = np.mean(entropy, axis=(0, 2))  # Average over batch and sequence
        
        # Compute average attention distance
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        position_indices = np.arange(seq_len)
        position_diffs = position_indices[:, np.newaxis] - position_indices[np.newaxis, :]
        abs_position_diffs = np.abs(position_diffs)
        attention_distance = np.sum(attention_weights * abs_position_diffs[np.newaxis, np.newaxis, :, :], axis=-1)
        attention_distance = np.mean(attention_distance, axis=(0, 2))
        
        # Find positions with highest attention values
        max_attention = np.max(attention_weights, axis=-1)
        max_attention = np.mean(max_attention, axis=(0, 2))
        
        analysis = {
            'attention_weights': attention_weights,
            'entropy': entropy,
            'max_attention': max_attention,
            'attention_distance': attention_distance,
        }
        
        return analysis
    
    def compute_gradient_norms(self, x: np.ndarray, target: np.ndarray) -> Dict:
        """
        Compute gradient norms for analysis of training dynamics.
        """
        # Forward pass
        output, cache = self.multi_head_attention_forward(x)
        
        if output is None:
            return {'error': 'Forward pass not implemented'}
        
        # Compute loss (e.g., MSE with target)
        loss = 0.5 * np.sum((output - target) ** 2)
        
        # Backward pass
        dout = output - target
        dx, grads = self.multi_head_attention_backward(dout, cache)
        
        # Compute gradient norms
        grad_norms = {
            'input_grad_norm': np.linalg.norm(dx) if dx is not None else 0.0,
            'W_q_grad_norm': np.linalg.norm(grads['dW_q']) if grads['dW_q'] is not None else 0.0,
            'W_k_grad_norm': np.linalg.norm(grads['dW_k']) if grads['dW_k'] is not None else 0.0,
            'W_v_grad_norm': np.linalg.norm(grads['dW_v']) if grads['dW_v'] is not None else 0.0,
            'W_o_grad_norm': np.linalg.norm(grads['dW_o']) if grads['dW_o'] is not None else 0.0,
        }
        grad_norms['total_grad_norm'] = sum(grad_norms.values())
        
        return grad_norms

def visualize_attention_patterns(attention_weights: np.ndarray, tokens: List[str] = None):
    """
    Visualize attention patterns as heatmaps.
    attention_weights: (n_heads, seq_len, seq_len)
    """
    n_heads = attention_weights.shape[0]
    
    # Create subplot for each attention head
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for head in range(n_heads):
        im = axes[head].imshow(attention_weights[head], cmap='Blues', aspect='auto')
        axes[head].set_title(f'Head {head + 1}')
        
        if tokens is not None:
            axes[head].set_xticks(range(len(tokens)))
            axes[head].set_yticks(range(len(tokens)))
            axes[head].set_xticklabels(tokens, rotation=45)
            axes[head].set_yticklabels(tokens)
        
        plt.colorbar(im, ax=axes[head])
    
    plt.tight_layout()
    plt.show()

def gradient_flow_analysis(model: AttentionMechanism, x: np.ndarray, 
                         num_steps: int = 10) -> Dict:
    """
    Analyze how gradients flow through attention during training.
    """
    # Simulate training steps and track gradient statistics
    learning_rate = 0.01
    
    results = {
        'step': [],
        'grad_norms': [],
        'attention_entropy': [],
        'loss': []
    }
    
    # For each training step:
    for step in range(num_steps):
        # 1. Forward pass
        output, cache = model.multi_head_attention_forward(x)
        if output is None:
            break
            
        # 2. Compute loss (random target for simulation)
        target = np.random.randn(*output.shape)
        loss = 0.5 * np.sum((output - target) ** 2)
        
        # 3. Backward pass
        dout = output - target
        dx, grads = model.multi_head_attention_backward(dout, cache)
        
        # 4. Track gradient norms and attention statistics
        total_grad_norm = sum(np.linalg.norm(grad) for grad in grads.values() if grad is not None)
        
        # Compute attention entropy
        attention_weights = cache['attention_weights']
        eps = 1e-8
        entropy = -np.sum(attention_weights * np.log(attention_weights + eps), axis=-1)
        avg_entropy = np.mean(entropy)
        
        results['step'].append(step)
        results['grad_norms'].append(total_grad_norm)
        results['attention_entropy'].append(avg_entropy)
        results['loss'].append(loss)
        
        # 5. Update parameters (simple SGD) - sum over batch dimension if needed
        dW_q = grads['dW_q']
        if dW_q.ndim == 3:  # (batch, seq, d_model) -> (seq, d_model)
            dW_q = np.mean(dW_q, axis=0)
        model.W_q -= learning_rate * dW_q
        
        dW_k = grads['dW_k'] 
        if dW_k.ndim == 3:
            dW_k = np.mean(dW_k, axis=0)
        model.W_k -= learning_rate * dW_k
        
        dW_v = grads['dW_v']
        if dW_v.ndim == 3:
            dW_v = np.mean(dW_v, axis=0)
        model.W_v -= learning_rate * dW_v
        
        dW_o = grads['dW_o']
        if dW_o.ndim == 3:
            dW_o = np.mean(dW_o, axis=0)
        model.W_o -= learning_rate * dW_o
    
    return results

def test_gradient_correctness(model: AttentionMechanism, x: np.ndarray, epsilon: float = 1e-5):
    """
    Test gradient correctness using numerical differentiation.
    """
    print("=== Testing Gradient Correctness ===")
    
    # Create random target
    target = np.random.randn(x.shape[0], x.shape[1], model.d_model)
    
    def compute_loss(W_q, W_k, W_v, W_o):
        # Save original weights
        orig_W_q, orig_W_k, orig_W_v, orig_W_o = model.W_q, model.W_k, model.W_v, model.W_o
        
        # Set new weights
        model.W_q, model.W_k, model.W_v, model.W_o = W_q, W_k, W_v, W_o
        
        # Forward pass
        output, _ = model.multi_head_attention_forward(x)
        loss = 0.5 * np.sum((output - target) ** 2)
        
        # Restore weights
        model.W_q, model.W_k, model.W_v, model.W_o = orig_W_q, orig_W_k, orig_W_v, orig_W_o
        
        return loss
    
    # Compute analytical gradients
    output, cache = model.multi_head_attention_forward(x)
    dout = output - target
    _, grads = model.multi_head_attention_backward(dout, cache)
    
    # Test a few random elements of each weight matrix
    weights = [('W_q', model.W_q, grads['dW_q']), 
               ('W_k', model.W_k, grads['dW_k']),
               ('W_v', model.W_v, grads['dW_v']),
               ('W_o', model.W_o, grads['dW_o'])]
    
    for name, weight, analytical_grad in weights:
        if analytical_grad is None:
            continue
        # Test 2 random positions to avoid indexing issues
        for _ in range(2):
            i, j = np.random.randint(0, min(weight.shape[0], analytical_grad.shape[0])), np.random.randint(0, min(weight.shape[1], analytical_grad.shape[1]))
            
            # Numerical gradient
            weight_plus = weight.copy()
            weight_plus[i, j] += epsilon
            
            weight_minus = weight.copy()
            weight_minus[i, j] -= epsilon
            
            if name == 'W_q':
                loss_plus = compute_loss(weight_plus, model.W_k, model.W_v, model.W_o)
                loss_minus = compute_loss(weight_minus, model.W_k, model.W_v, model.W_o)
            elif name == 'W_k':
                loss_plus = compute_loss(model.W_q, weight_plus, model.W_v, model.W_o)
                loss_minus = compute_loss(model.W_q, weight_minus, model.W_v, model.W_o)
            elif name == 'W_v':
                loss_plus = compute_loss(model.W_q, model.W_k, weight_plus, model.W_o)
                loss_minus = compute_loss(model.W_q, model.W_k, weight_minus, model.W_o)
            else:  # W_o
                loss_plus = compute_loss(model.W_q, model.W_k, model.W_v, weight_plus)
                loss_minus = compute_loss(model.W_q, model.W_k, model.W_v, weight_minus)
            
            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Extract analytical gradient properly - sum over batch if 3D
            if analytical_grad.ndim == 3:
                analytical = np.mean(analytical_grad, axis=0)[i, j]
            else:
                analytical = analytical_grad[i, j]
            
            # Convert to scalars if needed
            if isinstance(analytical, np.ndarray):
                analytical = analytical.item()
            
            if hasattr(numerical_grad, 'item'):
                numerical_grad = numerical_grad.item()
            
            relative_error = abs(numerical_grad - analytical) / (abs(numerical_grad) + abs(analytical) + 1e-8)
            
            # Handle case where relative_error might be an array
            if np.isscalar(relative_error):
                error_val = relative_error
            else:
                error_val = np.mean(relative_error)
                
            if error_val < 1e-5:
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status} {name}[{i},{j}]: analytical={analytical:.6f}, numerical={numerical_grad:.6f}, error={error_val:.2e}")

def attention_memory_analysis(d_model: int, seq_lengths: List[int], n_heads: int = 8):
    """
    Analyze memory usage of attention mechanism for different sequence lengths.
    """
    print("=== Attention Memory Analysis ===")
    
    batch_size = 1  # Assume batch size of 1 for analysis
    bytes_per_float = 4  # float32
    
    results = {
        'seq_lengths': seq_lengths,
        'qkv_memory': [],      # Memory for Q, K, V
        'scores_memory': [],   # Memory for attention scores  
        'gradients_memory': [], # Memory for gradients
        'total_memory': []     # Total memory
    }
    
    for seq_len in seq_lengths:
        # Calculate memory requirements
        # QKV memory: 3 * batch_size * seq_len * d_model * 4 bytes
        qkv_memory = 3 * batch_size * seq_len * d_model * bytes_per_float
        
        # Scores memory: batch_size * n_heads * seq_len^2 * 4 bytes
        scores_memory = batch_size * n_heads * seq_len * seq_len * bytes_per_float
        
        # Gradients memory (approximately same as forward pass)
        gradients_memory = qkv_memory + scores_memory
        
        # Parameter memory: 4 weight matrices of size (d_model, d_model)
        param_memory = 4 * d_model * d_model * bytes_per_float
        
        total_memory = qkv_memory + scores_memory + gradients_memory + param_memory
        
        results['qkv_memory'].append(qkv_memory / (1024**2))  # Convert to MB
        results['scores_memory'].append(scores_memory / (1024**2))
        results['gradients_memory'].append(gradients_memory / (1024**2))
        results['total_memory'].append(total_memory / (1024**2))
        
        print(f"Seq len {seq_len}: QKV={qkv_memory/(1024**2):.1f}MB, "
              f"Scores={scores_memory/(1024**2):.1f}MB, Total={total_memory/(1024**2):.1f}MB")
    
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