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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
import math

class AttentionMechanism(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize attention parameters"""
        # Initialize weight matrices with proper scaling
        # W_q, W_k, W_v: (d_model, d_model)
        # W_o: (d_model, d_model)
        # Use Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        std = math.sqrt(2.0 / (self.d_model + self.d_model))
        
        self.W_q = nn.Parameter(torch.randn(self.d_model, self.d_model) * std)
        self.W_k = nn.Parameter(torch.randn(self.d_model, self.d_model) * std)
        self.W_v = nn.Parameter(torch.randn(self.d_model, self.d_model) * std)
        self.W_o = nn.Parameter(torch.randn(self.d_model, self.d_model) * std)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
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
        
        # Compute attention scores
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask  # mask has -inf for masked positions
        masked_scores = scores
        
        # Apply softmax with numerical stability
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)
        
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
    
    def scaled_dot_product_attention_backward(self, dout: torch.Tensor, 
                                            attention_weights: torch.Tensor, 
                                            cache: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass for scaled dot-product attention.
        
        dout: gradient w.r.t. attention output (batch_size, n_heads, seq_len, d_k)
        attention_weights: (batch_size, n_heads, seq_len, seq_len)
        
        Returns: dQ, dK, dV gradients
        """
        Q, K, V = cache['Q'], cache['K'], cache['V']
        d_k = cache['d_k']
        
        # Gradient w.r.t. V
        dV = torch.matmul(attention_weights.transpose(-2, -1), dout)
        
        # Gradient w.r.t. attention_weights
        d_attention_weights = torch.matmul(dout, V.transpose(-2, -1))
        
        # Gradient through softmax (this is the tricky part!)
        # For softmax gradient: d_softmax = softmax * (d_upstream - (d_upstream * softmax).sum(dim=-1, keepdim=True))
        softmax_sum = torch.sum(d_attention_weights * attention_weights, dim=-1, keepdim=True)
        d_scores = attention_weights * (d_attention_weights - softmax_sum)
        
        # Apply mask gradient if mask was used
        if cache['mask'] is not None:
            d_scores = d_scores.masked_fill(cache['mask'] == float('-inf'), 0.0)
        
        # Gradient w.r.t. Q and K
        dQ = torch.matmul(d_scores, K) / math.sqrt(d_k)
        dK = torch.matmul(d_scores.transpose(-2, -1), Q) / math.sqrt(d_k)
        
        return dQ, dK, dV
    
    def multi_head_attention_forward(self, x: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-head attention forward pass.
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V projections
        Q_proj = torch.matmul(x, self.W_q)
        K_proj = torch.matmul(x, self.W_k)
        V_proj = torch.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        # From (batch_size, seq_len, d_model) to (batch_size, n_heads, seq_len, d_k)
        Q_heads = Q_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights, attention_cache = self.scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        
        # Concatenate heads and apply output projection
        # Reshape back to (batch_size, seq_len, d_model)
        concat_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = torch.matmul(concat_output, self.W_o)
        
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
    
    def multi_head_attention_backward(self, dout: torch.Tensor, cache: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-head attention backward pass.
        """
        # Implement backward pass through multi-head attention
        # Work backwards through:
        # 1. Output projection (W_o)
        dW_o = torch.matmul(cache['concat_output'].transpose(-2, -1), dout)
        d_concat = torch.matmul(dout, self.W_o.transpose(-2, -1))
        
        # 2. Concatenation of heads
        batch_size, seq_len, d_model = d_concat.shape
        d_attention_output = d_concat.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled dot-product attention for each head
        dQ_heads, dK_heads, dV_heads = self.scaled_dot_product_attention_backward(
            d_attention_output, cache['attention_weights'], cache['attention_cache'])
        
        # 4. Head reshaping
        dQ_proj = dQ_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dK_proj = dK_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dV_proj = dV_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 5. Q, K, V projections
        dW_q = torch.matmul(cache['x'].transpose(-2, -1), dQ_proj)
        dW_k = torch.matmul(cache['x'].transpose(-2, -1), dK_proj)
        dW_v = torch.matmul(cache['x'].transpose(-2, -1), dV_proj)
        
        dx_q = torch.matmul(dQ_proj, self.W_q.transpose(-2, -1))
        dx_k = torch.matmul(dK_proj, self.W_k.transpose(-2, -1))
        dx_v = torch.matmul(dV_proj, self.W_v.transpose(-2, -1))
        dx = dx_q + dx_k + dx_v
        
        grads = {
            'dW_q': dW_q,
            'dW_k': dW_k,
            'dW_v': dW_v,
            'dW_o': dW_o
        }
        
        return dx, grads
    
    def analyze_attention_patterns(self, x: torch.Tensor, 
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
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
        entropy = torch.mean(entropy, dim=(0, 2))  # Average over batch and sequence
        
        # Compute average attention distance
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        position_indices = torch.arange(seq_len, device=attention_weights.device).float()
        position_diffs = position_indices.unsqueeze(0) - position_indices.unsqueeze(1)
        abs_position_diffs = torch.abs(position_diffs)
        attention_distance = torch.sum(attention_weights * abs_position_diffs.unsqueeze(0).unsqueeze(0), dim=-1)
        attention_distance = torch.mean(attention_distance, dim=(0, 2))
        
        # Find positions with highest attention values
        max_attention, max_indices = torch.max(attention_weights, dim=-1)
        max_attention = torch.mean(max_attention, dim=(0, 2))
        
        analysis = {
            'attention_weights': attention_weights,
            'entropy': entropy,
            'max_attention': max_attention,
            'attention_distance': attention_distance,
        }
        
        return analysis
    
    def compute_gradient_norms(self, x: torch.Tensor, target: torch.Tensor) -> Dict:
        """
        Compute gradient norms for analysis of training dynamics.
        """
        # Forward pass
        output, cache = self.multi_head_attention_forward(x)
        
        if output is None:
            return {'error': 'Forward pass not implemented'}
        
        # Compute loss (e.g., MSE with target)
        loss = F.mse_loss(output, target)
        
        # Backward pass using torch.autograd
        loss.backward(retain_graph=True)
        
        # Compute gradient norms
        grad_norms = {
            'input_grad_norm': torch.norm(x.grad).item() if x.grad is not None else 0.0,
            'W_q_grad_norm': torch.norm(self.W_q.grad).item() if self.W_q.grad is not None else 0.0,
            'W_k_grad_norm': torch.norm(self.W_k.grad).item() if self.W_k.grad is not None else 0.0,
            'W_v_grad_norm': torch.norm(self.W_v.grad).item() if self.W_v.grad is not None else 0.0,
            'W_o_grad_norm': torch.norm(self.W_o.grad).item() if self.W_o.grad is not None else 0.0,
        }
        grad_norms['total_grad_norm'] = sum(grad_norms.values())
        
        return grad_norms

def visualize_attention_patterns(attention_weights: torch.Tensor, tokens: List[str] = None):
    """
    Visualize attention patterns as heatmaps.
    attention_weights: (n_heads, seq_len, seq_len)
    """
    # Create heatmap visualization of attention patterns
    # Convert torch tensor to numpy for matplotlib
    attention_np = attention_weights.detach().cpu().numpy()
    
    n_heads = attention_weights.shape[0]
    
    # Create subplot for each attention head
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for head in range(n_heads):
        im = axes[head].imshow(attention_np[head], cmap='Blues', aspect='auto')
        axes[head].set_title(f'Head {head + 1}')
        
        if tokens is not None:
            axes[head].set_xticks(range(len(tokens)))
            axes[head].set_yticks(range(len(tokens)))
            axes[head].set_xticklabels(tokens, rotation=45)
            axes[head].set_yticklabels(tokens)
        
        plt.colorbar(im, ax=axes[head])
    
    plt.tight_layout()
    plt.show()

def gradient_flow_analysis(model: AttentionMechanism, x: torch.Tensor, 
                         num_steps: int = 10) -> Dict:
    """
    Analyze how gradients flow through attention during training.
    """
    # Simulate training steps and track gradient statistics
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
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
        target = torch.randn_like(output)
        loss = F.mse_loss(output, target)
        
        # 3. Backward pass using torch.autograd
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Track gradient norms and attention statistics
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Compute attention entropy
        attention_weights = cache['attention_weights']
        eps = 1e-8
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
        avg_entropy = torch.mean(entropy).item()
        
        results['step'].append(step)
        results['grad_norms'].append(total_grad_norm)
        results['attention_entropy'].append(avg_entropy)
        results['loss'].append(loss.item())
        
        # 5. Update parameters (simple SGD)
        optimizer.step()
    
    return results

def test_gradient_correctness(model: AttentionMechanism, x: torch.Tensor, epsilon: float = 1e-5):
    """
    Test gradient correctness using torch.autograd.gradcheck.
    """
    print("=== Testing Gradient Correctness ===")
    
    # Use torch.autograd.gradcheck to verify gradients
    try:
        # Create a simple function that computes loss
        def attention_loss(input_tensor):
            output, _ = model.multi_head_attention_forward(input_tensor)
            return torch.sum(output ** 2)
        
        # Test gradient correctness
        test_passed = torch.autograd.gradcheck(attention_loss, x, eps=epsilon, atol=1e-4)
        
        if test_passed:
            print("✓ Gradient check passed!")
        else:
            print("✗ Gradient check failed!")
            
    except Exception as e:
        print(f"✗ Gradient check failed with error: {e}")

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
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
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
        dout = torch.randn(*output.shape)
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
    print("1. Use torch.autograd.gradcheck for gradient verification")
    print("2. Use F.softmax with dim=-1 for numerically stable softmax")
    print("3. Be careful with tensor dimensions in multi-head attention")
    print("4. Use tensor.requires_grad_(True) for gradient tracking")
    print("5. Consider using torch.nn.MultiheadAttention for comparison")
    print("6. Use torch.cuda.memory_summary() for memory profiling")