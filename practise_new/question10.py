"""
Question 10: Efficient Transformer with Linear Attention (Hard)

Implement an efficient Transformer variant that uses linear attention instead of standard
quadratic attention. Linear attention reduces complexity from O(n²) to O(n) for sequence
length n, making it suitable for very long sequences.

Key concepts:
1. Linear attention mechanism using kernel methods
2. Feature maps φ(x) and ψ(x) for queries and keys
3. Associative property: Attention(Q,K,V) = φ(Q)(ψ(K)ᵀV)
4. Causal masking for autoregressive generation
5. Layer normalization and residual connections

Requirements:
- Implement both non-causal and causal linear attention
- Use ELU + 1 feature map for positive values
- Support multi-head attention
- Compare performance with standard attention

Formula: LinearAttention(Q,K,V) = φ(Q) * (ψ(K)ᵀ * V) / (φ(Q) * ψ(K)ᵀ * 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def elu_feature_map(x):
    """ELU + 1 feature map to ensure positive values"""
    # TODO: Implement φ(x) = ELU(x) + 1
    # This ensures all values are positive for the kernel method
    return F.elu(x) + 1

class LinearAttention(nn.Module):
    """Linear attention mechanism with O(n) complexity"""
    def __init__(self, dim, heads=8, causal=False):
        super(LinearAttention, self).__init__()
        self.heads = heads
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor (batch_size, seq_len, dim) 
            mask: attention mask (optional)
        Returns:
            output tensor (batch_size, seq_len, dim)
        """
        # TODO: Implement linear attention forward pass
        # 1. Compute Q, K, V projections
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.heads, -1)
        k = self.k_proj(x).view(batch_size, seq_len, self.heads, -1)
        v = self.v_proj(x).view(batch_size, seq_len, self.heads, -1)

        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 
        # 2. Apply feature maps φ and ψ
        q = elu_feature_map(q)
        k = elu_feature_map(k)
        v = elu_feature_map(v)


        # 4. Handle causal masking if needed
        # 5. Apply output projection
        pass
    
    def causal_linear_attention(self, q, k, v):
        """Efficient causal linear attention using running sums"""
        batch_size, heads, seq_len, head_dim = q.shape
        
        # Initialize running sums - this is the key for O(n) complexity
        kv_sum = torch.zeros(batch_size, heads, head_dim, head_dim, device=q.device)
        k_sum = torch.zeros(batch_size, heads, head_dim, device=q.device)
        
        outputs = []
        
        for i in range(seq_len):
            # Current query
            q_i = q[:, :, i]  # (batch, heads, head_dim)
            k_i = k[:, :, i]  # (batch, heads, head_dim)
            v_i = v[:, :, i]  # (batch, heads, head_dim)
            
            # Update running sums (O(1) operation!)
            kv_sum += k_i.unsqueeze(-1) @ v_i.unsqueeze(-2)  # outer product
            k_sum += k_i
            
            # Compute attention for current position
            numerator = q_i.unsqueeze(-2) @ kv_sum  # (batch, heads, 1, head_dim)
            denominator = (q_i * k_sum).sum(dim=-1, keepdim=True)  # (batch, heads, 1)
            
            output_i = numerator.squeeze(-2) / (denominator + 1e-8)
            outputs.append(output_i)
        
        return torch.stack(outputs, dim=2)  # (batch, heads, seq_len, head_dim)

class LinearTransformerBlock(nn.Module):
    """Transformer block with linear attention"""
    def __init__(self, dim, heads=8, mlp_ratio=4, causal=False, dropout=0.1):
        super(LinearTransformerBlock, self).__init__()
        # TODO: Initialize transformer block components
        # - Linear attention layer
        # - Feed-forward network  
        # - Layer normalization
        # - Dropout
        pass
    
    def forward(self, x, mask=None):
        # TODO: Implement transformer block forward pass
        # 1. Self-attention with residual connection and layer norm
        # 2. Feed-forward with residual connection and layer norm
        pass

class LinearTransformer(nn.Module):
    """Complete Linear Transformer model"""
    def __init__(self, vocab_size, dim, depth, heads=8, causal=True, 
                 max_seq_len=1024, dropout=0.1):
        super(LinearTransformer, self).__init__()
        # TODO: Initialize complete transformer
        # - Token embedding
        # - Positional encoding
        # - Transformer blocks
        # - Output head
        pass
    
    def forward(self, x, mask=None):
        # TODO: Forward pass through full transformer
        pass

class StandardAttention(nn.Module):
    """Standard quadratic attention for comparison"""
    def __init__(self, dim, heads=8, causal=False):
        super(StandardAttention, self).__init__()
        self.heads = heads
        self.causal = causal
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.heads, -1)
        k = self.k_proj(x).view(batch_size, seq_len, self.heads, -1)
        v = self.v_proj(x).view(batch_size, seq_len, self.heads, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        
        if self.causal and mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

def compare_attention_complexity():
    """Compare computational complexity of linear vs standard attention"""
    # TODO: Benchmark both attention mechanisms
    # Test with different sequence lengths: [128, 512, 1024, 2048]
    # Measure time and memory usage
    # Plot complexity curves
    pass

def test_causal_property():
    """Test that causal linear attention maintains causality"""
    # TODO: Verify causal masking works correctly
    # Compare outputs with different sequence lengths
    # Ensure future tokens don't affect past predictions
    pass

def test_attention_equivalence():
    """Test numerical equivalence between implementations"""
    # TODO: Compare linear and standard attention outputs
    # Should be similar for short sequences
    # Test gradient flow
    pass

# Test your implementation
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test linear attention
    batch_size, seq_len, dim = 2, 128, 256
    heads = 8
    
    # Create models
    linear_attn = LinearAttention(dim, heads, causal=True).to(device)
    standard_attn = StandardAttention(dim, heads, causal=True).to(device)
    
    # Test input
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Test forward passes
    print("Testing Linear Attention...")
    linear_out = linear_attn(x)
    print(f"Linear attention output shape: {linear_out.shape}")
    
    print("Testing Standard Attention...")
    standard_out = standard_attn(x)
    print(f"Standard attention output shape: {standard_out.shape}")
    
    # Test full transformer
    vocab_size = 10000
    model = LinearTransformer(
        vocab_size=vocab_size,
        dim=dim,
        depth=6,
        heads=heads,
        causal=True,
        max_seq_len=512
    ).to(device)
    
    # Test with token ids
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    output = model(token_ids)
    print(f"Transformer output shape: {output.shape}")
    
    # Run complexity comparison
    print("\nRunning complexity comparison...")
    compare_attention_complexity()
    
    # Test causal property
    print("Testing causal property...")
    test_causal_property()
    
    print("Linear Transformer implementation test completed!")