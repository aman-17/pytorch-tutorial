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
        self.out_proj = nn.Linear(dim, dim, bias=False)

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
        
        # 3. Compute linear attention
        if self.causal:
            # Use causal linear attention for autoregressive tasks
            attn_output = self.causal_linear_attention(q, k, v)
        else:
            # Non-causal linear attention: φ(Q) * (ψ(K)ᵀ * V) / (φ(Q) * ψ(K)ᵀ * 1)
            kv = torch.matmul(k.transpose(-2, -1), v)  # (batch, heads, head_dim, head_dim)
            k_sum = k.sum(dim=-2, keepdim=True)  # (batch, heads, 1, head_dim)
            
            numerator = torch.matmul(q, kv)  # (batch, heads, seq_len, head_dim)
            denominator = torch.matmul(q, k_sum.transpose(-2, -1))  # (batch, heads, seq_len, 1)
            
            attn_output = numerator / (denominator + 1e-8)
        
        # 4. Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq_len, dim)
        
        return self.out_proj(attn_output)
    
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
        # Linear attention layer
        self.attn = LinearAttention(dim, heads, causal)
        self.norm1 = nn.LayerNorm(dim)
        
        # Feed-forward network
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        
        # Feed-forward with residual connection and layer norm
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x

class LinearTransformer(nn.Module):
    """Complete Linear Transformer model"""
    def __init__(self, vocab_size, dim, depth, heads=8, causal=True, 
                 max_seq_len=1024, dropout=0.1):
        super(LinearTransformer, self).__init__()
        self.dim = dim
        self.causal = causal
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            LinearTransformerBlock(dim, heads, causal=causal, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Token and positional embeddings
        token_embs = self.token_emb(x)  # (batch, seq_len, dim)
        pos_embs = self.pos_emb[:, :seq_len, :]  # (1, seq_len, dim)
        x = self.dropout(token_embs + pos_embs)
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output head
        x = self.norm(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        
        return logits

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim // self.heads)
        
        if self.causal and mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

def compare_attention_complexity():
    """Compare computational complexity of linear vs standard attention"""
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 256
    heads = 8
    batch_size = 2
    
    seq_lengths = [128, 256, 512, 1024]
    if torch.cuda.is_available():
        seq_lengths.append(2048)
    
    linear_times = []
    standard_times = []
    
    print("Comparing attention mechanisms:")
    print(f"Device: {device}")
    print(f"Dimension: {dim}, Heads: {heads}, Batch size: {batch_size}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        # Create models
        linear_attn = LinearAttention(dim, heads, causal=True).to(device)
        standard_attn = StandardAttention(dim, heads, causal=True).to(device)
        
        # Create input
        x = torch.randn(batch_size, seq_len, dim).to(device)
        
        # Warm up
        for _ in range(5):
            _ = linear_attn(x)
            _ = standard_attn(x)
        
        # Time linear attention
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(10):
            _ = linear_attn(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        linear_time = (time.time() - start) / 10
        
        # Time standard attention
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(10):
            _ = standard_attn(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        standard_time = (time.time() - start) / 10
        
        linear_times.append(linear_time)
        standard_times.append(standard_time)
        
        speedup = standard_time / linear_time
        print(f"Seq len {seq_len:4d}: Linear={linear_time:.4f}s, Standard={standard_time:.4f}s, Speedup={speedup:.2f}x")
    
    print("-" * 60)
    print("Linear attention shows better scaling for longer sequences!")

def test_causal_property():
    """Test that causal linear attention maintains causality"""
    print("Testing causal property...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 128
    heads = 4
    batch_size = 1
    
    # Create causal linear attention
    attn = LinearAttention(dim, heads, causal=True).to(device)
    
    # Test with different sequence lengths
    short_seq_len = 10
    long_seq_len = 15
    
    # Create input sequences
    short_x = torch.randn(batch_size, short_seq_len, dim).to(device)
    long_x = torch.randn(batch_size, long_seq_len, dim).to(device)
    
    # Copy short sequence to beginning of long sequence
    long_x[:, :short_seq_len] = short_x
    
    # Get outputs
    short_out = attn(short_x)
    long_out = attn(long_x)
    
    # Check that outputs for first short_seq_len positions are identical
    diff = torch.abs(short_out - long_out[:, :short_seq_len]).max()
    
    print(f"Max difference in outputs: {diff.item():.8f}")
    
    if diff < 1e-5:
        print("✓ Causal property maintained! Future tokens don't affect past predictions.")
    else:
        print("✗ Causal property violated!")
    
    return diff < 1e-5

def test_attention_equivalence():
    """Test numerical equivalence between implementations"""
    print("Testing attention equivalence...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 128
    heads = 4
    batch_size = 2
    seq_len = 64  # Short sequence for better equivalence
    
    # Create both attention mechanisms
    linear_attn = LinearAttention(dim, heads, causal=False).to(device)
    standard_attn = StandardAttention(dim, heads, causal=False).to(device)
    
    # Initialize with same weights
    with torch.no_grad():
        standard_attn.q_proj.weight.copy_(linear_attn.q_proj.weight)
        standard_attn.k_proj.weight.copy_(linear_attn.k_proj.weight)
        standard_attn.v_proj.weight.copy_(linear_attn.v_proj.weight)
        standard_attn.out_proj.weight.copy_(linear_attn.out_proj.weight)
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True).to(device)
    x_copy = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    linear_out = linear_attn(x)
    standard_out = standard_attn(x_copy)
    
    # Compare outputs
    diff = torch.abs(linear_out - standard_out).mean()
    print(f"Mean output difference: {diff.item():.6f}")
    
    # Test gradients
    loss_linear = linear_out.sum()
    loss_standard = standard_out.sum()
    
    loss_linear.backward()
    loss_standard.backward()
    
    grad_diff = torch.abs(x.grad - x_copy.grad).mean()
    print(f"Mean gradient difference: {grad_diff.item():.6f}")
    
    print("Note: Some difference is expected due to different attention mechanisms")
    print("Linear attention approximates standard attention behavior")

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
    
    # Test attention equivalence
    print("\nTesting attention equivalence...")
    test_attention_equivalence()
    
    print("\nLinear Transformer implementation test completed!")