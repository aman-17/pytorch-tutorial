import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA)

    In MHA, each head has its own Q, K, V projections.
    For n_heads=8, we have 8 separate K and V matrices.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Each head gets its own Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # d_model -> d_model
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # d_model -> d_model
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # d_model -> d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multiple heads
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose to (batch, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA)

    In MQA, all heads share the same K and V projections.
    For n_heads=8, we have 8 separate Q matrices but only 1 shared K and 1 shared V.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Each head gets its own Q projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # d_model -> d_model

        # BUT K and V are shared across all heads (single projection each)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)  # d_model -> head_dim
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)  # d_model -> head_dim

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Multiple Q projections (one per head)
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)

        # Single K and V projections (shared across all heads)
        K = self.k_proj(x)  # (batch, seq_len, head_dim)
        V = self.v_proj(x)  # (batch, seq_len, head_dim)

        # Expand K and V to match Q's head dimension
        K = K.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        V = V.unsqueeze(1)  # (batch, 1, seq_len, head_dim)

        # Scaled dot-product attention (K and V broadcast across heads)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

def compare_mha_vs_mqa():
    d_model = 512
    n_heads = 8
    seq_len = 100
    batch_size = 2

    print("=" * 80)
    print("MULTI-HEAD ATTENTION (MHA) vs MULTI-QUERY ATTENTION (MQA)")
    print("=" * 80)

    mha = MultiHeadAttention(d_model, n_heads)
    mqa = MultiQueryAttention(d_model, n_heads)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")
    print(f"Model config: d_model={d_model}, n_heads={n_heads}, head_dim={d_model//n_heads}")

    # Count parameters
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())

    print(f"\nPARAMETER COUNT:")
    print(f"MHA: {mha_params:,} parameters")
    print(f"MQA: {mqa_params:,} parameters")
    print(f"MQA saves: {mha_params - mqa_params:,} parameters ({((mha_params - mqa_params) / mha_params * 100):.1f}% reduction)")

    # Analyze projection dimensions
    print(f"\nPROJECTION DIMENSIONS:")
    print(f"MHA:")
    print(f"  Q projection: {d_model} -> {d_model} (separate for each head)")
    print(f"  K projection: {d_model} -> {d_model} (separate for each head)")
    print(f"  V projection: {d_model} -> {d_model} (separate for each head)")
    print(f"MQA:")
    print(f"  Q projection: {d_model} -> {d_model} (separate for each head)")
    print(f"  K projection: {d_model} -> {d_model//n_heads} (shared across heads)")
    print(f"  V projection: {d_model} -> {d_model//n_heads} (shared across heads)")

    # Memory usage during inference (focusing on KV cache)
    head_dim = d_model // n_heads
    mha_kv_cache_size = 2 * seq_len * n_heads * head_dim  # K and V for each head
    mqa_kv_cache_size = 2 * seq_len * head_dim  # Single K and V shared

    print(f"\nKV CACHE MEMORY (for sequence length {seq_len}):")
    print(f"MHA KV cache: {mha_kv_cache_size:,} elements")
    print(f"MQA KV cache: {mqa_kv_cache_size:,} elements")
    print(f"MQA saves: {mha_kv_cache_size - mqa_kv_cache_size:,} elements ({((mha_kv_cache_size - mqa_kv_cache_size) / mha_kv_cache_size * 100):.1f}% reduction)")

    # Forward pass
    print(f"\nFORWARD PASS:")
    mha_out = mha(x)
    mqa_out = mqa(x)
    print(f"MHA output shape: {mha_out.shape}")
    print(f"MQA output shape: {mqa_out.shape}")

    print(f"\nKEY DIFFERENCES:")
    print(f"📊 PARAMETER EFFICIENCY:")
    print(f"   • MHA: Each head has separate K,V projections")
    print(f"   • MQA: All heads share single K,V projections")
    print(f"   • Result: MQA uses ~25% fewer parameters")

    print(f"\n💾 MEMORY EFFICIENCY:")
    print(f"   • MHA: KV cache grows with number of heads")
    print(f"   • MQA: KV cache size independent of number of heads")
    print(f"   • Result: Significant memory savings during inference")

    print(f"\n⚡ COMPUTATIONAL DIFFERENCES:")
    print(f"   • MHA: n_heads separate K,V computations")
    print(f"   • MQA: Single K,V computation, broadcasted to heads")
    print(f"   • Result: Faster K,V projection, same attention computation")

    print(f"\n🎯 WHEN TO USE:")
    print(f"   • MHA: Maximum model capacity, less memory constrained")
    print(f"   • MQA: Large-scale inference, memory-constrained environments")
    print(f"   • MQA particularly beneficial for long sequences and large batch sizes")

if __name__ == "__main__":
    compare_mha_vs_mqa()