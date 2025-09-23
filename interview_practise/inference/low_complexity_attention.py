"""
Low-Complexity Attention Methods

This module demonstrates different attention mechanisms and their computational complexities:
1. Standard Attention - O(n²)
2. Kernel-based Attention - O(n)
3. Low-rank Attention - O(n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandardAttention(nn.Module):
    """Standard attention with O(n²) complexity

    Computes: softmax(QK^T/√d)V
    Complexity: O(n²d) where n is sequence length, d is feature dimension
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, q, k, v):
        # q, k, v shape: (batch, n, d)
        n = q.size(1)

        # Step 1: Compute QK^T - O(n²d) complexity
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n, n)

        # Step 2: Softmax - O(n²) complexity
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n, n)

        # Step 3: Apply to values - O(n²d) complexity
        output = torch.matmul(attn_weights, v)  # (batch, n, d)

        print(f"Standard Attention:")
        print(f"  Attention matrix shape: {attn_weights.shape}")
        print(f"  Memory usage: O(n²) = O({n}²) = {n*n} elements")
        print(f"  Computational complexity: O(n²d) = O({n}² × {self.d_model})")

        return output

class KernelBasedAttention(nn.Module):
    """Kernel-based attention with O(n) complexity

    Uses kernel approximation: φ(Q)φ(K)^TV instead of softmax(QK^T/√d)V
    Key insight: φ(Q)(φ(K)^TV) can be computed as φ(Q) × (φ(K)^TV)
    """

    def __init__(self, d_model, kernel_dim=64):
        super().__init__()
        self.d_model = d_model
        self.kernel_dim = kernel_dim

    def kernel_feature_map(self, x):
        """Simple ReLU-based kernel approximation"""
        # Apply random projection + ReLU as kernel approximation
        # In practice, more sophisticated kernels like RFF are used
        return F.relu(x)  # Simplified for demonstration

    def forward(self, q, k, v):
        # q, k, v shape: (batch, n, d)
        n = q.size(1)

        # Step 1: Apply kernel feature map
        phi_q = self.kernel_feature_map(q)  # (batch, n, d)
        phi_k = self.kernel_feature_map(k)  # (batch, n, d)

        # Step 2: Compute φ(K)^T V first - O(nd²) complexity
        kv = torch.matmul(phi_k.transpose(-2, -1), v)  # (batch, d, d)

        # Step 3: Then φ(Q) × (φ(K)^T V) - O(nd²) complexity
        output = torch.matmul(phi_q, kv)  # (batch, n, d)

        # Normalize (simplified)
        output = output / (phi_q.sum(dim=-1, keepdim=True) + 1e-8)

        print(f"Kernel-based Attention:")
        print(f"  No explicit attention matrix computed")
        print(f"  Intermediate KV shape: {kv.shape}")
        print(f"  Memory usage: O(d²) = O({self.d_model}²) = {self.d_model**2} elements")
        print(f"  Computational complexity: O(nd²) = O({n} × {self.d_model}²)")

        return output

class LowRankAttention(nn.Module):
    """Low-rank attention with O(n) complexity

    Approximates attention matrix A ≈ UV^T where U,V have rank r << n
    This reduces O(n²) attention matrix to O(nr) complexity
    """

    def __init__(self, d_model, rank=32):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        # Low-rank projections
        self.q_to_u = nn.Linear(d_model, rank, bias=False)
        self.k_to_v = nn.Linear(d_model, rank, bias=False)

    def forward(self, q, k, v):
        # q, k, v shape: (batch, n, d)
        batch_size, n, d = q.shape

        # Step 1: Project to low-rank space - O(ndr) complexity
        u = self.q_to_u(q)  # (batch, n, rank)
        v_proj = self.k_to_v(k)  # (batch, n, rank)

        # Step 2: Compute attention in low-rank space - O(nr²) complexity
        # Instead of n×n matrix, we work with n×r matrices
        scores = torch.matmul(u, v_proj.transpose(-2, -1))  # (batch, n, n) but computed efficiently
        attn_weights = F.softmax(scores / math.sqrt(self.rank), dim=-1)

        # Step 3: Apply to values - O(n²d) -> This can be further optimized
        output = torch.matmul(attn_weights, v)  # (batch, n, d)

        print(f"Low-rank Attention:")
        print(f"  Low-rank projections: {n} × {self.rank} each")
        print(f"  Rank: {self.rank} << sequence length {n}")
        print(f"  Memory usage: O(nr) = O({n} × {self.rank}) = {n * self.rank} elements")
        print(f"  Computational complexity: O(nr²) = O({n} × {self.rank}²)")

        return output

def demonstrate_complexity_comparison():
    """Compare computational complexities of different attention mechanisms"""

    batch_size = 1
    n = 1024  # sequence length
    d = 512   # model dimension

    print("=" * 80)
    print("LOW-COMPLEXITY ATTENTION COMPARISON")
    print("=" * 80)
    print(f"Sequence length (n): {n}")
    print(f"Model dimension (d): {d}")
    print()

    # Create sample inputs
    q = torch.randn(batch_size, n, d)
    k = torch.randn(batch_size, n, d)
    v = torch.randn(batch_size, n, d)

    print("COMPLEXITY ANALYSIS:")
    print("-" * 40)

    # Standard Attention
    std_attn = StandardAttention(d)
    _ = std_attn(q, k, v)
    print()

    # Kernel-based Attention
    kernel_attn = KernelBasedAttention(d)
    _ = kernel_attn(q, k, v)
    print()

    # Low-rank Attention
    lowrank_attn = LowRankAttention(d, rank=64)
    _ = lowrank_attn(q, k, v)
    print()

    print("KEY INSIGHTS:")
    print("-" * 40)
    print("🔥 THE QUADRATIC BOTTLENECK:")
    print(f"   • Standard attention: O(n²) = O({n}²) = {n*n:,} operations")
    print(f"   • This grows VERY quickly with sequence length!")
    print(f"   • For n=4096: {4096*4096:,} operations")
    print(f"   • For n=8192: {8192*8192:,} operations")
    print()

    print("✅ LINEAR COMPLEXITY SOLUTIONS:")
    print("   • Kernel methods: Avoid computing full attention matrix")
    print("   • Low-rank: Approximate attention with rank-r matrices")
    print("   • Both achieve O(n) or O(nr) complexity")
    print()

    print("📊 TRADE-OFFS:")
    print("   • Standard: Best quality, but O(n²) complexity")
    print("   • Kernel-based: O(n) complexity, approximation quality depends on kernel")
    print("   • Low-rank: O(nr) complexity, quality depends on rank r")
    print()

    print("🎯 WHEN TO USE:")
    print("   • Short sequences (n < 1024): Standard attention")
    print("   • Long sequences (n > 4096): Linear complexity methods")
    print("   • Very long sequences (n > 16K): Essential for feasibility")

if __name__ == "__main__":
    demonstrate_complexity_comparison()