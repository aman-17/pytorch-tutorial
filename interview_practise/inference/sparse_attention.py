"""
Sparse Attention Patterns in Multi-Head Attention

Sparse attention reduces O(n²) complexity by only computing attention
for selected positions, using different sparsity patterns:

1. Local Attention - only attend to nearby positions
2. Strided Attention - attend to every k-th position
3. Random Attention - attend to randomly sampled positions
4. Block-Local Attention - attend within fixed-size blocks

Each pattern trades off between computational efficiency and modeling capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

class DenseAttention(nn.Module):
    """Standard dense attention - attends to all positions O(n²)"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch, seq_len, d_model)
        seq_len = q.size(1)

        # Compute all pairwise attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n, n)

        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

class LocalSparseAttention(nn.Module):
    """Local sparse attention - only attend to nearby positions

    Each position i attends to positions [i-window//2, i+window//2]
    Reduces complexity from O(n²) to O(n × window_size)
    """

    def __init__(self, d_model, window_size=64):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(d_model)

    def create_local_mask(self, seq_len, device):
        """Create mask for local attention pattern"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = True

        return mask

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.shape
        device = q.device

        # Create local attention mask
        local_mask = self.create_local_mask(seq_len, device)

        # Compute scores only for local positions
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores.masked_fill_(~local_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights, local_mask

class StridedSparseAttention(nn.Module):
    """Strided sparse attention - attend to every k-th position

    Each position attends to positions [0, stride, 2*stride, ...]
    Good for capturing long-range dependencies with fixed intervals
    """

    def __init__(self, d_model, stride=8):
        super().__init__()
        self.d_model = d_model
        self.stride = stride
        self.scale = 1.0 / math.sqrt(d_model)

    def create_strided_mask(self, seq_len, device):
        """Create mask for strided attention pattern"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            # Attend to positions at multiples of stride
            for j in range(0, seq_len, self.stride):
                if j <= i:  # Causal constraint
                    mask[i, j] = True

        return mask

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.shape
        device = q.device

        # Create strided attention mask
        strided_mask = self.create_strided_mask(seq_len, device)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores.masked_fill_(~strided_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights, strided_mask

class RandomSparseAttention(nn.Module):
    """Random sparse attention - attend to randomly sampled positions

    Each position attends to a fixed number of randomly selected positions
    Provides good coverage while maintaining sparsity
    """

    def __init__(self, d_model, num_random=32, seed=42):
        super().__init__()
        self.d_model = d_model
        self.num_random = num_random
        self.scale = 1.0 / math.sqrt(d_model)
        self.seed = seed

    def create_random_mask(self, seq_len, device):
        """Create mask for random attention pattern"""
        torch.manual_seed(self.seed)  # For reproducibility
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            # Always attend to self
            mask[i, i] = True

            # Randomly sample positions to attend to (causal)
            available_positions = list(range(i))  # Only past positions
            if len(available_positions) > 0:
                num_to_sample = min(self.num_random - 1, len(available_positions))
                sampled = torch.randperm(len(available_positions))[:num_to_sample]
                for idx in sampled:
                    mask[i, available_positions[idx]] = True

        return mask

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.shape
        device = q.device

        random_mask = self.create_random_mask(seq_len, device)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores.masked_fill_(~random_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights, random_mask

class BlockLocalAttention(nn.Module):
    """Block-local sparse attention - attend within fixed blocks

    Divides sequence into blocks, attention only within each block
    Very efficient but limited cross-block interactions
    """

    def __init__(self, d_model, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(d_model)

    def create_block_mask(self, seq_len, device):
        """Create mask for block-local attention"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        for block in range(num_blocks):
            start = block * self.block_size
            end = min((block + 1) * self.block_size, seq_len)

            # Attention within block only
            mask[start:end, start:end] = True

        return mask

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.shape
        device = q.device

        block_mask = self.create_block_mask(seq_len, device)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores.masked_fill_(~block_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights, block_mask

def visualize_attention_patterns():
    """Visualize different sparse attention patterns"""
    seq_len = 128
    d_model = 64
    device = torch.device('cpu')

    # Create dummy inputs
    q = torch.randn(1, seq_len, d_model)
    k = torch.randn(1, seq_len, d_model)
    v = torch.randn(1, seq_len, d_model)

    # Initialize attention mechanisms
    local_attn = LocalSparseAttention(d_model, window_size=16)
    strided_attn = StridedSparseAttention(d_model, stride=8)
    random_attn = RandomSparseAttention(d_model, num_random=16)
    block_attn = BlockLocalAttention(d_model, block_size=32)

    # Get attention patterns
    patterns = {}

    _, _, local_mask = local_attn(q, k, v)
    patterns['Local (window=16)'] = local_mask

    _, _, strided_mask = strided_attn(q, k, v)
    patterns['Strided (stride=8)'] = strided_mask

    _, _, random_mask = random_attn(q, k, v)
    patterns['Random (k=16)'] = random_mask

    _, _, block_mask = block_attn(q, k, v)
    patterns['Block-Local (block=32)'] = block_mask

    # Plot patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, mask) in enumerate(patterns.items()):
        ax = axes[idx]
        ax.imshow(mask.float().numpy(), cmap='Blues', origin='upper')
        ax.set_title(f'{name}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

        # Calculate sparsity
        sparsity = (mask.sum().item() / mask.numel()) * 100
        ax.text(0.02, 0.98, f'Sparsity: {sparsity:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/Users/amanr/programs/pytorch-tutorial/sparse_attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()

    return patterns

def compare_sparse_attention():
    """Compare computational complexity and performance of sparse attention"""
    seq_len = 512
    d_model = 128
    batch_size = 1

    # Create inputs
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    print("="*80)
    print("SPARSE ATTENTION COMPARISON")
    print("="*80)
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print()

    # Dense attention
    dense_attn = DenseAttention(d_model)
    dense_out, dense_weights = dense_attn(q, k, v)
    dense_ops = seq_len * seq_len  # Number of attention computations

    print(f"📊 DENSE ATTENTION:")
    print(f"   Attention operations: {dense_ops:,}")
    print(f"   Memory: O(n²) = {seq_len**2:,} elements")
    print(f"   Complexity: O(n²)")
    print()

    # Sparse attention variants
    sparse_variants = [
        ("Local (window=64)", LocalSparseAttention(d_model, 64)),
        ("Strided (stride=16)", StridedSparseAttention(d_model, 16)),
        ("Random (k=64)", RandomSparseAttention(d_model, 64)),
        ("Block-Local (block=128)", BlockLocalAttention(d_model, 128))
    ]

    for name, attention in sparse_variants:
        if hasattr(attention, 'create_local_mask'):
            mask = attention.create_local_mask(seq_len, q.device)
        elif hasattr(attention, 'create_strided_mask'):
            mask = attention.create_strided_mask(seq_len, q.device)
        elif hasattr(attention, 'create_random_mask'):
            mask = attention.create_random_mask(seq_len, q.device)
        elif hasattr(attention, 'create_block_mask'):
            mask = attention.create_block_mask(seq_len, q.device)

        sparse_ops = mask.sum().item()
        sparsity = (sparse_ops / dense_ops) * 100
        speedup = dense_ops / sparse_ops

        print(f"📈 {name.upper()}:")
        print(f"   Attention operations: {int(sparse_ops):,}")
        print(f"   Sparsity: {sparsity:.1f}% of dense")
        print(f"   Theoretical speedup: {speedup:.1f}x")
        print(f"   Memory reduction: {dense_ops/sparse_ops:.1f}x")
        print()

    print("🎯 WHEN TO USE EACH PATTERN:")
    print()
    print("LOCAL ATTENTION:")
    print("   • Natural language: nearby words are most relevant")
    print("   • Time series: recent values matter most")
    print("   • Image patches: local spatial relationships")
    print()

    print("STRIDED ATTENTION:")
    print("   • Long sequences with periodic patterns")
    print("   • Capturing long-range dependencies at regular intervals")
    print("   • Complementary to local attention")
    print()

    print("RANDOM ATTENTION:")
    print("   • When you need global context but can't afford dense attention")
    print("   • Good empirical performance with proper sampling")
    print("   • Provides theoretical guarantees on coverage")
    print()

    print("BLOCK-LOCAL ATTENTION:")
    print("   • Very efficient for long sequences")
    print("   • When interactions within blocks are most important")
    print("   • Can be combined with cross-block attention")

    return sparse_variants

if __name__ == "__main__":
    print("Comparing sparse attention patterns...")
    compare_sparse_attention()
    print("\nGenerating visualization...")
    visualize_attention_patterns()