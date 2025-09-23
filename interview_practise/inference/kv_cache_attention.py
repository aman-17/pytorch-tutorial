import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer('k_cache', torch.zeros(1, max_seq_len, n_heads, self.head_dim))
        self.register_buffer('v_cache', torch.zeros(1, max_seq_len, n_heads, self.head_dim))

    def forward(self, x, position=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        if use_cache and position is not None:
            self.k_cache[:, position:position+seq_len] = k
            self.v_cache[:, position:position+seq_len] = v

            k_full = self.k_cache[:, :position+seq_len]
            v_full = self.v_cache[:, :position+seq_len]
        else:
            k_full = k
            v_full = v
            if use_cache:
                self.k_cache[:, :seq_len] = k
                self.v_cache[:, :seq_len] = v

        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if use_cache and position is not None:
            causal_mask = torch.triu(torch.ones(seq_len, position+seq_len), diagonal=position+1)
            scores.masked_fill_(causal_mask.bool(), float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            scores.masked_fill_(causal_mask.bool(), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_full)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

    def clear_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

def demonstrate_prefilling_vs_decoding():
    d_model = 512
    n_heads = 8
    seq_len = 10
    vocab_size = 1000

    attention = AttentionWithKVCache(d_model, n_heads)
    embedding = nn.Embedding(vocab_size, d_model)

    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    print("=" * 60)
    print("PREFILLING STAGE")
    print("=" * 60)
    print(f"Input sequence length: {seq_len}")
    print("Processing entire sequence at once...")

    attention.clear_cache()

    x = embedding(input_ids)
    output_prefill = attention(x, use_cache=True)

    print(f"Output shape: {output_prefill.shape}")
    print(f"KV cache shapes after prefilling:")
    print(f"  K cache: {attention.k_cache[:, :seq_len].shape}")
    print(f"  V cache: {attention.v_cache[:, :seq_len].shape}")

    print("\n" + "=" * 60)
    print("DECODING STAGE")
    print("=" * 60)
    print("Generating tokens one by one using cached K,V...")

    current_pos = seq_len
    generated_tokens = []

    for step in range(3):
        next_token_id = torch.randint(0, vocab_size, (1, 1))
        generated_tokens.append(next_token_id.item())

        print(f"\nDecoding step {step + 1}:")
        print(f"  Generating token at position {current_pos}")
        print(f"  New token ID: {next_token_id.item()}")

        x_new = embedding(next_token_id)

        output_decode = attention(x_new, position=current_pos, use_cache=True)

        print(f"  Using cached K,V from positions 0-{current_pos-1}")
        print(f"  Computing attention for position {current_pos} with all previous positions")
        print(f"  Output shape: {output_decode.shape}")

        current_pos += 1

    print(f"\nGenerated token IDs: {generated_tokens}")
    print(f"Final KV cache usage: positions 0-{current_pos-1}")

    print("\n" + "=" * 60)
    print("KEY DIFFERENCES")
    print("=" * 60)
    print("PREFILLING:")
    print("  - Processes entire input sequence simultaneously")
    print("  - Computes K,V for all positions and caches them")
    print("  - Attention matrix is (seq_len x seq_len)")
    print("  - Efficient parallel computation")

    print("\nDECODING:")
    print("  - Processes one new token at a time")
    print("  - Reuses cached K,V from previous positions")
    print("  - Only computes K,V for the new position")
    print("  - Attention matrix is (1 x current_seq_len)")
    print("  - Memory efficient for long sequences")

if __name__ == "__main__":
    demonstrate_prefilling_vs_decoding()