"""
Speculative Decoding Implementation

Speculative decoding accelerates autoregressive generation by:
1. Using a small "draft" model to generate multiple tokens speculatively
2. Using the large "target" model to verify these tokens in parallel
3. Accepting correct tokens, rejecting and resampling incorrect ones

Key insight: Even if only some speculated tokens are accepted,
we still get speedup by processing multiple tokens in one target model forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import List, Tuple

class SimpleLanguageModel(nn.Module):
    """Simple transformer-like model for demonstration"""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)  # Max sequence length

        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache=False) -> torch.Tensor:
        seq_len = input_ids.size(1)

        # Token + position embeddings
        token_emb = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)

        x = token_emb + pos_emb

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

class StandardDecoding:
    """Standard autoregressive decoding - one token at a time"""

    def __init__(self, model: nn.Module):
        self.model = model

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20,
                temperature: float = 1.0) -> Tuple[torch.Tensor, List[float]]:
        """Generate tokens one by one"""
        self.model.eval()

        generated_ids = input_ids.clone()
        generation_times = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                start_time = time.time()

                # Forward pass through entire sequence
                logits = self.model(generated_ids)  # (batch, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :] / temperature  # Last position

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                step_time = time.time() - start_time
                generation_times.append(step_time)

                print(f"Standard step {step+1}: Generated token {next_token.item()} "
                      f"(time: {step_time:.4f}s)")

        return generated_ids, generation_times

class SpeculativeDecoding:
    """Speculative decoding using draft model + target model"""

    def __init__(self, draft_model: nn.Module, target_model: nn.Module):
        self.draft_model = draft_model
        self.target_model = target_model

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20,
                speculation_length: int = 4, temperature: float = 1.0) -> Tuple[torch.Tensor, List[int]]:
        """Generate using speculative decoding"""
        self.draft_model.eval()
        self.target_model.eval()

        generated_ids = input_ids.clone()
        accepted_counts = []
        step = 0

        with torch.no_grad():
            while step < max_new_tokens:
                print(f"\n--- Speculative Step {step+1} ---")

                # Phase 1: Draft model generates K speculative tokens
                draft_tokens = self._draft_phase(generated_ids, speculation_length, temperature)
                print(f"Draft tokens: {draft_tokens}")

                # Phase 2: Target model verifies all tokens in parallel
                accepted_tokens = self._verification_phase(
                    generated_ids, draft_tokens, temperature
                )

                print(f"Accepted tokens: {accepted_tokens}")
                print(f"Acceptance rate: {len(accepted_tokens)}/{len(draft_tokens)} "
                      f"({len(accepted_tokens)/len(draft_tokens)*100:.1f}%)")

                # Append accepted tokens
                if accepted_tokens:
                    new_tokens = torch.tensor([accepted_tokens], device=generated_ids.device)
                    generated_ids = torch.cat([generated_ids, new_tokens], dim=1)

                accepted_counts.append(len(accepted_tokens))
                step += len(accepted_tokens)

                # If no tokens accepted, we still make progress by sampling 1 token
                if len(accepted_tokens) == 0:
                    logits = self.target_model(generated_ids)
                    next_token_logits = logits[:, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    step += 1
                    print(f"Fallback: Generated token {next_token.item()}")

        return generated_ids, accepted_counts

    def _draft_phase(self, input_ids: torch.Tensor, speculation_length: int,
                    temperature: float) -> List[int]:
        """Phase 1: Draft model generates speculative tokens"""
        draft_sequence = input_ids.clone()
        draft_tokens = []

        for i in range(speculation_length):
            logits = self.draft_model(draft_sequence)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(next_token.item())
            draft_sequence = torch.cat([draft_sequence, next_token], dim=1)

        return draft_tokens

    def _verification_phase(self, input_ids: torch.Tensor, draft_tokens: List[int],
                          temperature: float) -> List[int]:
        """Phase 2: Target model verifies draft tokens"""
        # Create sequence with all draft tokens
        draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
        extended_sequence = torch.cat([input_ids, draft_tensor], dim=1)

        # Get target model probabilities for all positions in one forward pass
        logits = self.target_model(extended_sequence)  # (1, seq_len + K, vocab_size)

        accepted_tokens = []
        original_length = input_ids.size(1)

        # Verify each draft token
        for i, draft_token in enumerate(draft_tokens):
            position = original_length + i
            target_logits = logits[:, position - 1, :] / temperature  # Position before current
            target_probs = F.softmax(target_logits, dim=-1)

            # Acceptance probability: min(1, p_target / p_draft)
            # For simplicity, we'll use a threshold-based acceptance
            target_prob = target_probs[0, draft_token].item()

            # Simple acceptance rule: accept if target prob > threshold
            acceptance_threshold = 0.1  # Tunable parameter
            if target_prob > acceptance_threshold:
                accepted_tokens.append(draft_token)
                print(f"  Token {draft_token} accepted (p={target_prob:.4f})")
            else:
                print(f"  Token {draft_token} rejected (p={target_prob:.4f})")
                break  # Reject this and all subsequent tokens

        return accepted_tokens

def demonstrate_speculative_decoding():
    """Compare standard vs speculative decoding"""

    print("="*80)
    print("SPECULATIVE DECODING DEMONSTRATION")
    print("="*80)

    # Model configurations
    vocab_size = 1000

    # Draft model: smaller, faster
    draft_config = {"d_model": 128, "n_layers": 2, "n_heads": 4}
    draft_model = SimpleLanguageModel(vocab_size, **draft_config)

    # Target model: larger, slower
    target_config = {"d_model": 256, "n_layers": 4, "n_heads": 8}
    target_model = SimpleLanguageModel(vocab_size, **target_config)

    print(f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}")
    print(f"Target model parameters: {sum(p.numel() for p in target_model.parameters()):,}")
    print()

    # Input prompt
    input_ids = torch.randint(0, vocab_size, (1, 10))  # Batch size 1, seq len 10
    max_new_tokens = 12

    print(f"Input sequence: {input_ids[0].tolist()}")
    print(f"Generating {max_new_tokens} new tokens...")
    print()

    # Standard decoding
    print("🐌 STANDARD DECODING:")
    print("-" * 40)
    start_time = time.time()
    standard_decoder = StandardDecoding(target_model)
    standard_result, standard_times = standard_decoder.generate(input_ids, max_new_tokens)
    standard_total_time = time.time() - start_time

    print(f"Generated sequence: {standard_result[0].tolist()}")
    print(f"Total time: {standard_total_time:.4f}s")
    print(f"Average time per token: {np.mean(standard_times):.4f}s")
    print()

    # Speculative decoding
    print("🚀 SPECULATIVE DECODING:")
    print("-" * 40)
    start_time = time.time()
    speculative_decoder = SpeculativeDecoding(draft_model, target_model)
    spec_result, accepted_counts = speculative_decoder.generate(
        input_ids, max_new_tokens, speculation_length=4
    )
    spec_total_time = time.time() - start_time

    print(f"Generated sequence: {spec_result[0].tolist()}")
    print(f"Total time: {spec_total_time:.4f}s")
    print(f"Speedup: {standard_total_time / spec_total_time:.2f}x")
    print()

    # Analysis
    total_accepted = sum(accepted_counts)
    total_draft_attempts = len(accepted_counts) * 4  # speculation_length
    acceptance_rate = total_accepted / total_draft_attempts if total_draft_attempts > 0 else 0

    print("📊 ANALYSIS:")
    print("-" * 40)
    print(f"Total tokens accepted: {total_accepted}")
    print(f"Total speculation attempts: {total_draft_attempts}")
    print(f"Average acceptance rate: {acceptance_rate:.2%}")
    print(f"Tokens per target model call: {total_accepted / len(accepted_counts):.2f}")
    print()

    print("🎯 KEY INSIGHTS:")
    print("-" * 40)
    print("✅ BENEFITS:")
    print("   • Parallelizes token verification in target model")
    print("   • Even 50% acceptance rate gives significant speedup")
    print("   • Scales with difference between draft/target model speeds")
    print()
    print("⚠️  CONSIDERATIONS:")
    print("   • Requires good draft model (fast + reasonable quality)")
    print("   • Memory overhead for storing draft tokens")
    print("   • Acceptance rate depends on draft model quality")
    print("   • More complex implementation than standard decoding")

if __name__ == "__main__":
    demonstrate_speculative_decoding()