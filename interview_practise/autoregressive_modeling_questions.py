"""
PyTorch Autoregressive Modeling Interview Questions - Practice Set
Focus: Autoregressive models, sequence generation, RNNs, transformers, causal attention

These questions cover common PyTorch interview topics around autoregressive modeling,
including RNNs, LSTMs, GRUs, transformer decoders, and sequence generation techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import math


"""
================================================================================
QUESTION 1: Implement a Vanilla RNN Cell from Scratch
================================================================================

Scenario: You need to implement a basic RNN cell without using nn.RNN to demonstrate
understanding of the underlying mechanics.

Task: Implement an RNN cell that:
1. Takes input and previous hidden state
2. Computes new hidden state using tanh activation
3. Supports proper weight initialization
4. Can be used for sequence processing

Key Concepts:
- RNN forward pass: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
- Weight initialization strategies
- Hidden state management
- Gradient flow considerations
"""

class VanillaRNNCell(nn.Module):
    """
    Basic RNN cell implementation.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        bias: Whether to include bias terms
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(VanillaRNNCell, self).__init__()

        # TODO: Initialize parameters
        # 1. W_ih: input to hidden weights (hidden_size, input_size)
        # 2. W_hh: hidden to hidden weights (hidden_size, hidden_size)
        # 3. b_ih, b_hh: bias terms (hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Initialize weights with Xavier uniform initialization
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            self.b_ih = nn.Parameter(torch.empty(hidden_size))
            self.b_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with proper scaling"""
        # TODO: Implement Xavier initialization
        # std = sqrt(1 / hidden_size) for weights

        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of RNN cell.

        Args:
            x: Input tensor (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)

        Returns:
            h_next: Next hidden state (batch_size, hidden_size)
        """
        # TODO: Implement forward pass
        # 1. Initialize hidden state if None
        # 2. Compute h_t = tanh(W_ih @ x + W_hh @ h_{t-1} + b)

        batch_size = x.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        # Compute hidden state
        h_next = torch.mm(x, self.W_ih.t()) + torch.mm(h_prev, self.W_hh.t())

        if self.use_bias:
            h_next = h_next + self.b_ih + self.b_hh

        h_next = torch.tanh(h_next)

        return h_next


class VanillaRNN(nn.Module):
    """
    Multi-layer RNN using custom RNN cells.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        num_layers: Number of RNN layers
        bias: Whether to include bias
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True):
        super(VanillaRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create RNN cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(VanillaRNNCell(layer_input_size, hidden_size, bias))

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through RNN.

        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            h_0: Initial hidden states (num_layers, batch_size, hidden_size)

        Returns:
            output: Output sequence (batch_size, seq_len, hidden_size)
            h_n: Final hidden states (num_layers, batch_size, hidden_size)
        """
        # TODO: Implement multi-layer RNN forward pass
        # 1. Process each timestep through all layers
        # 2. Track hidden states for each layer
        # 3. Return outputs and final hidden states

        batch_size, seq_len, _ = x.size()

        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                            device=x.device, dtype=x.dtype)

        h_current = h_0
        outputs = []

        # Process each timestep
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Process through each layer
            for layer in range(self.num_layers):
                h_current[layer] = self.cells[layer](x_t, h_current[layer])
                x_t = h_current[layer]  # Output of layer becomes input to next layer

            outputs.append(x_t)

        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return output, h_current


"""
================================================================================
QUESTION 2: Implement LSTM Cell with Detailed Gate Mechanics
================================================================================

Scenario: Implement an LSTM cell to understand how gates control information flow
and solve the vanishing gradient problem.

Task: Implement LSTM with:
1. Input, forget, output gates
2. Cell state and hidden state
3. Proper gate activation functions
4. Weight initialization for stability

Key Concepts:
- Gate mechanisms: forget, input, output
- Cell state vs hidden state
- Sigmoid and tanh activations
- Long-term dependency modeling
"""

class LSTMCell(nn.Module):
    """
    LSTM cell implementation.

    Gates:
    - Forget gate: f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
    - Cell candidate: g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)
    - Output gate: o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)

    Updates:
    - Cell state: c_t = f_t * c_{t-1} + i_t * g_t
    - Hidden state: h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(LSTMCell, self).__init__()

        # TODO: Initialize LSTM parameters
        # 1. Four sets of weights for i, f, g, o gates
        # 2. Each gate has weights for input and hidden state
        # 3. Bias terms for each gate

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Combined weight matrix for all gates
        # [W_ii, W_if, W_ig, W_io] concatenated
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))

        if bias:
            self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with proper scaling"""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LSTM cell.

        Args:
            x: Input tensor (batch_size, input_size)
            state: Tuple of (h_prev, c_prev) or None

        Returns:
            h_next: Next hidden state (batch_size, hidden_size)
            c_next: Next cell state (batch_size, hidden_size)
        """
        # TODO: Implement LSTM forward pass
        # 1. Initialize states if None
        # 2. Compute all four gates
        # 3. Update cell state
        # 4. Compute output hidden state

        batch_size = x.size(0)

        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        # Compute all gates at once
        gates = torch.mm(x, self.W_ih.t()) + torch.mm(h_prev, self.W_hh.t())
        if self.use_bias:
            gates = gates + self.b_ih + self.b_hh

        # Split gates
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

        # Apply activations
        i_t = torch.sigmoid(i_t)  # Input gate
        f_t = torch.sigmoid(f_t)  # Forget gate
        g_t = torch.tanh(g_t)     # Cell candidate
        o_t = torch.sigmoid(o_t)  # Output gate

        # Update cell state
        c_next = f_t * c_prev + i_t * g_t

        # Compute hidden state
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


"""
================================================================================
QUESTION 3: Implement Causal (Masked) Self-Attention for Autoregressive Models
================================================================================

Scenario: For autoregressive generation, you need attention that only looks at
previous tokens, not future ones.

Task: Implement causal self-attention with:
1. Proper masking to prevent attending to future positions
2. Scaled dot-product attention
3. Multi-head attention support
4. Efficient mask creation

Key Concepts:
- Causal masking (lower triangular mask)
- Attention mechanism for autoregressive models
- Preventing information leakage from future tokens
- Numerical stability with large negative values
"""

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism for autoregressive models.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(CausalSelfAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # TODO: Initialize projection matrices
        # 1. Q, K, V projections
        # 2. Output projection
        # 3. Dropout layer

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Register buffer for causal mask (not a parameter)
        self.register_buffer('mask', None)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future positions.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            mask: Causal mask (1, 1, seq_len, seq_len)
        """
        # TODO: Create lower triangular mask
        # 1. Use torch.tril to create triangular matrix
        # 2. Convert to mask with -inf for masked positions
        # 3. Return mask with proper shape for broadcasting

        # Create lower triangular matrix (1s on and below diagonal)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

        # Convert to attention mask (0 for positions to mask, 1 for positions to attend)
        # Then convert to additive mask (-inf for masked, 0 for unmasked)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        # Add dimensions for batch and heads
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        return mask

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attention_weights: (Optional) Attention weights
        """
        # TODO: Implement causal self-attention
        # 1. Project to Q, K, V
        # 2. Split into multiple heads
        # 3. Compute scaled dot-product attention with causal mask
        # 4. Concatenate heads and project

        batch_size, seq_len, d_model = x.size()

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Create or reuse causal mask
        if self.mask is None or self.mask.size(-1) < seq_len:
            self.mask = self.create_causal_mask(seq_len, x.device)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        scores = scores + self.mask[:, :, :seq_len, :seq_len]

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attn_output = torch.matmul(attention_weights, V)  # (batch, heads, seq, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq, heads, d_k)
        attn_output = attn_output.view(batch_size, seq_len, d_model)  # (batch, seq, d_model)

        # Final projection
        output = self.W_o(attn_output)

        if return_attention:
            return output, attention_weights
        return output


"""
================================================================================
QUESTION 4: Implement Autoregressive Sequence Generator with Sampling Strategies
================================================================================

Scenario: Given a trained autoregressive model, implement various sampling strategies
for text generation.

Task: Implement generation with:
1. Greedy decoding
2. Temperature sampling
3. Top-k sampling
4. Top-p (nucleus) sampling

Key Concepts:
- Different sampling strategies for generation
- Temperature control for diversity
- Top-k and top-p filtering
- Efficient sequential generation
"""

class AutoregressiveGenerator:
    """
    Text generation using various sampling strategies.

    Args:
        model: Autoregressive model (e.g., transformer decoder)
        vocab_size: Size of vocabulary
        max_length: Maximum generation length
    """

    def __init__(self, model: nn.Module, vocab_size: int, max_length: int = 100):
        self.model = model
        self.vocab_size = vocab_size
        self.max_length = max_length

    @torch.no_grad()
    def greedy_decode(self, context: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Greedy decoding: always select most likely next token.

        Args:
            context: Context tokens (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate

        Returns:
            generated: Generated sequence (batch_size, seq_len + max_new_tokens)
        """
        # TODO: Implement greedy decoding
        # 1. For each step, get model predictions
        # 2. Select token with highest probability
        # 3. Append to sequence and continue

        self.model.eval()
        generated = context

        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.model(generated)  # (batch_size, seq_len, vocab_size)

            # Take logits for last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Greedy selection
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def sample_with_temperature(self, context: torch.Tensor, max_new_tokens: int,
                               temperature: float = 1.0) -> torch.Tensor:
        """
        Sample with temperature scaling.

        Temperature controls randomness:
        - temperature < 1.0: more deterministic (sharper distribution)
        - temperature = 1.0: unchanged
        - temperature > 1.0: more random (flatter distribution)

        Args:
            context: Context tokens
            max_new_tokens: Number of tokens to generate
            temperature: Temperature parameter

        Returns:
            generated: Generated sequence
        """
        # TODO: Implement temperature sampling
        # 1. Divide logits by temperature
        # 2. Apply softmax
        # 3. Sample from resulting distribution

        self.model.eval()
        generated = context

        for _ in range(max_new_tokens):
            logits = self.model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def sample_top_k(self, context: torch.Tensor, max_new_tokens: int,
                     k: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """
        Top-k sampling: only sample from top k most likely tokens.

        Args:
            context: Context tokens
            max_new_tokens: Number of tokens to generate
            k: Number of top tokens to consider
            temperature: Temperature parameter

        Returns:
            generated: Generated sequence
        """
        # TODO: Implement top-k sampling
        # 1. Find top-k logits
        # 2. Mask out all other logits (set to -inf)
        # 3. Apply temperature and sample

        self.model.eval()
        generated = context

        for _ in range(max_new_tokens):
            logits = self.model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Get top-k values and indices
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)

            # Create mask for top-k
            next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits_filtered.scatter_(1, top_k_indices, top_k_logits)

            # Sample from top-k
            probs = F.softmax(next_token_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def sample_top_p(self, context: torch.Tensor, max_new_tokens: int,
                     p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """
        Top-p (nucleus) sampling: sample from smallest set of tokens whose
        cumulative probability exceeds p.

        Args:
            context: Context tokens
            max_new_tokens: Number of tokens to generate
            p: Cumulative probability threshold
            temperature: Temperature parameter

        Returns:
            generated: Generated sequence
        """
        # TODO: Implement top-p sampling
        # 1. Sort logits by probability
        # 2. Compute cumulative probabilities
        # 3. Find cutoff where cumulative prob exceeds p
        # 4. Mask and sample

        self.model.eval()
        generated = context

        for _ in range(max_new_tokens):
            logits = self.model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Sort probabilities in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False

            # Create mask in original order
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits_filtered = next_token_logits.clone()
            next_token_logits_filtered[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


"""
================================================================================
QUESTION 5: Implement Positional Encoding for Sequence Models
================================================================================

Scenario: Transformers don't have inherent position information. You need to add
positional encodings to give the model sequence order information.

Task: Implement:
1. Sinusoidal positional encoding (original Transformer paper)
2. Learnable positional embeddings
3. Relative positional encoding

Key Concepts:
- Why transformers need positional information
- Sinusoidal vs learnable encodings
- Relative vs absolute positions
- Generalization to longer sequences
"""

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(SinusoidalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # TODO: Compute sinusoidal positional encodings
        # 1. Create position indices [0, 1, 2, ..., max_len-1]
        # 2. Create dimension indices [0, 2, 4, ..., d_model-2]
        # 3. Compute encodings using sin/cos
        # 4. Register as buffer (not a trainable parameter)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term for scaling
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (saved in state_dict but not trained)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input embeddings (batch_size, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        # TODO: Add positional encoding to input
        # 1. Slice pe to match sequence length
        # 2. Add to input
        # 3. Apply dropout

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings (used in GPT, BERT, etc.).
    """

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super(LearnablePositionalEmbedding, self).__init__()

        # TODO: Create learnable positional embeddings
        # 1. Create embedding layer for positions
        # 2. Initialize with appropriate distribution

        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional embeddings to input.

        Args:
            x: Input embeddings (batch_size, seq_len, d_model)

        Returns:
            x with positional embeddings added
        """
        # TODO: Add positional embeddings
        # 1. Create position indices
        # 2. Look up embeddings
        # 3. Add to input

        batch_size, seq_len, d_model = x.size()

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Get positional embeddings
        pos_emb = self.pos_embedding(positions)

        # Add to input
        x = x + pos_emb
        return self.dropout(x)


"""
================================================================================
QUESTION 6: Implement Sequence-to-Sequence Model with Teacher Forcing
================================================================================

Scenario: For sequence-to-sequence tasks (translation, summarization), implement
a model with encoder-decoder architecture and teacher forcing during training.

Task: Implement:
1. Encoder RNN/LSTM
2. Decoder RNN/LSTM with attention
3. Teacher forcing during training
4. Autoregressive generation during inference

Key Concepts:
- Encoder-decoder architecture
- Teacher forcing vs autoregressive generation
- Attention mechanism for seq2seq
- Training vs inference mode differences
"""

class Seq2SeqEncoder(nn.Module):
    """
    RNN encoder for sequence-to-sequence model.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super(Seq2SeqEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Create encoder LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input sequence.

        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            outputs: Encoder outputs (batch_size, seq_len, hidden_size)
            state: Final hidden and cell states
        """
        outputs, state = self.lstm(x)
        return outputs, state


class Seq2SeqDecoder(nn.Module):
    """
    RNN decoder with attention for sequence-to-sequence model.
    """

    def __init__(self, output_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super(Seq2SeqDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # TODO: Create decoder components
        # 1. LSTM
        # 2. Attention mechanism
        # 3. Output projection

        self.lstm = nn.LSTM(output_size + hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Simple attention
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)

    def compute_attention(self, decoder_hidden: torch.Tensor,
                         encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: Current decoder hidden state (batch_size, hidden_size)
            encoder_outputs: All encoder outputs (batch_size, src_len, hidden_size)

        Returns:
            context: Attention-weighted context vector (batch_size, hidden_size)
        """
        # TODO: Implement attention mechanism
        # 1. Compute attention scores for each encoder output
        # 2. Apply softmax to get attention weights
        # 3. Compute weighted sum of encoder outputs

        batch_size, src_len, hidden_size = encoder_outputs.size()

        # Expand decoder hidden to match encoder output length
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        # Concatenate and compute scores
        combined = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
        scores = self.attention(combined).squeeze(2)  # (batch_size, src_len)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=1)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode one step.

        Args:
            x: Current input (batch_size, 1, output_size)
            state: Current LSTM state
            encoder_outputs: Encoder outputs for attention

        Returns:
            output: Predictions (batch_size, output_size)
            state: Updated LSTM state
        """
        # Get current hidden state for attention
        h_prev = state[0][-1]  # Last layer hidden state

        # Compute attention context
        context = self.compute_attention(h_prev, encoder_outputs)

        # Concatenate input with context
        lstm_input = torch.cat([x.squeeze(1), context], dim=1).unsqueeze(1)

        # LSTM step
        lstm_out, state = self.lstm(lstm_input, state)

        # Project to output vocabulary
        output = self.fc(lstm_out.squeeze(1))

        return output, state


class Seq2SeqModel(nn.Module):
    """
    Complete sequence-to-sequence model with attention.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super(Seq2SeqModel, self).__init__()

        self.encoder = Seq2SeqEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Seq2SeqDecoder(output_size, hidden_size, num_layers, dropout)
        self.output_size = output_size

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            src: Source sequence (batch_size, src_len, input_size)
            tgt: Target sequence (batch_size, tgt_len, output_size)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: Predictions (batch_size, tgt_len, output_size)
        """
        # TODO: Implement forward pass with teacher forcing
        # 1. Encode source sequence
        # 2. Initialize decoder state from encoder
        # 3. For each decoder step:
        #    - Use teacher forcing with probability teacher_forcing_ratio
        #    - Otherwise use model's own prediction

        batch_size, tgt_len, _ = tgt.size()

        # Encode
        encoder_outputs, encoder_state = self.encoder(src)

        # Initialize decoder state
        decoder_state = encoder_state

        # Storage for outputs
        outputs = []

        # First input is start token (first token of target)
        decoder_input = tgt[:, 0:1, :]

        for t in range(tgt_len):
            # Decode step
            output, decoder_state = self.decoder(decoder_input, decoder_state, encoder_outputs)
            outputs.append(output)

            # Teacher forcing: use ground truth as next input
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio

            if use_teacher_forcing and t < tgt_len - 1:
                decoder_input = tgt[:, t+1:t+2, :]
            else:
                # Use model's own prediction
                # Convert logits to one-hot or embedding
                decoder_input = F.one_hot(output.argmax(dim=-1), num_classes=self.output_size).float().unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)
        return outputs


"""
================================================================================
TEST CASES AND EXAMPLES
================================================================================
"""

def test_question1_vanilla_rnn():
    """Test Vanilla RNN Cell"""
    print("\n" + "="*80)
    print("QUESTION 1: Vanilla RNN Test")
    print("="*80)

    # Test RNN cell
    rnn_cell = VanillaRNNCell(input_size=10, hidden_size=20)
    x = torch.randn(32, 10)  # batch_size=32
    h = rnn_cell(x)

    print(f"RNN Cell:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {h.shape}")

    # Test multi-layer RNN
    rnn = VanillaRNN(input_size=10, hidden_size=20, num_layers=2)
    x_seq = torch.randn(32, 15, 10)  # batch_size=32, seq_len=15
    output, h_n = rnn(x_seq)

    print(f"\nMulti-layer RNN:")
    print(f"  Input shape: {x_seq.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Final hidden states shape: {h_n.shape}")

    print("\n" + "="*80)


def test_question2_lstm():
    """Test LSTM Cell"""
    print("\n" + "="*80)
    print("QUESTION 2: LSTM Test")
    print("="*80)

    lstm_cell = LSTMCell(input_size=10, hidden_size=20)
    x = torch.randn(32, 10)

    h, c = lstm_cell(x)

    print(f"LSTM Cell:")
    print(f"  Input shape: {x.shape}")
    print(f"  Hidden state shape: {h.shape}")
    print(f"  Cell state shape: {c.shape}")

    # Test sequential processing
    seq_len = 10
    h_states = []
    state = None

    for t in range(seq_len):
        x_t = torch.randn(32, 10)
        h, c = lstm_cell(x_t, state)
        state = (h, c)
        h_states.append(h)

    print(f"\nSequential processing:")
    print(f"  Processed {seq_len} timesteps")
    print(f"  Final hidden state shape: {h.shape}")

    print("\n" + "="*80)


def test_question3_causal_attention():
    """Test Causal Self-Attention"""
    print("\n" + "="*80)
    print("QUESTION 3: Causal Self-Attention Test")
    print("="*80)

    attention = CausalSelfAttention(d_model=128, n_heads=8)
    x = torch.randn(4, 20, 128)  # batch_size=4, seq_len=20, d_model=128

    output, attn_weights = attention(x, return_attention=True)

    print(f"Causal Self-Attention:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # Verify causal masking
    print(f"\nVerifying causal mask:")
    sample_attn = attn_weights[0, 0]  # First batch, first head
    print(f"  Attention weights (first head, first batch):")
    print(f"  Position 0 attends to: {(sample_attn[0] > 0).sum().item()} positions (should be 1)")
    print(f"  Position 5 attends to: {(sample_attn[5] > 0).sum().item()} positions (should be 6)")
    print(f"  Position 19 attends to: {(sample_attn[19] > 0).sum().item()} positions (should be 20)")

    print("\n" + "="*80)


def test_question4_generator():
    """Test Autoregressive Generator"""
    print("\n" + "="*80)
    print("QUESTION 4: Autoregressive Generator Test")
    print("="*80)

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            return self.fc(x)

    vocab_size = 100
    model = DummyModel(vocab_size=vocab_size, d_model=64)
    generator = AutoregressiveGenerator(model, vocab_size=vocab_size)

    context = torch.randint(0, vocab_size, (2, 10))  # batch_size=2, seq_len=10

    # Test greedy decoding
    greedy_output = generator.greedy_decode(context, max_new_tokens=5)
    print(f"Greedy decoding:")
    print(f"  Context shape: {context.shape}")
    print(f"  Generated shape: {greedy_output.shape}")

    # Test temperature sampling
    temp_output = generator.sample_with_temperature(context, max_new_tokens=5, temperature=0.8)
    print(f"\nTemperature sampling (T=0.8):")
    print(f"  Generated shape: {temp_output.shape}")

    # Test top-k sampling
    topk_output = generator.sample_top_k(context, max_new_tokens=5, k=10)
    print(f"\nTop-k sampling (k=10):")
    print(f"  Generated shape: {topk_output.shape}")

    # Test top-p sampling
    topp_output = generator.sample_top_p(context, max_new_tokens=5, p=0.9)
    print(f"\nTop-p sampling (p=0.9):")
    print(f"  Generated shape: {topp_output.shape}")

    print("\n" + "="*80)


def test_question5_positional_encoding():
    """Test Positional Encoding"""
    print("\n" + "="*80)
    print("QUESTION 5: Positional Encoding Test")
    print("="*80)

    # Test sinusoidal encoding
    sin_pe = SinusoidalPositionalEncoding(d_model=128, max_len=100)
    x = torch.randn(4, 20, 128)
    output = sin_pe(x)

    print(f"Sinusoidal Positional Encoding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  PE buffer shape: {sin_pe.pe.shape}")

    # Test learnable encoding
    learn_pe = LearnablePositionalEmbedding(max_len=100, d_model=128)
    output = learn_pe(x)

    print(f"\nLearnable Positional Embedding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Embedding weight shape: {learn_pe.pos_embedding.weight.shape}")

    print("\n" + "="*80)


def test_question6_seq2seq():
    """Test Sequence-to-Sequence Model"""
    print("\n" + "="*80)
    print("QUESTION 6: Sequence-to-Sequence Model Test")
    print("="*80)

    input_size = 50
    output_size = 50
    hidden_size = 128

    model = Seq2SeqModel(input_size, output_size, hidden_size, num_layers=2)

    # Create dummy data (one-hot encoded)
    src = F.one_hot(torch.randint(0, input_size, (8, 15)), num_classes=input_size).float()
    tgt = F.one_hot(torch.randint(0, output_size, (8, 12)), num_classes=output_size).float()

    # Test with teacher forcing
    output = model(src, tgt, teacher_forcing_ratio=1.0)

    print(f"Seq2Seq Model:")
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output shape: {output.shape}")

    # Test with no teacher forcing
    output_no_tf = model(src, tgt, teacher_forcing_ratio=0.0)
    print(f"\nWith teacher_forcing_ratio=0.0:")
    print(f"  Output shape: {output_no_tf.shape}")

    print("\n" + "="*80)


if __name__ == "__main__":
    """Run all tests"""

    print("\n" + "="*80)
    print("PyTorch Autoregressive Modeling Interview Questions - Practice Set")
    print("="*80)

    # Run tests
    test_question1_vanilla_rnn()
    test_question2_lstm()
    test_question3_causal_attention()
    test_question4_generator()
    test_question5_positional_encoding()
    test_question6_seq2seq()

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
