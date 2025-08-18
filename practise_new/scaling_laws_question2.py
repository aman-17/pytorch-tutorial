"""
Question 2: muP (Maximal Update Parameterization) Implementation (Medium-Hard)

Implement muP parameterization for a simple transformer to enable hyperparameter
transfer across different model widths. muP ensures that optimal hyperparameters
found for small models transfer to larger models.

Key muP principles:
1. Scale learning rates differently for different parameter types
2. Scale weight initialization based on fan-in
3. Scale attention logits and readout layers appropriately

Your task: Implement muP scaling rules and demonstrate hyperparameter transfer.
"""

import torch
import torch.nn as nn
import math

class muPLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 lr_scale=1.0, init_scale=1.0, readout=False, width_multiplier=1.0):
        super(muPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_scale = lr_scale
        self.readout = readout
        self.width_multiplier = width_multiplier
        
        # Create weight parameter with muP scaling
        # Initialize with muP scaling: std = init_scale / sqrt(in_features)
        std = init_scale / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Store muP-specific attributes for optimizer
        self.weight.lr_scale = lr_scale
        if self.bias is not None:
            self.bias.lr_scale = 1.0  # Biases always use base learning rate
        
    def forward(self, x):
        # Implement forward pass
        output = torch.mm(x.view(-1, x.size(-1)), self.weight.t()).view(*x.shape[:-1], self.out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        # For readout layers, scale output by 1/width_multiplier
        if self.readout:
            output = output / self.width_multiplier
            
        return output

class muPAttention(nn.Module):
    def __init__(self, d_model, n_heads, width_multiplier=1.0):
        super(muPAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.width_multiplier = width_multiplier
        
        # Create Q, K, V projection layers with appropriate muP scaling
        # Query and Key should have lr_scale = 1.0
        # Value should have lr_scale = 1.0
        self.q_proj = muPLinear(d_model, d_model, bias=False, lr_scale=1.0, width_multiplier=width_multiplier)
        self.k_proj = muPLinear(d_model, d_model, bias=False, lr_scale=1.0, width_multiplier=width_multiplier)
        self.v_proj = muPLinear(d_model, d_model, bias=False, lr_scale=1.0, width_multiplier=width_multiplier)
        
        # Create output projection with appropriate scaling
        self.o_proj = muPLinear(d_model, d_model, bias=False, lr_scale=1.0, width_multiplier=width_multiplier)
        
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        
        # Compute Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x) 
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # IMPORTANT: In muP, attention scores are scaled by 1/sqrt(d_k)
        # The width_multiplier scaling is handled in the linear layers, not here
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        
        # Apply softmax and compute attention output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        return output

class muPTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, width_multiplier=1.0):
        super(muPTransformerLayer, self).__init__()
        
        # Create attention layer
        self.attention = muPAttention(d_model, n_heads, width_multiplier)
        
        # Create feed-forward layers with muP scaling
        # First FF layer: lr_scale = 1.0
        # Second FF layer: lr_scale = 1.0, but mark as readout
        self.ff1 = muPLinear(d_model, d_ff, lr_scale=1.0, width_multiplier=width_multiplier)
        self.ff2 = muPLinear(d_ff, d_model, lr_scale=1.0, readout=True, width_multiplier=width_multiplier)
        
        # Create layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Store scaling for layer norm parameters
        for param in self.norm1.parameters():
            param.lr_scale = 1.0
        for param in self.norm2.parameters():
            param.lr_scale = 1.0
        
    def forward(self, x):
        # Implement transformer layer with residual connections
        # x -> LayerNorm -> Attention -> Add & Norm -> FFN -> Add & Norm
        
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x
        
        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.ff1(x)
        x = torch.relu(x)
        x = self.ff2(x)
        x = residual + x
        
        return x

class muPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_seq_len=512, base_width=64):
        super(muPTransformer, self).__init__()
        self.d_model = d_model
        self.base_width = base_width
        self.width_multiplier = d_model / base_width
        
        # Create embedding layer with muP scaling
        # Embedding should have lr_scale = 1.0, but std should scale with 1/sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=1.0/math.sqrt(d_model))
        self.embedding.weight.lr_scale = 1.0
        
        # Create positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=1.0/math.sqrt(d_model))
        self.pos_embedding.weight.lr_scale = 1.0
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            muPTransformerLayer(d_model, n_heads, d_ff, self.width_multiplier)
            for _ in range(n_layers)
        ])
        
        # Create output layer with muP readout scaling
        self.output_layer = muPLinear(d_model, vocab_size, bias=False, lr_scale=1.0, 
                                    readout=True, width_multiplier=self.width_multiplier)
        
    def forward(self, x):
        # Implement forward pass
        # embedding -> add position -> transformer layers -> output projection
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_embeds = self.embedding(x)
        
        # Positional embeddings
        pos_ids = torch.arange(seq_len, device=x.device)
        pos_embeds = self.pos_embedding(pos_ids).unsqueeze(0)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        output = self.output_layer(x)
        
        return output

def create_mup_optimizer(model, base_lr=1e-3):
    """Create optimizer with muP-appropriate learning rates"""
    param_groups = []
    
    # Group parameters by their muP scaling requirements
    # 1. Embedding parameters: lr = base_lr
    # 2. Weight matrices: lr = base_lr * lr_scale  
    # 3. Readout layer: lr = base_lr * lr_scale
    # 4. Biases and LayerNorms: lr = base_lr
    
    # Simple approach - group by lr_scale value
    lr_groups = {}
    
    for name, param in model.named_parameters():
        if hasattr(param, 'lr_scale'):
            lr_scale = param.lr_scale
        else:
            lr_scale = 1.0  # Default learning rate scale
        
        if lr_scale not in lr_groups:
            lr_groups[lr_scale] = []
        lr_groups[lr_scale].append(param)
    
    # Create parameter groups
    for lr_scale, params in lr_groups.items():
        param_groups.append({
            'params': params,
            'lr': base_lr * lr_scale,
            'name': f'lr_scale_{lr_scale}'
        })
    
    # If no groups were created, use all parameters with base lr
    if not param_groups:
        param_groups.append({'params': model.parameters(), 'lr': base_lr})
    
    return torch.optim.AdamW(param_groups)

def test_coordinate_check():
    """
    Coordinate check: Verify that activations have stable statistics
    across different widths when using muP parameterization.
    """
    # Create models with different widths (64, 128, 256)
    # Use same input and check activation statistics
    # Verify that activation means/stds are similar across widths
    
    widths = [64, 128, 256]
    results = {}
    
    # Create same input for all models
    batch_size, seq_len = 4, 16
    vocab_size = 1000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    for width in widths:
        print(f"\nTesting width {width}...")
        
        # Create model and run forward pass
        model = muPTransformer(
            vocab_size=vocab_size,
            d_model=width, 
            n_heads=4,
            n_layers=2,
            d_ff=width*2,
            base_width=64
        )
        
        # Collect activation statistics from each layer
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append({
                    'mean': output.detach().mean().item(),
                    'std': output.detach().std().item(),
                    'shape': output.shape
                })
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, muPLinear):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        results[width] = {
            'activations': activations,
            'output_mean': output.mean().item(),
            'output_std': output.std().item()
        }
        
        print(f"  Output mean: {results[width]['output_mean']:.4f}")
        print(f"  Output std: {results[width]['output_std']:.4f}")
    
    # Print/plot results to verify coordinate check
    print(f"\n=== Coordinate Check Results ===")
    print("If muP is working correctly, statistics should be similar across widths:")
    for width in widths:
        print(f"Width {width}: mean={results[width]['output_mean']:.4f}, std={results[width]['output_std']:.4f}")
    
    return results

def demonstrate_hyperparameter_transfer():
    """
    Demonstrate that optimal learning rate found for small model
    transfers to larger model with muP.
    """
    # Train small model (width=64) with different learning rates
    # Find optimal learning rate
    # Apply same learning rate to larger model (width=256)
    # Show that performance transfers
    
    print("=== Hyperparameter Transfer Demonstration ===")
    
    # Small model training
    small_model = muPTransformer(vocab_size=1000, d_model=64, n_heads=4, 
                               n_layers=2, d_ff=256)
    
    # Test different learning rates on small model
    test_lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    small_model_losses = []
    
    print("\nTesting learning rates on small model (d_model=64):")
    
    for lr in test_lrs:
        model = muPTransformer(vocab_size=1000, d_model=64, n_heads=4, 
                             n_layers=2, d_ff=256)
        optimizer = create_mup_optimizer(model, base_lr=lr)
        
        # Simulate a few training steps
        total_loss = 0
        num_steps = 10
        
        for step in range(num_steps):
            # Create dummy batch
            batch_size, seq_len = 8, 32
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            targets = torch.randint(0, 1000, (batch_size, seq_len))
            
            # Forward pass
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), targets.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_steps
        small_model_losses.append(avg_loss)
        print(f"  LR {lr:.1e}: Average loss = {avg_loss:.4f}")
    
    # Find optimal learning rate (lowest loss)
    optimal_idx = small_model_losses.index(min(small_model_losses))
    optimal_lr = test_lrs[optimal_idx]
    print(f"\nOptimal LR for small model: {optimal_lr:.1e}")
    
    # Large model training with optimal LR
    print(f"\nApplying optimal LR to large model (d_model=256):")
    large_model = muPTransformer(vocab_size=1000, d_model=256, n_heads=8,
                               n_layers=2, d_ff=1024)
    
    optimizer = create_mup_optimizer(large_model, base_lr=optimal_lr)
    
    # Test the same optimal LR on large model
    total_loss = 0
    num_steps = 10
    
    for step in range(num_steps):
        # Create dummy batch
        batch_size, seq_len = 8, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        outputs = large_model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    large_model_loss = total_loss / num_steps
    print(f"Large model loss with transferred LR: {large_model_loss:.4f}")
    print(f"Small model loss with optimal LR: {small_model_losses[optimal_idx]:.4f}")
    
    if abs(large_model_loss - small_model_losses[optimal_idx]) < 0.5:
        print("✓ Hyperparameter transfer successful! Losses are similar.")
    else:
        print("✗ Hyperparameter transfer may need tuning.")

# Test your implementation
if __name__ == "__main__":
    print("=== Testing muP Implementation ===")
    
    # Test basic model creation
    model = muPTransformer(vocab_size=1000, d_model=128, n_heads=4, 
                          n_layers=2, d_ff=512)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
        print(f"Model output shape: {output.shape}")
    
    # Test optimizer creation
    optimizer = create_mup_optimizer(model)
    print(f"Optimizer created with {len(optimizer.param_groups)} parameter groups")
    
    # Run coordinate check
    print("\n=== Running Coordinate Check ===")
    coord_results = test_coordinate_check()
    
    # Demonstrate hyperparameter transfer
    print("\n=== Testing Hyperparameter Transfer ===")
    demonstrate_hyperparameter_transfer()