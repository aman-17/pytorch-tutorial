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
                 lr_scale=1.0, init_scale=1.0, readout=False):
        super(muPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_scale = lr_scale
        self.readout = readout
        
        # TODO: Create weight parameter
        # Initialize with muP scaling: std = init_scale / sqrt(in_features)
        
        # TODO: Create bias parameter if needed
        
        # TODO: Store muP-specific attributes for optimizer
        
    def forward(self, x):
        # TODO: Implement forward pass
        # For readout layers, scale output by 1/width_multiplier
        pass

class muPAttention(nn.Module):
    def __init__(self, d_model, n_heads, width_multiplier=1.0):
        super(muPAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.width_multiplier = width_multiplier
        
        # TODO: Create Q, K, V projection layers with appropriate muP scaling
        # Query and Key should have lr_scale = 1.0
        # Value should have lr_scale = 1.0
        
        # TODO: Create output projection with appropriate scaling
        
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        
        # TODO: Compute Q, K, V
        
        # TODO: Reshape for multi-head attention
        
        # TODO: Compute attention scores
        # IMPORTANT: Scale attention logits by 1/sqrt(d_k) AND by width_multiplier
        # This is crucial for muP to work correctly
        
        # TODO: Apply softmax and compute attention output
        
        # TODO: Reshape and apply output projection
        pass

class muPTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, width_multiplier=1.0):
        super(muPTransformerLayer, self).__init__()
        
        # TODO: Create attention layer
        
        # TODO: Create feed-forward layers with muP scaling
        # First FF layer: lr_scale = 1.0
        # Second FF layer: lr_scale = 1.0, but mark as readout
        
        # TODO: Create layer norms
        
    def forward(self, x):
        # TODO: Implement transformer layer with residual connections
        # x -> LayerNorm -> Attention -> Add & Norm -> FFN -> Add & Norm
        pass

class muPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_seq_len=512, base_width=64):
        super(muPTransformer, self).__init__()
        self.d_model = d_model
        self.base_width = base_width
        self.width_multiplier = d_model / base_width
        
        # TODO: Create embedding layer with muP scaling
        # Embedding should have lr_scale = 1.0, special initialization
        
        # TODO: Create positional encoding
        
        # TODO: Create transformer layers
        
        # TODO: Create output layer with muP readout scaling
        
    def forward(self, x):
        # TODO: Implement forward pass
        # embedding -> add position -> transformer layers -> output projection
        pass

def create_mup_optimizer(model, base_lr=1e-3):
    """Create optimizer with muP-appropriate learning rates"""
    param_groups = []
    
    # TODO: Group parameters by their muP scaling requirements
    # 1. Embedding parameters: lr = base_lr
    # 2. Weight matrices: lr = base_lr * lr_scale  
    # 3. Readout layer: lr = base_lr * lr_scale
    # 4. Biases and LayerNorms: lr = base_lr
    
    # Hint: Use named_parameters() and check for muP attributes
    
    return torch.optim.AdamW(param_groups)

def test_coordinate_check():
    """
    Coordinate check: Verify that activations have stable statistics
    across different widths when using muP parameterization.
    """
    # TODO: Create models with different widths (64, 128, 256)
    # TODO: Use same input and check activation statistics
    # TODO: Verify that activation means/stds are similar across widths
    
    widths = [64, 128, 256]
    results = {}
    
    for width in widths:
        # TODO: Create model and run forward pass
        # TODO: Collect activation statistics from each layer
        pass
    
    # TODO: Print/plot results to verify coordinate check
    return results

def demonstrate_hyperparameter_transfer():
    """
    Demonstrate that optimal learning rate found for small model
    transfers to larger model with muP.
    """
    # TODO: Train small model (width=64) with different learning rates
    # TODO: Find optimal learning rate
    # TODO: Apply same learning rate to larger model (width=256)
    # TODO: Show that performance transfers
    
    print("=== Hyperparameter Transfer Demonstration ===")
    
    # Small model training
    small_model = muPTransformer(vocab_size=1000, d_model=64, n_heads=4, 
                               n_layers=2, d_ff=256)
    
    # TODO: Train with different LRs and find optimal
    
    # Large model training  
    large_model = muPTransformer(vocab_size=1000, d_model=256, n_heads=8,
                               n_layers=2, d_ff=1024)
    
    # TODO: Apply optimal LR from small model to large model
    # TODO: Compare performance

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