"""
Question 1: Data Parallel Training with Gradient Synchronization
Scenario: You're training a 7B parameter model across 8 GPUs using PyTorch DDP.

Key concepts:
1. DistributedDataParallel (DDP) setup and initialization
2. Gradient accumulation with proper synchronization
3. Gradient clipping across distributed processes
4. Learning rate scaling for multi-GPU training
5. Process group management and communication backends

Tasks:
- Set up DDP initialization and process group
- Implement gradient accumulation with proper synchronization
- Handle gradient clipping across all processes
- Implement learning rate scaling for multi-GPU training
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import os

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.dropout(x)
        x = x + shortcut
        
        # Feed-forward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut
        return x

class LargeTransformer(nn.Module):
    """7B parameter transformer model for distributed training"""
    def __init__(self, vocab_size=50000, d_model=4096, n_layers=32, n_heads=32, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        # Tie input and output embeddings (common practice)
        self.output_head.weight = self.embedding.weight
    
    def forward(self, input_ids, attention_mask=None):
        # Token embeddings + positional embeddings
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.embedding(input_ids)
        pos_ids = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
        pos_embeds = self.pos_embedding(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits

def setup_ddp_training(rank, world_size, backend='nccl'):
    """
    Initialize distributed training environment
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    # Set up environment variables (MASTER_ADDR, MASTER_PORT)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device for current rank
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Using GPU {rank}")
    else:
        print(f"Rank {rank}: Using CPU (CUDA not available)")

class DistributedTrainer:
    def __init__(self, model, rank, world_size, gradient_accumulation_steps=4):
        self.rank = rank
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_count = 0
        
        # Initialize distributed training components
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        self.model = DDP(model, device_ids=[rank])
        
        # Setup optimizer with proper learning rate scaling
        base_lr = 1e-4
        scaled_lr = self.scale_learning_rate(base_lr)
        self.optimizer = AdamW(self.model.parameters(), lr=scaled_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

    def scale_learning_rate(self, base_lr):
        """Scale learning rate for distributed training"""
        # Linear scaling: lr = base_lr * world_size
        # This is the most common approach for large batch training
        scaled_lr = base_lr * self.world_size
        
        if self.rank == 0:
            print(f"Scaling learning rate: {base_lr} -> {scaled_lr} (factor: {self.world_size})")
        
        return scaled_lr
    
    def sync_gradients(self):
        """Synchronize gradients across all processes"""
        # DDP automatically synchronizes gradients during backward pass
        # We just need to ensure all processes are synchronized
        torch.cuda.synchronize()
        
        # Apply gradient clipping
        self.clip_gradients_distributed()
        
        # Update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

    
    def clip_gradients_distributed(self, max_norm=1.0):
        """Clip gradients across all distributed processes"""
        # Compute local gradient norm
        local_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                local_norm += param.grad.data.norm(2).item() ** 2
        local_norm = local_norm ** 0.5
        
        # Convert to tensor for all_reduce
        norm_tensor = torch.tensor(local_norm ** 2, device=f'cuda:{self.rank}')
        
        # Reduce across all processes to get global norm
        dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
        global_norm = (norm_tensor.item() ** 0.5)
        
        # Apply clipping based on global norm
        clip_coef = max_norm / (global_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
    
    def training_step(self, batch, accumulate_grad=True):
        """Single training step with gradient accumulation"""
        # Move batch to device
        device = f'cuda:{self.rank}'
        for key in batch:
            batch[key] = batch[key].to(device)
        
        # Forward pass
        outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
        
        # Scale loss for gradient accumulation
        if accumulate_grad:
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.step_count += 1
        
        # Synchronize and update when accumulation is complete
        if self.step_count % self.gradient_accumulation_steps == 0:
            self.sync_gradients()
        
        return loss.item()

    def save_checkpoint_distributed(self, epoch, step, loss):
        """Save checkpoint in distributed setting"""
        # Only rank 0 should save the checkpoint
        if self.rank == 0:
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "model_state_dict": self.model.module.state_dict(),  # Unwrap DDP
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "world_size": self.world_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps
            }
            
            checkpoint_path = f"checkpoint_epoch_{epoch}_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Synchronize all processes
        dist.barrier()
    
    def load_checkpoint_distributed(self, checkpoint_path):
        """Load checkpoint in distributed setting"""
        # Load checkpoint on all ranks
        map_location = f'cuda:{self.rank}'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state dict (DDP automatically handles the module prefix)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore training metadata
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        loss = checkpoint["loss"]
        
        # Ensure synchronization across ranks
        dist.barrier()
        
        if self.rank == 0:
            print(f"Checkpoint loaded: epoch {epoch}, step {step}, loss {loss:.4f}")
        
        return epoch, step, loss

def create_dummy_data(batch_size, seq_len, vocab_size):
    """Create dummy training data"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, attention_mask, labels

def main():
    """Main training function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, help='Rank of current process')
    parser.add_argument('--world-size', type=int, default=1, help='Total number of processes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save-interval', type=int, default=1000, help='Save checkpoint every N steps')
    args = parser.parse_args()
    
    # Get rank and world_size from environment if using torchrun
    rank = int(os.environ.get('LOCAL_RANK', args.rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    
    # Setup distributed environment (skip for single GPU demo)
    if world_size > 1:
        setup_ddp_training(rank, world_size)
    else:
        print("Running single GPU demo (no distributed setup needed)")
    
    # Hyperparameters
    vocab_size = 50000
    d_model = 4096
    n_layers = 32
    n_heads = 32
    batch_size = 4  # Per GPU batch size
    seq_len = 2048
    
    # Create model
    model = LargeTransformer(vocab_size, d_model, n_layers, n_heads, seq_len)
    
    # Create trainer
    if world_size > 1:
        trainer = DistributedTrainer(model, rank, world_size)
    else:
        # Single GPU trainer for demo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        class SingleGPUTrainer:
            def __init__(self, model, optimizer, rank=0):
                self.model = model
                self.optimizer = optimizer
                self.rank = rank
                
            def training_step(self, batch):
                device = next(self.model.parameters()).device
                for key in batch:
                    batch[key] = batch[key].to(device)
                
                outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                return loss.item()
                
            def save_checkpoint_distributed(self, epoch, step, loss):
                print(f"Would save checkpoint: epoch {epoch}, step {step}, loss {loss:.4f}")
        
        trainer = SingleGPUTrainer(model, optimizer, rank)
    
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    
    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Simulate training steps
        for step in range(100):  # 100 steps per epoch
            # Create dummy batch
            input_ids, attention_mask, labels = create_dummy_data(batch_size, seq_len, vocab_size)
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask, 
                'labels': labels
            }
            
            # Training step
            loss = trainer.training_step(batch)
            global_step += 1
            
            # Logging
            if rank == 0 and step % 10 == 0:
                print(f"Step {global_step}, Loss: {loss:.4f}")
            
            # Save checkpoint
            if global_step % args.save_interval == 0:
                trainer.save_checkpoint_distributed(epoch, global_step, loss)
    
    # Cleanup
    dist.destroy_process_group()

    
    if rank == 0:
        print("Training completed!")
    
if __name__ == "__main__":
    # Example usage:
    # torchrun --nproc_per_node=8 --nnodes=1 question1.py --epochs 5
    main()