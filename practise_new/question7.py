"""
Question 7: Advanced Learning Rate Scheduler with Warmup (Medium-Hard)

Implement a custom learning rate scheduler that combines multiple scheduling strategies:
1. Linear warmup for the first few epochs
2. Cosine annealing with restarts
3. Exponential decay after each restart

This scheduler is commonly used in transformer training and provides smooth
learning rate transitions with periodic restarts to escape local minima.

Requirements:
- Linear warmup from 0 to base_lr over warmup_epochs
- Cosine annealing from peak to min_lr over cycle_length epochs  
- After each cycle, multiply base_lr by decay_factor
- Support for multiple restarts

Example usage:
scheduler = WarmupCosineRestartScheduler(optimizer, base_lr=1e-3, warmup_epochs=5, 
                                       cycle_length=20, min_lr=1e-6, decay_factor=0.8)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

class WarmupCosineRestartScheduler:
    def __init__(self, optimizer, base_lr, warmup_epochs, cycle_length, 
                 min_lr=0, decay_factor=1.0, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.last_epoch = last_epoch
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # Initialize current epoch
        self.current_epoch = last_epoch + 1
    
    def get_lr(self, epoch):
        """Calculate learning rate for given epoch"""
        # Determine current cycle and position within cycle
        total_cycle_length = self.warmup_epochs + self.cycle_length
        cycle_num = epoch // total_cycle_length
        epoch_in_cycle = epoch % total_cycle_length
        
        # Calculate current base learning rate (with decay)
        current_base_lr = self.base_lr * (self.decay_factor ** cycle_num)
        
        if epoch_in_cycle < self.warmup_epochs:
            # Warmup phase: linear increase from 0 to current_base_lr
            lr = current_base_lr * (epoch_in_cycle / self.warmup_epochs)
        else:
            # Cosine annealing phase
            cosine_epoch = epoch_in_cycle - self.warmup_epochs
            cosine_progress = cosine_epoch / self.cycle_length
            
            # Cosine annealing from current_base_lr to min_lr
            lr = self.min_lr + (current_base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * cosine_progress)) / 2
        
        return lr
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.current_epoch
        
        # Calculate new learning rate
        new_lr = self.get_lr(epoch)
        
        # Update optimizer's learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update current epoch
        self.current_epoch = epoch + 1
    
    def get_current_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def plot_lr_schedule(scheduler_class, total_epochs=100, **scheduler_kwargs):
    """Plot learning rate schedule for visualization"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping plot.")
        return
    
    # Create dummy optimizer and scheduler
    dummy_model = SimpleModel()
    dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=scheduler_kwargs.get('base_lr', 1e-3))
    scheduler = scheduler_class(optimizer=dummy_optimizer, **scheduler_kwargs)
    
    # Simulate training for total_epochs and collect learning rates
    epochs = []
    learning_rates = []
    
    for epoch in range(total_epochs):
        lr = scheduler.get_lr(epoch)
        epochs.append(epoch)
        learning_rates.append(lr)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, learning_rates, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Warmup Cosine Restart Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotations for different phases
    warmup_epochs = scheduler_kwargs.get('warmup_epochs', 5)
    cycle_length = scheduler_kwargs.get('cycle_length', 20)
    
    # Mark warmup phases
    for cycle in range(total_epochs // (warmup_epochs + cycle_length) + 1):
        warmup_start = cycle * (warmup_epochs + cycle_length)
        warmup_end = warmup_start + warmup_epochs
        if warmup_end < total_epochs:
            plt.axvspan(warmup_start, warmup_end, alpha=0.2, color='green', label='Warmup' if cycle == 0 else "")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Learning rate schedule plotted for {total_epochs} epochs")

class SimpleModel(nn.Module):
    """Simple model for testing the scheduler"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def test_scheduler():
    """Test the custom scheduler with actual training loop"""
    print("\nTesting scheduler with training loop...")
    
    # Create model, optimizer, and scheduler
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineRestartScheduler(
        optimizer=optimizer,
        base_lr=1e-3,
        warmup_epochs=5,
        cycle_length=10,
        min_lr=1e-6,
        decay_factor=0.9
    )
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    criterion = nn.MSELoss()
    
    print("Training simulation:")
    for epoch in range(30):
        # Simulate training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step(epoch)
        current_lr = scheduler.get_current_lr()[0]
        
        if epoch % 5 == 0:  # Print every 5 epochs
            print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}, Loss = {loss.item():.6f}")
    
    print("Scheduler test completed!")

# Test your implementation
if __name__ == "__main__":
    # Create test setup
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Test scheduler
    scheduler = WarmupCosineRestartScheduler(
        optimizer=optimizer,
        base_lr=1e-3,
        warmup_epochs=5,
        cycle_length=20,
        min_lr=1e-6,
        decay_factor=0.8
    )
    
    # Test learning rate calculation
    print("Learning rate schedule:")
    for epoch in range(50):
        lr = scheduler.get_lr(epoch)
        print(f"Epoch {epoch:2d}: LR = {lr:.6f}")
        
        if epoch % 10 == 0:  # Print every 10 epochs
            continue
    
    # Run comprehensive test
    test_scheduler()
    
    # Uncomment to plot schedule (requires matplotlib)
    # plot_lr_schedule(WarmupCosineRestartScheduler, total_epochs=100,
    #                 base_lr=1e-3, warmup_epochs=5, cycle_length=20, 
    #                 min_lr=1e-6, decay_factor=0.8)