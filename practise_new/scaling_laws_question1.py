"""
Question 1: Kaplan Scaling Laws Implementation (Medium)

Implement a simulation of Kaplan scaling laws that models how test loss scales with
model parameters (N), dataset size (D), and compute (C).

According to Kaplan et al., the test loss follows these power laws:
- L(N) = (Nc/N)^αN when data is not the bottleneck
- L(D) = (Dc/D)^αD when parameters are not the bottleneck  
- L(C) = (Cc/C)^αC for compute-limited training

Where αN ≈ 0.076, αD ≈ 0.095, αC ≈ 0.050

Your task: Implement functions to calculate loss scaling and find optimal allocation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class KaplanScalingLaws:
    def __init__(self):
        # Kaplan et al. empirical constants
        self.alpha_N = 0.076  # Parameters scaling exponent
        self.alpha_D = 0.095  # Data scaling exponent  
        self.alpha_C = 0.050  # Compute scaling exponent
        
        # Critical scales (where power laws break down)
        self.Nc = 8.8e6      # Critical parameter count
        self.Dc = 5.4e6      # Critical dataset size
        self.Cc = 3.1e17     # Critical compute (FLOPs)
        
    def loss_from_parameters(self, N):
        """Calculate test loss given model parameters"""
        # L(N) = (Nc/N)^αN
        # Handle case where N might be less than Nc
        if N < self.Nc:
            # For very small models, use a different scaling to avoid division issues
            return (self.Nc / N) ** self.alpha_N
        return (self.Nc / N) ** self.alpha_N
    
    def loss_from_data(self, D):
        """Calculate test loss given dataset size"""
        # L(D) = (Dc/D)^αD
        # Handle case where D might be less than Dc
        if D < self.Dc:
            # For small datasets, extrapolate the power law
            return (self.Dc / D) ** self.alpha_D
        return (self.Dc / D) ** self.alpha_D
    
    def loss_from_compute(self, C):
        """Calculate test loss given compute budget"""
        # L(C) = (Cc/C)^αC
        # Handle case where C might be less than Cc
        if C < self.Cc:
            # For low compute, extrapolate the power law
            return (self.Cc / C) ** self.alpha_C
        return (self.Cc / C) ** self.alpha_C
    
    def optimal_parameters_for_compute(self, C):
        """Find optimal model size given compute budget"""
        # According to Kaplan, optimal N scales as C^0.73
        # N_opt = Nc * (C/Cc)^0.73
        return self.Nc * (C / self.Cc) ** 0.73
    
    def optimal_data_for_compute(self, C):
        """Find optimal dataset size given compute budget"""
        # According to Kaplan, optimal D scales as C^0.27
        # D_opt = Dc * (C/Cc)^0.27
        return self.Dc * (C / self.Cc) ** 0.27
    
    def compute_equivalent_loss(self, N, D, C):
        """
        Calculate loss when all three factors matter.
        Use the fact that losses combine as: L = max(L(N), L(D), L(C))
        """
        # Calculate loss from each factor and return the maximum
        # This represents the bottleneck factor
        loss_N = self.loss_from_parameters(N)
        loss_D = self.loss_from_data(D)
        loss_C = self.loss_from_compute(C)
        return max(loss_N, loss_D, loss_C)

def plot_scaling_curves():
    """Plot the three fundamental scaling curves"""
    scaling = KaplanScalingLaws()
    
    # Create parameter ranges for plotting
    N_range = np.logspace(6, 10, 100)  # 1e6 to 1e10 parameters
    D_range = np.logspace(6, 10, 100)  # 1e6 to 1e10 tokens
    C_range = np.logspace(17, 21, 100)  # 1e17 to 1e21 FLOPs
    
    # Calculate losses for each range
    losses_N = [scaling.loss_from_parameters(N) for N in N_range]
    losses_D = [scaling.loss_from_data(D) for D in D_range]  
    losses_C = [scaling.loss_from_compute(C) for C in C_range]
    
    # Create 3 subplots showing each scaling law
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Loss vs Parameters
    axes[0].loglog(N_range, losses_N, 'b-', linewidth=2, label=f'α_N = {scaling.alpha_N}')
    axes[0].axvline(scaling.Nc, color='r', linestyle='--', alpha=0.7, label=f'Nc = {scaling.Nc:.1e}')
    axes[0].set_xlabel('Model Parameters (N)')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('Loss vs Model Size')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Loss vs Dataset Size
    axes[1].loglog(D_range, losses_D, 'g-', linewidth=2, label=f'α_D = {scaling.alpha_D}')
    axes[1].axvline(scaling.Dc, color='r', linestyle='--', alpha=0.7, label=f'Dc = {scaling.Dc:.1e}')
    axes[1].set_xlabel('Dataset Size (D)')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Loss vs Dataset Size')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Loss vs Compute
    axes[2].loglog(C_range, losses_C, 'purple', linewidth=2, label=f'α_C = {scaling.alpha_C}')
    axes[2].axvline(scaling.Cc, color='r', linestyle='--', alpha=0.7, label=f'Cc = {scaling.Cc:.1e}')
    axes[2].set_xlabel('Compute Budget (C)')
    axes[2].set_ylabel('Test Loss')
    axes[2].set_title('Loss vs Compute')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def find_optimal_allocation(compute_budget):
    """
    Given a compute budget, find the optimal model size and dataset size
    that minimizes test loss according to Kaplan scaling laws.
    """
    scaling = KaplanScalingLaws()
    
    # TODO: Use the optimal allocation formulas
    optimal_N = scaling.optimal_parameters_for_compute(compute_budget)
    optimal_D = scaling.optimal_data_for_compute(compute_budget)
    
    # TODO: Calculate the resulting loss
    optimal_loss = scaling.compute_equivalent_loss(optimal_N, optimal_D, compute_budget)
    
    return optimal_N, optimal_D, optimal_loss

# Test your implementation
if __name__ == "__main__":
    scaling = KaplanScalingLaws()
    
    # Test individual scaling laws
    test_N = 1e8  # 100M parameters
    test_D = 1e9  # 1B tokens
    test_C = 1e20 # 100 quintillion FLOPs
    
    print("=== Testing Individual Scaling Laws ===")
    print(f"Loss from {test_N:.0e} parameters: {scaling.loss_from_parameters(test_N):.4f}")
    print(f"Loss from {test_D:.0e} tokens: {scaling.loss_from_data(test_D):.4f}")
    print(f"Loss from {test_C:.0e} FLOPs: {scaling.loss_from_compute(test_C):.4f}")
    
    # Test optimal allocation
    print("\n=== Testing Optimal Allocation ===")
    compute_budgets = [1e19, 1e20, 1e21]
    
    for C in compute_budgets:
        N_opt, D_opt, L_opt = find_optimal_allocation(C)
        print(f"Compute: {C:.0e} | Optimal N: {N_opt:.0e} | Optimal D: {D_opt:.0e} | Loss: {L_opt:.4f}")
    
    # Plot scaling curves
    print("\n=== Generating Scaling Plots ===")
    plot_scaling_curves()
    print("Scaling plots generated!")