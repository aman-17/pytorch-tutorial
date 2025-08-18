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
        # TODO: Implement L(N) = (Nc/N)^αN
        # Handle case where N might be less than Nc
        return (self.Nc / N) ** self.alpha_N
        pass
    
    def loss_from_data(self, D):
        """Calculate test loss given dataset size"""
        # TODO: Implement L(D) = (Dc/D)^αD
        # Handle case where D might be less than Dc
        return (self.Dc / D) ** self.alpha_D
        pass
    
    def loss_from_compute(self, C):
        """Calculate test loss given compute budget"""
        # TODO: Implement L(C) = (Cc/C)^αC
        # Handle case where C might be less than Cc
        return (self.Cc / C) ** self.alpha_C
        pass
    
    def optimal_parameters_for_compute(self, C):
        """Find optimal model size given compute budget"""
        # TODO: According to Kaplan, optimal N scales as C^0.73
        # Implement: N_opt = Nc * (C/Cc)^0.73
        return self.Nc * (C / self.Cc) ** 0.73
        pass
    
    def optimal_data_for_compute(self, C):
        """Find optimal dataset size given compute budget"""
        # TODO: According to Kaplan, optimal D scales as C^0.27
        # Implement: D_opt = Dc * (C/Cc)^0.27
        return self.Dc * (C / self.Cc) ** 0.27
        pass
    
    def compute_equivalent_loss(self, N, D, C):
        """
        Calculate loss when all three factors matter.
        Use the fact that losses combine as: L = max(L(N), L(D), L(C))
        """
        # TODO: Calculate loss from each factor and return the maximum
        # This represents the bottleneck factor
        return max(self.loss_from_parameters(N), self.loss_from_data(D), self.loss_from_compute(C))
        pass

def plot_scaling_curves():
    """Plot the three fundamental scaling curves"""
    scaling = KaplanScalingLaws()
    
    # TODO: Create parameter ranges for plotting
    # N_range: from 1e6 to 1e10 parameters
    # D_range: from 1e6 to 1e10 tokens  
    # C_range: from 1e17 to 1e21 FLOPs
    N_range = np.logspace(6, 10, 100)  # 1e6 to 1e10 parameters
    D_range = np.logspace(6, 10, 100)  # 1e6 to 1e10 tokens
    C_range = np.logspace(17, 21, 100)  # 1e17 to 1e21 FLO
    
    # TODO: Calculate losses for each range
    losses_N = [scaling.loss_from_parameters(N) for N in N_range]
    losses_D = [scaling.loss_from_data(D) for D in D_range]  
    losses_C = [scaling.loss_from_compute(C) for C in C_range]
    
    # TODO: Create 3 subplots showing each scaling law
    # Use log-log plots and label axes properly
    
    pass

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