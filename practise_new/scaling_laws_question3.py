"""
Question 3: Chinchilla Optimal Compute Allocation (Medium)

Implement the Chinchilla scaling laws that determine optimal allocation of compute
between model parameters and training data. Chinchilla showed that models like GPT-3
were "undertrained" and should use more training data for optimal performance.

Key Chinchilla findings:
- Optimal parameters N and tokens D should scale equally with compute: N ∝ C^0.5, D ∝ C^0.5
- For every 2x increase in compute, increase both model size and data by ~1.4x
- Chinchilla-optimal models significantly outperform parameter-count-matched models

Your task: Implement Chinchilla scaling laws and compare with Kaplan predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ScalingConstants:
    """Constants from Chinchilla paper for different fitting approaches"""
    # Approach 1: Parametric fitting  
    A: float = 406.4
    B: float = 410.7
    E: float = 1.69
    alpha: float = 0.34
    beta: float = 0.28
    
    # Approach 2: Chinchilla coefficients
    a: float = 1.82
    b: float = 1.38

class ChinchillaScaling:
    def __init__(self):
        self.constants = ScalingConstants()
        
        # Reference values from Chinchilla paper
        self.N0 = 8.8e6     # Reference parameter count
        self.D0 = 5.4e6     # Reference dataset size (tokens)
        
    def loss_parametric(self, N, D):
        """
        Compute loss using Chinchilla's parametric approach (Equation 1).
        L(N,D) = E + A/N^alpha + B/D^beta
        """
        # TODO: Implement the parametric loss function
        # Use self.constants.A, B, E, alpha, beta
        pass
    
    def optimal_params_chinchilla(self, C):
        """
        Compute optimal N and D given compute budget using Chinchilla's approach.
        Uses the constraint that C = 6*N*D (FLOPs approximation)
        And optimization: dL/dN = dL/dD = 0 under constraint
        """
        # TODO: Implement optimal allocation
        # From Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5
        # Use the specific coefficients from the paper
        
        # Hint: The optimal allocation gives roughly equal scaling
        # N_opt = self.constants.a * (C/6)**(1/2)  
        # D_opt = self.constants.b * (C/6)**(1/2)
        pass
    
    def optimal_params_kaplan(self, C):
        """
        Compute optimal N and D according to Kaplan scaling laws for comparison.
        Kaplan: N_opt ∝ C^0.73, D_opt ∝ C^0.27
        """
        # TODO: Implement Kaplan's allocation strategy
        # Use power laws: N ∝ C^0.73, D ∝ C^0.27
        # Normalize using self.N0, self.D0
        pass
    
    def compute_flops(self, N, D):
        """
        Estimate compute (FLOPs) required for training.
        Approximation: C ≈ 6*N*D (forward + backward pass)
        """
        # TODO: Implement FLOPs calculation
        pass
    
    def loss_kaplan(self, N, D):
        """
        Kaplan loss function for comparison: L = (Nc/N)^αN + (Dc/D)^αD
        """
        # TODO: Implement Kaplan loss
        # Use αN = 0.076, αD = 0.095 from Kaplan paper
        Nc = 8.8e6
        Dc = 5.4e6
        alpha_N = 0.076
        alpha_D = 0.095
        pass

def compare_scaling_predictions():
    """Compare Chinchilla vs Kaplan predictions for different compute budgets"""
    scaling = ChinchillaScaling()
    
    # TODO: Define range of compute budgets
    compute_budgets = np.logspace(18, 22, 20)  # 1e18 to 1e22 FLOPs
    
    results = {
        'compute': compute_budgets,
        'chinchilla_N': [],
        'chinchilla_D': [], 
        'chinchilla_loss': [],
        'kaplan_N': [],
        'kaplan_D': [],
        'kaplan_loss': []
    }
    
    for C in compute_budgets:
        # TODO: Get optimal allocations from both methods
        
        # Chinchilla optimal
        # N_chin, D_chin = scaling.optimal_params_chinchilla(C)
        # loss_chin = scaling.loss_parametric(N_chin, D_chin)
        
        # Kaplan optimal  
        # N_kap, D_kap = scaling.optimal_params_kaplan(C)
        # loss_kap = scaling.loss_kaplan(N_kap, D_kap)
        
        # TODO: Store results
        pass
    
    return results

def plot_scaling_comparison(results):
    """Plot comparison between Chinchilla and Kaplan scaling"""
    # TODO: Create subplots comparing:
    # 1. Model size (N) vs Compute
    # 2. Dataset size (D) vs Compute  
    # 3. Loss vs Compute
    # 4. N vs D relationship
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO: Plot 1 - Model size vs Compute
    # TODO: Plot 2 - Dataset size vs Compute
    # TODO: Plot 3 - Loss vs Compute
    # TODO: Plot 4 - N vs D relationship
    
    plt.tight_layout()
    return fig

def analyze_gpt3_efficiency():
    """
    Analyze whether GPT-3 was optimally trained according to Chinchilla.
    GPT-3: 175B parameters, ~300B tokens, ~3.1e23 FLOPs
    """
    scaling = ChinchillaScaling()
    
    # GPT-3 specifications
    gpt3_params = 175e9      # 175B parameters
    gpt3_tokens = 300e9      # 300B tokens  
    gpt3_compute = scaling.compute_flops(gpt3_params, gpt3_tokens)
    
    print("=== GPT-3 Analysis ===")
    print(f"GPT-3 Parameters: {gpt3_params:.1e}")
    print(f"GPT-3 Training Tokens: {gpt3_tokens:.1e}")
    print(f"GPT-3 Estimated Compute: {gpt3_compute:.1e} FLOPs")
    
    # TODO: Calculate Chinchilla-optimal allocation for same compute
    # optimal_N, optimal_D = scaling.optimal_params_chinchilla(gpt3_compute)
    
    # TODO: Compare losses
    # gpt3_loss = scaling.loss_parametric(gpt3_params, gpt3_tokens)
    # optimal_loss = scaling.loss_parametric(optimal_N, optimal_D)
    
    # TODO: Print comparison and recommendations
    print(f"\nChinchilla-optimal for same compute:")
    # print(f"Optimal Parameters: {optimal_N:.1e}")
    # print(f"Optimal Tokens: {optimal_D:.1e}")
    # print(f"GPT-3 Loss: {gpt3_loss:.4f}")
    # print(f"Optimal Loss: {optimal_loss:.4f}")
    # print(f"Loss Improvement: {((gpt3_loss - optimal_loss) / gpt3_loss * 100):.1f}%")

def chinchilla_training_calculator(target_performance, max_compute):
    """
    Calculate training requirements to achieve target performance
    within compute constraints.
    """
    scaling = ChinchillaScaling()
    
    # TODO: Given target loss and compute constraint, find feasible N,D
    # This involves solving: loss_parametric(N,D) = target_performance
    # Subject to: compute_flops(N,D) <= max_compute
    
    print(f"\n=== Training Calculator ===")
    print(f"Target Loss: {target_performance:.4f}")
    print(f"Max Compute: {max_compute:.1e} FLOPs")
    
    # TODO: Binary search or optimization to find optimal N,D
    # that achieves target performance within compute budget
    pass

# Test your implementation
if __name__ == "__main__":
    print("=== Testing Chinchilla Scaling Laws ===")
    
    scaling = ChinchillaScaling()
    
    # Test basic functions
    test_N = 100e9   # 100B parameters
    test_D = 200e9   # 200B tokens
    test_C = 1e21    # 1 zettaFLOP
    
    print("=== Basic Function Tests ===")
    print(f"Parametric loss for N={test_N:.0e}, D={test_D:.0e}: {scaling.loss_parametric(test_N, test_D):.4f}")
    print(f"Compute for N={test_N:.0e}, D={test_D:.0e}: {scaling.compute_flops(test_N, test_D):.1e} FLOPs")
    
    # Test optimal allocations
    print(f"\n=== Optimal Allocations for C={test_C:.0e} ===")
    chin_N, chin_D = scaling.optimal_params_chinchilla(test_C)
    kap_N, kap_D = scaling.optimal_params_kaplan(test_C)
    
    print(f"Chinchilla optimal - N: {chin_N:.0e}, D: {chin_D:.0e}")
    print(f"Kaplan optimal - N: {kap_N:.0e}, D: {kap_D:.0e}")
    
    # Compare scaling predictions
    print(f"\n=== Scaling Comparison ===")
    results = compare_scaling_predictions()
    
    # Plot results
    fig = plot_scaling_comparison(results)
    print("Scaling comparison plots generated!")
    
    # Analyze GPT-3
    analyze_gpt3_efficiency()
    
    # Training calculator example
    chinchilla_training_calculator(target_performance=2.0, max_compute=1e22)