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
        # Implement the parametric loss function
        # Use self.constants.A, B, E, alpha, beta
        A = self.constants.A
        B = self.constants.B
        E = self.constants.E
        alpha = self.constants.alpha
        beta = self.constants.beta
        
        loss = E + A / (N ** alpha) + B / (D ** beta)
        return loss
    
    def optimal_params_chinchilla(self, C):
        """
        Compute optimal N and D given compute budget using Chinchilla's approach.
        Uses the constraint that C = 6*N*D (FLOPs approximation)
        And optimization: dL/dN = dL/dD = 0 under constraint
        """
        # Implement optimal allocation
        # From Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5
        # Use the specific coefficients from the paper
        
        # The optimal allocation gives roughly equal scaling
        # Based on Chinchilla paper coefficients
        a = self.constants.a
        b = self.constants.b
        
        # Adjust scaling to match Chinchilla findings
        # N_opt and D_opt should scale equally with compute
        sqrt_c = np.sqrt(C / 6)  # Normalize by 6 FLOPs per parameter per token
        
        N_opt = a * 1e8 * sqrt_c  # Scale coefficient appropriately
        D_opt = b * 1e8 * sqrt_c  # Scale coefficient appropriately
        
        return N_opt, D_opt
    
    def optimal_params_kaplan(self, C):
        """
        Compute optimal N and D according to Kaplan scaling laws for comparison.
        Kaplan: N_opt ∝ C^0.73, D_opt ∝ C^0.27
        """
        # Implement Kaplan's allocation strategy
        # Use power laws: N ∝ C^0.73, D ∝ C^0.27
        # Normalize using self.N0, self.D0
        
        # Reference compute for normalization
        C0 = 6 * self.N0 * self.D0  # FLOPs for reference model
        
        # Kaplan's allocation
        N_opt = self.N0 * (C / C0) ** 0.73
        D_opt = self.D0 * (C / C0) ** 0.27
        
        return N_opt, D_opt
    
    def compute_flops(self, N, D):
        """
        Estimate compute (FLOPs) required for training.
        Approximation: C ≈ 6*N*D (forward + backward pass)
        """
        # Implement FLOPs calculation
        return 6 * N * D
    
    def loss_kaplan(self, N, D):
        """
        Kaplan loss function for comparison: L = (Nc/N)^αN + (Dc/D)^αD
        """
        # Implement Kaplan loss
        # Use αN = 0.076, αD = 0.095 from Kaplan paper
        Nc = 8.8e6
        Dc = 5.4e6
        alpha_N = 0.076
        alpha_D = 0.095
        
        loss_N = (Nc / N) ** alpha_N
        loss_D = (Dc / D) ** alpha_D
        
        return loss_N + loss_D

def compare_scaling_predictions():
    """Compare Chinchilla vs Kaplan predictions for different compute budgets"""
    scaling = ChinchillaScaling()
    
    # Define range of compute budgets
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
        # Get optimal allocations from both methods
        
        # Chinchilla optimal
        N_chin, D_chin = scaling.optimal_params_chinchilla(C)
        loss_chin = scaling.loss_parametric(N_chin, D_chin)
        
        # Kaplan optimal  
        N_kap, D_kap = scaling.optimal_params_kaplan(C)
        loss_kap = scaling.loss_kaplan(N_kap, D_kap)
        
        # Store results
        results['chinchilla_N'].append(N_chin)
        results['chinchilla_D'].append(D_chin)
        results['chinchilla_loss'].append(loss_chin)
        results['kaplan_N'].append(N_kap)
        results['kaplan_D'].append(D_kap)
        results['kaplan_loss'].append(loss_kap)
    
    return results

def plot_scaling_comparison(results):
    """Plot comparison between Chinchilla and Kaplan scaling"""
    # Create subplots comparing:
    # 1. Model size (N) vs Compute
    # 2. Dataset size (D) vs Compute  
    # 3. Loss vs Compute
    # 4. N vs D relationship
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    compute = results['compute']
    
    # Plot 1 - Model size vs Compute
    axes[0, 0].loglog(compute, results['chinchilla_N'], 'b-', label='Chinchilla', linewidth=2)
    axes[0, 0].loglog(compute, results['kaplan_N'], 'r--', label='Kaplan', linewidth=2)
    axes[0, 0].set_xlabel('Compute (FLOPs)')
    axes[0, 0].set_ylabel('Optimal Parameters (N)')
    axes[0, 0].set_title('Model Size vs Compute')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2 - Dataset size vs Compute
    axes[0, 1].loglog(compute, results['chinchilla_D'], 'b-', label='Chinchilla', linewidth=2)
    axes[0, 1].loglog(compute, results['kaplan_D'], 'r--', label='Kaplan', linewidth=2)
    axes[0, 1].set_xlabel('Compute (FLOPs)')
    axes[0, 1].set_ylabel('Optimal Dataset Size (D)')
    axes[0, 1].set_title('Dataset Size vs Compute')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3 - Loss vs Compute
    axes[1, 0].loglog(compute, results['chinchilla_loss'], 'b-', label='Chinchilla', linewidth=2)
    axes[1, 0].loglog(compute, results['kaplan_loss'], 'r--', label='Kaplan', linewidth=2)
    axes[1, 0].set_xlabel('Compute (FLOPs)')
    axes[1, 0].set_ylabel('Optimal Loss')
    axes[1, 0].set_title('Loss vs Compute')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4 - N vs D relationship
    axes[1, 1].loglog(results['chinchilla_N'], results['chinchilla_D'], 'b-', label='Chinchilla', linewidth=2)
    axes[1, 1].loglog(results['kaplan_N'], results['kaplan_D'], 'r--', label='Kaplan', linewidth=2)
    axes[1, 1].set_xlabel('Parameters (N)')
    axes[1, 1].set_ylabel('Dataset Size (D)')
    axes[1, 1].set_title('N vs D Relationship')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
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
    
    # Calculate Chinchilla-optimal allocation for same compute
    optimal_N, optimal_D = scaling.optimal_params_chinchilla(gpt3_compute)
    
    # Compare losses
    gpt3_loss = scaling.loss_parametric(gpt3_params, gpt3_tokens)
    optimal_loss = scaling.loss_parametric(optimal_N, optimal_D)
    
    # Print comparison and recommendations
    print(f"\nChinchilla-optimal for same compute:")
    print(f"Optimal Parameters: {optimal_N:.1e}")
    print(f"Optimal Tokens: {optimal_D:.1e}")
    print(f"GPT-3 Loss: {gpt3_loss:.4f}")
    print(f"Optimal Loss: {optimal_loss:.4f}")
    
    if gpt3_loss > optimal_loss:
        improvement = ((gpt3_loss - optimal_loss) / gpt3_loss * 100)
        print(f"Loss Improvement: {improvement:.1f}%")
        print(f"\n📈 Analysis: GPT-3 was undertrained!")
        print(f"   - Should use {optimal_N/gpt3_params:.1f}x fewer parameters")
        print(f"   - Should use {optimal_D/gpt3_tokens:.1f}x more training tokens")
    else:
        print(f"GPT-3 appears to be optimally trained according to Chinchilla.")

def chinchilla_training_calculator(target_performance, max_compute):
    """
    Calculate training requirements to achieve target performance
    within compute constraints.
    """
    scaling = ChinchillaScaling()
    
    # Given target loss and compute constraint, find feasible N,D
    # This involves solving: loss_parametric(N,D) = target_performance
    # Subject to: compute_flops(N,D) <= max_compute
    
    print(f"\n=== Training Calculator ===")
    print(f"Target Loss: {target_performance:.4f}")
    print(f"Max Compute: {max_compute:.1e} FLOPs")
    
    # Use Chinchilla optimal allocation as starting point
    optimal_N, optimal_D = scaling.optimal_params_chinchilla(max_compute)
    optimal_loss = scaling.loss_parametric(optimal_N, optimal_D)
    
    print(f"\nChinchilla optimal allocation:")
    print(f"Parameters: {optimal_N:.1e}")
    print(f"Tokens: {optimal_D:.1e}")
    print(f"Resulting Loss: {optimal_loss:.4f}")
    
    if optimal_loss <= target_performance:
        print(f"✓ Target achievable with Chinchilla-optimal allocation!")
        
        # Find minimum compute needed for target performance
        # Binary search over compute budgets
        low_compute = 1e18
        high_compute = max_compute
        
        for _ in range(20):  # Binary search iterations
            mid_compute = (low_compute + high_compute) / 2
            mid_N, mid_D = scaling.optimal_params_chinchilla(mid_compute)
            mid_loss = scaling.loss_parametric(mid_N, mid_D)
            
            if mid_loss <= target_performance:
                high_compute = mid_compute
            else:
                low_compute = mid_compute
        
        required_compute = high_compute
        req_N, req_D = scaling.optimal_params_chinchilla(required_compute)
        
        print(f"\nMinimum compute needed:")
        print(f"Compute: {required_compute:.1e} FLOPs")
        print(f"Parameters: {req_N:.1e}")
        print(f"Tokens: {req_D:.1e}")
        print(f"Efficiency: {required_compute/max_compute:.1%} of max compute")
        
    else:
        print(f"✗ Target not achievable with available compute.")
        print(f"Need {target_performance/optimal_loss:.2f}x better performance.")
        print(f"Consider:")
        print(f"  - Increasing compute budget")
        print(f"  - Improving model architecture")
        print(f"  - Better data quality")

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