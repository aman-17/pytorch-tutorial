"""
Question 4: Power Law Fitting and Scaling Analysis (Medium-Hard)

Implement tools to fit power laws to empirical scaling data and analyze
deviations from theoretical predictions. This is crucial for understanding
when scaling laws break down and for extrapolating to larger scales.

You'll work with synthetic training data that follows power law relationships
with noise, and implement robust fitting techniques used in scaling law research.

Your task: Implement power law fitting, uncertainty quantification, and 
extrapolation tools used in real scaling law analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings

@dataclass
class PowerLawFit:
    """Results from power law fitting"""
    exponent: float
    coefficient: float
    r_squared: float
    exponent_std: float
    coefficient_std: float
    residuals: np.ndarray

class ScalingAnalyzer:
    def __init__(self):
        # Common scaling law exponents for reference
        self.kaplan_exponents = {
            'parameters': -0.076,
            'data': -0.095, 
            'compute': -0.050
        }
        
    def generate_synthetic_data(self, x_range, true_exponent, true_coeff, 
                              noise_level=0.1, n_points=50):
        """
        Generate synthetic scaling data with realistic noise.
        y = true_coeff * x^true_exponent + noise
        """
        # Generate x values (log-spaced)
        x_min, x_max = x_range
        x = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
        
        # Compute true y values using power law
        y_true = true_coeff * np.power(x, true_exponent)
        
        # Add realistic noise (log-normal or multiplicative)
        # Use multiplicative noise: y = y_true * exp(normal_noise)
        noise = np.random.lognormal(0, noise_level, n_points)
        y = y_true * noise
        
        return x, y
    
    def fit_power_law_linear(self, x, y):
        """
        Fit power law using linear regression in log space.
        log(y) = log(coeff) + exponent * log(x)
        """
        # Take logarithms of x and y
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Use scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        # Convert back to original space
        exponent = slope
        coefficient = np.exp(intercept)
        
        # Calculate R^2 and standard errors
        r_squared = r_value ** 2
        exponent_std = std_err
        
        # Estimate coefficient standard error using delta method
        coefficient_std = coefficient * std_err  # Rough approximation
        
        # Calculate residuals in original space
        y_pred = coefficient * np.power(x, exponent)
        residuals = y - y_pred
        
        return PowerLawFit(
            exponent=exponent,
            coefficient=coefficient,
            r_squared=r_squared,
            exponent_std=exponent_std,
            coefficient_std=coefficient_std,
            residuals=residuals
        )
    
    def fit_power_law_nonlinear(self, x, y, initial_guess=None):
        """
        Fit power law using nonlinear least squares.
        More robust to outliers than linear method.
        """
        def power_law(x, coeff, exponent):
            return coeff * np.power(x, exponent)
        
        # Set reasonable initial guess if not provided
        if initial_guess is None:
            # Use linear fit as initial guess
            linear_fit = self.fit_power_law_linear(x, y)
            initial_guess = [linear_fit.coefficient, linear_fit.exponent]
        
        try:
            # Use scipy.optimize.curve_fit with bounds
            popt, pcov = optimize.curve_fit(
                power_law, x, y,
                p0=initial_guess,
                bounds=([1e-10, -np.inf], [np.inf, np.inf]),  # coefficient > 0
                maxfev=10000
            )
            
            coefficient, exponent = popt
            param_errors = np.sqrt(np.diag(pcov))
            coefficient_std, exponent_std = param_errors
            
            # Calculate R^2 and parameter uncertainties
            y_pred = power_law(x, coefficient, exponent)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            residuals = y - y_pred
            
            return PowerLawFit(
                exponent=exponent,
                coefficient=coefficient,
                r_squared=r_squared,
                exponent_std=exponent_std,
                coefficient_std=coefficient_std,
                residuals=residuals
            )
            
        except Exception as e:
            # Handle fitting failures gracefully
            print(f"Nonlinear fitting failed: {e}")
            print("Falling back to linear fit...")
            return self.fit_power_law_linear(x, y)
    
    def detect_scaling_regime_breaks(self, x, y, min_points=10):
        """
        Detect points where scaling law breaks down using changepoint detection.
        Returns indices where scaling regime changes.
        """
        # Implement simplified sliding window analysis
        # For each possible breakpoint, fit power laws to segments
        # Find breakpoints that minimize total fitting error
        
        breakpoints = []
        n = len(x)
        
        if n < 2 * min_points:
            return breakpoints
        
        # Fit single power law for reference
        single_fit = self.fit_power_law_linear(x, y)
        single_error = np.sum(single_fit.residuals ** 2)
        
        best_improvement = 0
        best_breakpoint = None
        
        # Scan through potential breakpoints
        for i in range(min_points, n - min_points):
            try:
                # Fit power laws to left and right segments
                left_fit = self.fit_power_law_linear(x[:i], y[:i])
                right_fit = self.fit_power_law_linear(x[i:], y[i:])
                
                # Calculate total error
                left_error = np.sum(left_fit.residuals ** 2)
                right_error = np.sum(right_fit.residuals ** 2)
                total_error = left_error + right_error
                
                # Check improvement
                improvement = single_error - total_error
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_breakpoint = i
                    
            except:
                continue
        
        # Add breakpoint if improvement is significant (simple heuristic)
        if best_improvement > 0.1 * single_error:
            breakpoints.append(best_breakpoint)
        
        return breakpoints
    
    def extrapolate_with_uncertainty(self, fit_result, x_new, confidence=0.95):
        """
        Extrapolate power law with uncertainty bounds.
        """
        # Use fitted parameters to predict at x_new
        y_pred = fit_result.coefficient * np.power(x_new, fit_result.exponent)
        
        # Simplified uncertainty propagation
        # In practice, this would use the full covariance matrix
        coeff_error = fit_result.coefficient_std
        exp_error = fit_result.exponent_std
        
        # Rough approximation of prediction uncertainty
        relative_coeff_error = coeff_error / fit_result.coefficient
        relative_exp_error = exp_error * np.log(x_new)  # Simplified
        
        relative_total_error = np.sqrt(relative_coeff_error**2 + relative_exp_error**2)
        
        # Calculate confidence intervals (assuming normal distribution)
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, df=len(fit_result.residuals) - 2)
        
        error_bounds = y_pred * relative_total_error * t_value
        lower_bound = y_pred - error_bounds
        upper_bound = y_pred + error_bounds
        
        return {
            'prediction': y_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence
        }
    
    def compare_scaling_laws(self, x, y, theoretical_exponents):
        """
        Compare empirical data against theoretical scaling law predictions.
        """
        results = {}
        
        # Fit empirical power law
        empirical_fit = self.fit_power_law_nonlinear(x, y)
        results['empirical'] = empirical_fit
        
        # For each theoretical exponent, fit coefficient only
        for name, theory_exp in theoretical_exponents.items():
            # Fix exponent, fit coefficient using least squares
            # y = coeff * x^theory_exp, so coeff = mean(y / x^theory_exp)
            
            x_powered = np.power(x, theory_exp)
            coeff_optimal = np.sum(y * x_powered) / np.sum(x_powered * x_powered)
            
            # Calculate predictions and goodness of fit
            y_pred = coeff_optimal * x_powered
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store results
            theoretical_fit = PowerLawFit(
                exponent=theory_exp,
                coefficient=coeff_optimal,
                r_squared=r_squared,
                exponent_std=0.0,  # Fixed
                coefficient_std=0.0,  # Not calculated for simplicity
                residuals=residuals
            )
            
            results[name] = theoretical_fit
            
        return results
    
    def analyze_residuals(self, x, y, fit_result):
        """
        Analyze residuals to check for systematic deviations.
        """
        # TODO: Calculate residuals
        predicted = fit_result.coefficient * np.power(x, fit_result.exponent)
        residuals = y - predicted
        
        analysis = {
            'residuals': residuals,
            'relative_residuals': residuals / predicted,
            'mean_abs_error': np.mean(np.abs(residuals)),
            'rmse': np.sqrt(np.mean(residuals**2))
        }
        
        # TODO: Test for systematic patterns in residuals
        # TODO: Check for autocorrelation
        # TODO: Test normality of residuals
        
        return analysis

def simulate_training_run(model_sizes, max_tokens_per_size, base_lr=1e-3):
    """
    Simulate training runs for different model sizes to generate scaling data.
    """
    results = {
        'model_sizes': model_sizes,
        'final_losses': [],
        'compute_used': [],
        'optimal_tokens': []
    }
    
    for N in model_sizes:
        # TODO: Simulate training with realistic loss curves
        # TODO: Use Chinchilla-like scaling to determine optimal training length
        # TODO: Add realistic noise and variation
        
        # Simplified simulation - replace with more realistic model
        optimal_D = max_tokens_per_size
        compute = 6 * N * optimal_D  # FLOPs approximation
        
        # Kaplan-like loss with noise
        loss = 1.0 + 8e6 / N**0.076 + 5e6 / optimal_D**0.095
        loss *= np.random.lognormal(0, 0.05)  # Add noise
        
        results['final_losses'].append(loss)
        results['compute_used'].append(compute)
        results['optimal_tokens'].append(optimal_D)
    
    return results

def plot_scaling_analysis(x, y, fit_results, extrapolation_range=None):
    """
    Create comprehensive scaling analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO: Plot 1 - Data with fits
    ax1 = axes[0, 0]
    # TODO: Plot original data points
    # TODO: Plot fitted power law
    # TODO: Plot confidence intervals if available
    # TODO: Add legend and labels
    
    # TODO: Plot 2 - Residuals
    ax2 = axes[0, 1] 
    # TODO: Plot residuals vs x
    # TODO: Add zero line
    # TODO: Check for patterns
    
    # TODO: Plot 3 - Log-log plot
    ax3 = axes[1, 0]
    # TODO: Plot in log-log space to check linearity
    # TODO: Show deviations from power law
    
    # TODO: Plot 4 - Extrapolation
    ax4 = axes[1, 1]
    # TODO: Show extrapolation with uncertainty bounds
    # TODO: Compare with different extrapolation methods
    
    plt.tight_layout()
    return fig

def scaling_law_study():
    """
    Comprehensive study comparing different scaling law approaches.
    """
    analyzer = ScalingAnalyzer()
    
    print("=== Scaling Law Analysis Study ===")
    
    # TODO: Generate synthetic data with known power law
    # TODO: Add realistic noise and potential regime breaks
    
    # Test Case 1: Clean power law
    print("\n1. Testing with clean power law data...")
    x1 = np.logspace(6, 10, 30)  # 1M to 10B parameters
    true_exp = -0.076
    y1 = 2.0 * np.power(x1, true_exp) + np.random.normal(0, 0.01, len(x1))
    
    # TODO: Fit using both methods
    # linear_fit = analyzer.fit_power_law_linear(x1, y1)
    # nonlinear_fit = analyzer.fit_power_law_nonlinear(x1, y1)
    
    # TODO: Compare results
    
    # Test Case 2: Data with regime break
    print("\n2. Testing with scaling regime break...")
    # TODO: Generate data that follows different power laws in different regimes
    
    # Test Case 3: Comparison with theoretical predictions
    print("\n3. Comparing with theoretical scaling laws...")
    theoretical_exps = {
        'kaplan_params': -0.076,
        'kaplan_data': -0.095,
        'chinchilla': -0.5
    }
    
    # TODO: Compare empirical fit with theoretical predictions
    # comparison = analyzer.compare_scaling_laws(x1, y1, theoretical_exps)
    
    # TODO: Generate plots
    # fig = plot_scaling_analysis(x1, y1, [linear_fit, nonlinear_fit])

# Test your implementation
if __name__ == "__main__":
    print("=== Testing Scaling Analysis Tools ===")
    
    analyzer = ScalingAnalyzer()
    
    # Generate test data
    x_test, y_test = analyzer.generate_synthetic_data(
        x_range=(1e7, 1e10),
        true_exponent=-0.08,
        true_coeff=3.0,
        noise_level=0.15,
        n_points=25
    )
    
    print("Generated synthetic scaling data")
    
    # Test fitting methods
    print("\n=== Testing Fitting Methods ===")
    
    if y_test is not None:
        linear_fit = analyzer.fit_power_law_linear(x_test, y_test)
        nonlinear_fit = analyzer.fit_power_law_nonlinear(x_test, y_test)
        
        print(f"Linear fit - Exponent: {linear_fit.exponent:.4f} ± {linear_fit.exponent_std:.4f}")
        print(f"Nonlinear fit - Exponent: {nonlinear_fit.exponent:.4f} ± {nonlinear_fit.exponent_std:.4f}")
        
        # Test regime break detection
        print(f"\n=== Testing Regime Break Detection ===")
        breakpoints = analyzer.detect_scaling_regime_breaks(x_test, y_test)
        print(f"Detected {len(breakpoints)} potential regime breaks")
        
        # Test extrapolation
        print(f"\n=== Testing Extrapolation ===")
        x_future = np.array([1e11, 1e12])  # 100B, 1T parameters
        predictions = analyzer.extrapolate_with_uncertainty(nonlinear_fit, x_future)
        
        if predictions is not None:
            print(f"Extrapolated predictions generated")
    
    # Run comprehensive study
    print(f"\n=== Running Comprehensive Study ===")
    scaling_law_study()
    
    print("Scaling analysis complete!")