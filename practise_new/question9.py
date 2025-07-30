"""
Question 9: Variational Autoencoder (VAE) with Custom Prior (Hard)

Implement a Variational Autoencoder with a custom mixture of Gaussians prior instead of
the standard unit Gaussian. This requires implementing the reparameterization trick,
custom KL divergence calculation, and proper handling of the mixture prior.

Key concepts:
1. Encoder network that outputs mean and log-variance
2. Reparameterization trick for differentiable sampling
3. Decoder network for reconstruction
4. Custom KL divergence for mixture of Gaussians prior
5. Evidence Lower Bound (ELBO) loss

Requirements:
- Use mixture of K Gaussians as prior: p(z) = Σ π_k N(μ_k, Σ_k)
- Implement proper KL divergence calculation
- Support both 2D and higher dimensional latent spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    """Encoder network that outputs posterior parameters"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # TODO: Create encoder layers
        # Output both mean and log_variance for latent distribution
        pass
    
    def forward(self, x):
        # TODO: Forward pass to get posterior parameters
        # Return: (mean, log_var) both of shape (batch_size, latent_dim)
        pass

class Decoder(nn.Module):
    """Decoder network for reconstruction"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # TODO: Create decoder layers
        pass
    
    def forward(self, z):
        # TODO: Forward pass from latent code to reconstruction
        pass

class MixtureOfGaussiansPrior(nn.Module):
    """Mixture of Gaussians prior distribution"""
    def __init__(self, n_components, latent_dim):
        super(MixtureOfGaussiansPrior, self).__init__()
        # TODO: Initialize mixture parameters
        # - component weights (π_k)
        # - component means (μ_k) 
        # - component covariances (Σ_k)
        pass
    
    def log_prob(self, z):
        """Compute log probability under mixture prior"""
        # TODO: Implement log p(z) for mixture of Gaussians
        # Use logsumexp for numerical stability
        pass
    
    def sample(self, n_samples):
        """Sample from mixture prior"""
        # TODO: Sample from mixture distribution
        # 1. Sample component indices according to mixture weights
        # 2. Sample from selected Gaussian components
        pass

class VAE(nn.Module):
    """Variational Autoencoder with mixture prior"""
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components=3):
        super(VAE, self).__init__()
        # TODO: Initialize encoder, decoder, and prior
        pass
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for differentiable sampling"""
        # TODO: Implement reparameterization trick
        # z = μ + σ * ε, where ε ~ N(0, I)
        pass
    
    def forward(self, x):
        """Full forward pass through VAE"""
        # TODO: Implement VAE forward pass
        # 1. Encode to get posterior parameters
        # 2. Sample latent code using reparameterization
        # 3. Decode to get reconstruction
        # Return: (reconstruction, mu, log_var, z)
        pass
    
    def compute_kl_divergence(self, mu, log_var):
        """Compute KL divergence between posterior and mixture prior"""
        # TODO: Implement KL(q(z|x) || p(z)) for mixture prior
        # This is more complex than standard Gaussian KL divergence
        # Use Monte Carlo estimation if analytical form is difficult
        pass
    
    def elbo_loss(self, x, reconstruction, mu, log_var):
        """Compute Evidence Lower Bound (ELBO) loss"""
        # TODO: Compute ELBO = reconstruction_loss + KL_divergence
        # Use appropriate reconstruction loss (MSE, BCE, etc.)
        pass
    
    def generate(self, n_samples):
        """Generate new samples from the model"""
        # TODO: Sample from prior and decode
        pass

def train_vae(model, dataloader, optimizer, device, epochs=10):
    """Training loop for VAE"""
    # TODO: Implement training loop
    # 1. Forward pass through VAE
    # 2. Compute ELBO loss
    # 3. Backpropagation and optimization
    # 4. Track reconstruction and KL losses separately
    pass

def evaluate_vae(model, test_loader, device):
    """Evaluate VAE on test data"""
    # TODO: Compute test loss and other metrics
    # Could include:
    # - Reconstruction quality
    # - Log-likelihood estimation
    # - Latent space visualization (if 2D)
    pass

# Test your implementation
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create VAE model
    input_dim = 784  # MNIST: 28x28
    hidden_dim = 400
    latent_dim = 2
    n_components = 3
    
    model = VAE(input_dim, hidden_dim, latent_dim, n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim).to(device)
    
    recon, mu, log_var, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent log_var shape: {log_var.shape}")
    print(f"Latent sample shape: {z.shape}")
    
    # Test loss computation
    loss = model.elbo_loss(x, recon, mu, log_var)
    print(f"ELBO loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(n_samples=10)
    print(f"Generated samples shape: {generated.shape}")
    
    # Test prior sampling
    prior_samples = model.prior.sample(100)
    print(f"Prior samples shape: {prior_samples.shape}")
    
    print("VAE implementation test completed!")