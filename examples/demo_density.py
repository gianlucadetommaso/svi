import torch
import torch.nn as nn
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector

from svi.module import SVIModule
from svi.algorithms.svgd import SVGD
from svi.kernels.rbf import RBFKernel

# Target Distribution
def target_log_prob(x):
    # Mixture of two Gaussians in 2D
    device = x.device
    mix = D.Categorical(torch.tensor([0.3, 0.7], device=device))
    comp = D.Independent(D.Normal(
        torch.tensor([[-2.0, -2.0], [2.0, 2.0]], device=device), 
        torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device)
    ), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    return gmm.log_prob(x)

# 1. Define a "Model" that represents the structure of a single particle
# For density estimation in 2D, a particle is just a 2D vector.
class PointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(2)) # Dummy initialization

    def forward(self, x):
        return self.pos

def main():
    pl.seed_everything(42)
    
    # 3. Setup
    n_particles = 1000
    model = PointModel() # defines the shape of one particle
    
    kernel = RBFKernel()
    svgd = SVGD(kernel)
    
    svi_module = SVIModule(
        model=model,
        svi_algorithm=svgd,
        num_particles=n_particles,
        target_log_prob_fn=target_log_prob,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.1},
        prior_std=None # Target density is fully defined in target_log_prob
    )
    
    # Initialize particles uniformly [-4, 4] for better demo visualization
    with torch.no_grad():
        svi_module.particles.data = torch.rand(n_particles, 2) * 8 - 4

    # Dummy DataLoader (Lightning requires it)
    # Reduce dataset size to 1, so 1 epoch = 1 step.
    dummy_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

    # 4. Train
    print("Running SVIModule for Density Estimation...")
    # Train for 100 epochs, effectively 100 optimization steps.
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10, enable_checkpointing=False, logger=False, enable_progress_bar=False)
    trainer.fit(svi_module, dummy_loader)
    
    print("Optimization complete!")

    # 5. Visualization
    plt.figure(figsize=(8, 6))
    
    # Plot Target Density
    n_grid = 100
    x = np.linspace(-6, 6, n_grid)
    y = np.linspace(-6, 6, n_grid)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        log_prob = target_log_prob(grid)
        prob = torch.exp(log_prob).reshape(n_grid, n_grid)
    
    plt.contourf(X, Y, prob, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Probability Density')
    
    # Plot Particles
    particles_np = svi_module.particles.detach().numpy()
    plt.scatter(particles_np[:, 0], particles_np[:, 1], c='red', s=20, edgecolors='white', label='SVGD Particles')
    
    plt.title('SVGD Particles (SVIModule) approximating Target Density')
    plt.legend()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    
    plt.savefig('svi_density_2d_module.png')
    print("Plot saved to svi_density_2d_module.png")

if __name__ == "__main__":
    main()
