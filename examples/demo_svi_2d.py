import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from svi.algorithms.svgd import SVGD
from svi.kernels.rbf import RBFKernel
from svi.module import SVIModule


# 1. Define a simple 2D regression model
class RegressionModel2D(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (x1, x2), Output: y
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
        )
        self.head = nn.Linear(20, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)


def main():
    pl.seed_everything(42)

    # 2. Prepare Data (2D input)
    # Generate data: y = sin(x1) + cos(x2) + noise
    n_samples = 1000
    X = torch.rand(n_samples, 2) * 6 - 3  # Range [-3, 3]
    Y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + 0.1 * torch.randn(n_samples)
    Y = Y.unsqueeze(1)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Instantiate Base Model
    model = RegressionModel2D()

    # 4. Configure SVI
    kernel = RBFKernel()
    svgd = SVGD(kernel)

    # 5. Create SVI Lightning Module
    # Use multiple particles to capture uncertainty
    svi_module = SVIModule(
        model=model,
        svi_algorithm=svgd,
        num_particles=10,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.05},
        freeze_previous_layers=False,
        prior_std=1.0,  # Use a prior
    )

    # 6. Train using PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=10)
    trainer.fit(svi_module, dataloader)

    print("Training complete!")

    # 7. Visualization
    # Create a meshgrid for plotting
    n_grid = 50
    x1 = np.linspace(-3, 3, n_grid)
    x2 = np.linspace(-3, 3, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = torch.tensor(
        np.stack([X1.flatten(), X2.flatten()], axis=1), dtype=torch.float32
    )

    preds = []
    with torch.no_grad():
        for i in range(svi_module._num_particles):
            y_pred = svi_module(X_grid, i)
            preds.append(y_pred)

    preds = torch.stack(preds).squeeze(2)  # (n_particles, n_grid*n_grid)

    mean_pred = preds.mean(dim=0).numpy().reshape(n_grid, n_grid)
    std_pred = preds.std(dim=0).numpy().reshape(n_grid, n_grid)

    # Ground Truth
    Y_true = np.sin(X1) + np.cos(X2)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Ground Truth with Particles
    im1 = axes[0].contourf(X1, X2, Y_true, levels=20, cmap="viridis")
    axes[0].set_title("Ground Truth & Training Data")
    fig.colorbar(im1, ax=axes[0])
    # Overlay training data points (particles effectively learn from these)
    # Since this is regression, "particles" are model parameters, not points in input space.
    # But usually "visualize particles" in regression means visualize the PREDICTIONS of the particles.
    # If the user means "particles" as in the input data distribution, we plot X.
    # If the user means "particles" as in the model parameters, we can't easily plot them in 2D input space.
    # Assuming user wants to see the training data distribution on top of ground truth:
    axes[0].scatter(
        X[:, 0].numpy(), X[:, 1].numpy(), c="k", s=1, alpha=0.3, label="Training Data"
    )
    axes[0].legend()

    # Plot 2: Mean Prediction
    im2 = axes[1].contourf(X1, X2, mean_pred, levels=20, cmap="viridis")
    axes[1].set_title("Mean Prediction (10 particles)")
    fig.colorbar(im2, ax=axes[1])

    # Plot 3: Uncertainty (Std Dev)
    im3 = axes[2].contourf(X1, X2, std_pred, levels=20, cmap="plasma")
    axes[2].set_title("Predictive Uncertainty (Std Dev)")
    fig.colorbar(im3, ax=axes[2])

    # Overlay training points on uncertainty plot
    # axes[2].scatter(X[:, 0].numpy(), X[:, 1].numpy(), c='k', s=1, alpha=0.3)

    plt.tight_layout()
    plt.savefig("svi_2d_regression.png")
    print("Plot saved to svi_2d_regression.png")


if __name__ == "__main__":
    main()
