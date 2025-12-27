import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from svi.module import SVIModule
from svi.algorithms.svgd import SVGD
from svi.kernels.rbf import RBFKernel

# 1. Define a simple model (or load a pretrained one)
class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, 20),
            nn.GELU(),
        )
        self.head = nn.Linear(20, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

def main():
    pl.seed_everything(42)
    
    # 2. Prepare Data
    # Generate synthetic cubic data: y = x^3 + noise
    X = torch.linspace(-3, 3, 100).unsqueeze(1)
    Y = X.pow(3) + 3 * torch.randn_like(X)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # 3. Instantiate Base Model
    model = SimpleRegressionModel()
    
    # Pre-training could happen here, or we use random weights.
    
    # 4. Configure SVI
    # Use RBF Kernel
    kernel = RBFKernel() # Uses median heuristic by default
    
    # Use SVGD Algorithm
    svgd = SVGD(kernel)
    
    # 5. Create SVI Lightning Module
    # We allow full training (freeze_previous_layers=False) because with a small net and random init,
    # freezing the feature extractor at random weights makes it impossible to fit the data.
    # Alternatively, we could pre-train. But for this demo, let's train everything.
    svi_module = SVIModule(
        model=model,
        svi_algorithm=svgd,
        num_particles=100,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.1},
        freeze_previous_layers=False 
    )
    
    # 6. Train using PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=100)
    trainer.fit(svi_module, dataloader)
    
    print("Training complete!")
    
    # 7. Inference / Visualization
    # Make predictions with all particles
    X_test = torch.linspace(-5, 5, 100).unsqueeze(1)
    preds = []
    
    with torch.no_grad():
        for i in range(svi_module._num_particles):
            # The module forward helper takes (x, particle_idx)
            y_pred = svi_module(X_test, i)
            preds.append(y_pred)
            
    preds = torch.stack(preds).squeeze(2) # (n_particles, n_samples)
    
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    
    print(f"Test Input range: [-4, 4]")
    print(f"Mean prediction at 0.0: {mean_pred[-1]:.4f} (Expected near 0)")
    print(f"Std dev of predictions at 0.0: {std_pred[-1]:.4f}")

    # 8. Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X.numpy(), Y.numpy(), color='black', alpha=0.5, label='Training Data')
    
    # Plot individual particle predictions
    X_np = X_test.squeeze().numpy()
    for i in range(len(preds)):
        plt.plot(X_np, preds[i].numpy(), color='blue', alpha=0.1)
        
    # Plot mean prediction
    plt.plot(X_np, mean_pred.numpy(), color='red', linewidth=2, label='Mean Prediction')
    
    # Plot +/- 2 std dev
    plt.fill_between(
        X_np,
        (mean_pred - 2 * std_pred).numpy(),
        (mean_pred + 2 * std_pred).numpy(),
        color='red', alpha=0.2, label='Uncertainty (2 std)'
    )
    
    # Plot ground truth function y = x^3
    plt.plot(X_np, X_np**3, 'g--', label='Ground Truth $y=x^3$')
    
    plt.title('SVGD Regression Results')
    plt.legend()
    plt.ylim(-35, 35)
    plt.savefig('svi_demo_plot.png')
    print("Plot saved to svi_demo_plot.png")

if __name__ == "__main__":
    main()
