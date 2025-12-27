import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

# 1. Define a simple model (same as in demo_svi.py)
class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
        )
        self.head = nn.Linear(20, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

# Standard LightningModule for SGD training
class SGDModule(pl.LightningModule):
    def __init__(self, model, lr=0.1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Using Adam as in the SVI demo for fair comparison, 
        # though the file is named demo_sgd, "SGD" here refers to standard point estimation
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

def main():
    pl.seed_everything(42)
    
    # 2. Prepare Data (Same as demo_svi.py)
    # Generate synthetic cubic data: y = x^3 + noise
    X = torch.linspace(-3, 3, 1000).unsqueeze(1)
    Y = X.pow(3) + 3 * torch.randn_like(X)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # 3. Instantiate Base Model
    model = SimpleRegressionModel()
    
    # 4. Create SGD Lightning Module
    sgd_module = SGDModule(model=model, lr=0.1)
    
    # 5. Train using PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=100)
    trainer.fit(sgd_module, dataloader)
    
    print("Training complete!")
    
    # 6. Inference / Visualization
    X_test = torch.linspace(-4, 4, 100).unsqueeze(1)
    
    with torch.no_grad():
        preds = sgd_module(X_test)
            
    # For SGD, we have a single point estimate (no uncertainty from particles)
    mean_pred = preds.squeeze().numpy()
    X_np = X_test.squeeze().numpy()
    
    # 7. Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X.numpy(), Y.numpy(), color='black', alpha=0.5, label='Training Data')
    
    # Plot prediction
    plt.plot(X_np, mean_pred, color='red', linewidth=2, label='Prediction')
    
    # Plot ground truth function y = x^3
    plt.plot(X_np, X_np**3, 'g--', label='Ground Truth $y=x^3$')
    
    plt.title('SGD (Point Estimate) Regression Results')
    plt.legend()
    plt.ylim(-30, 30)
    plt.savefig('sgd_demo_plot.png')
    print("Plot saved to sgd_demo_plot.png")

if __name__ == "__main__":
    main()
