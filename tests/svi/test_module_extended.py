import torch
import torch.nn as nn
import unittest
from svi.module import SVIModule
from svi.algorithms.svgd import SVGD
from svi.kernels.rbf import RBFKernel

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class TestSVIModuleExtended(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.kernel = RBFKernel()
        self.svgd = SVGD(self.kernel)
        self.module = SVIModule(
            model=self.model,
            svi_algorithm=self.svgd,
            num_particles=2,
            freeze_previous_layers=False
        )

    def test_compute_loss_mse(self):
        # Float targets -> MSE
        y_hat = torch.tensor([[1.0], [2.0]])
        y = torch.tensor([[1.0], [2.0]])
        loss = self.module._compute_loss(y_hat, y)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.0)))
        
        y_wrong = torch.tensor([[0.0], [0.0]])
        loss_wrong = self.module._compute_loss(y_hat, y_wrong)
        # MSE: ((1-0)^2 + (2-0)^2) / 2 = 2.5
        self.assertTrue(torch.isclose(loss_wrong, torch.tensor(2.5)))

    def test_compute_loss_cross_entropy(self):
        # Long/Int targets -> Cross Entropy
        # 2 classes, batch size 2
        y_hat = torch.tensor([[10.0, -10.0], [-10.0, 10.0]]) # Strong prediction for class 0 then class 1
        y = torch.tensor([0, 1], dtype=torch.long)
        
        loss = self.module._compute_loss(y_hat, y)
        # Loss should be near 0
        self.assertTrue(loss < 0.01)

    def test_configure_optimizers(self):
        optimizer = self.module.configure_optimizers()
        
        # Check that the optimizer is valid
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        
        # Check that it optimizes the particles
        param_groups = optimizer.param_groups
        self.assertEqual(len(param_groups), 1)
        
        params = param_groups[0]['params']
        self.assertEqual(len(params), 1)
        
        # Ensure the parameter being optimized is indeed self.module.particles
        self.assertIs(params[0], self.module.particles)
