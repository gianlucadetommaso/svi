import unittest

import torch
import torch.nn as nn

from svi.algorithms.svgd import SVGD
from svi.kernels.rbf import RBFKernel
from svi.module import SVIModule


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestSVIModule(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.kernel = RBFKernel()
        self.svgd = SVGD(self.kernel)

    def test_init_freezing(self):
        # By default freeze_previous_layers=True
        # fc2 (last layer) should have grad, fc1 should not
        module = SVIModule(
            model=self.model,
            svi_algorithm=self.svgd,
            num_particles=4,
            freeze_previous_layers=True,
        )

        self.assertTrue(module.base_model.fc2.weight.requires_grad)
        self.assertTrue(module.base_model.fc2.bias.requires_grad)
        self.assertFalse(module.base_model.fc1.weight.requires_grad)

    def test_forward_different_particles(self):
        module = SVIModule(
            model=self.model,
            svi_algorithm=self.svgd,
            num_particles=2,
            freeze_previous_layers=False,
        )

        # Ensure particles are different (random perturbation)
        p1 = module.particles[0].clone()
        p2 = module.particles[1].clone()
        self.assertFalse(torch.allclose(p1, p2))

        x = torch.randn(1, 10)

        # Forward with particle 0
        y0 = module(x, 0)
        # Forward with particle 1
        y1 = module(x, 1)

        self.assertFalse(torch.allclose(y0, y1))

    def test_training_step(self):
        module = SVIModule(
            model=self.model,
            svi_algorithm=self.svgd,
            num_particles=2,
            freeze_previous_layers=True,
        )

        # Configure optimizer
        optimizer = module.configure_optimizers()

        # Mock optimizers() method to return the optimizer
        module.optimizers = lambda: optimizer

        x = torch.randn(5, 10)
        y = torch.randn(5, 1)  # MSE target
        batch = (x, y)

        # Just run training step to see if it crashes
        loss = module.training_step(batch, 0)
        self.assertIn("loss", loss)
