import torch
import unittest
from svi.algorithms.svgd import SVGD
from svi.kernels.rbf import RBFKernel

class TestSVGD(unittest.TestCase):
    def test_direction_shape(self):
        n_particles = 10
        dim = 5
        X = torch.randn(n_particles, dim)
        score = torch.randn(n_particles, dim)
        
        kernel = RBFKernel()
        svgd = SVGD(kernel)
        
        direction = svgd.compute_direction(X, score)
        self.assertEqual(direction.shape, (n_particles, dim))

    def test_single_particle(self):
        # 1 particle
        X = torch.tensor([[0.0]])
        score = torch.tensor([[1.0]]) # grad log p
        
        # For 1 particle:
        # phi(x) = k(x,x) * score + grad_x k(x,x)
        # k(x,x) = 1
        # grad_x k(x,x) = 0 (peak of gaussian)
        # phi = score
        
        kernel = RBFKernel()
        svgd = SVGD(kernel)
        
        direction = svgd.compute_direction(X, score)
        self.assertTrue(torch.allclose(direction, score))
        
    def test_svgd_is_sgd_for_single_particle(self):
        # With 1 particle, SVGD direction should be exactly the score (gradient of log prob)
        # Because K(x,x)=1 and grad_x K(x,x)=0
        # The update direction is solely determined by the score.
        
        # We can simulate an SGD step and an SVGD step and compare.
        # Note: SVGD uses direction = score.
        # SGD update: theta_new = theta + lr * score (gradient ascent on log prob)
        # SVGD update: theta_new = theta + lr * phi(theta)
        
        # Let's verify for a random vector
        dim = 5
        X = torch.randn(1, dim)
        score = torch.randn(1, dim)
        
        kernel = RBFKernel()
        svgd = SVGD(kernel)
        
        # SVGD direction
        direction_svgd = svgd.compute_direction(X, score)
        
        # Expected SGD direction (just the score)
        direction_sgd = score
        
        self.assertTrue(torch.allclose(direction_svgd, direction_sgd), 
                        "SVGD with 1 particle should be equivalent to SGD (direction == score)")

    def test_missing_score_error(self):
        n_particles = 5
        dim = 2
        X = torch.randn(n_particles, dim)
        
        kernel = RBFKernel()
        svgd = SVGD(kernel)
        
        with self.assertRaises(ValueError):
            svgd.compute_direction(X, score=None)
