import torch
import unittest
from svi.kernels.rbf import RBFKernel

class TestRBFKernel(unittest.TestCase):
    def test_forward_shape(self):
        n_particles = 10
        dim = 5
        X = torch.randn(n_particles, dim)
        
        kernel = RBFKernel()
        K, grad_K = kernel.forward(X)
        
        self.assertEqual(K.shape, (n_particles, n_particles))
        self.assertEqual(grad_K.shape, (n_particles, dim))
        
    def test_values_identity(self):
        # With fixed bandwidth, we can check values
        X = torch.tensor([[0.0], [1.0]])
        h = 1.0
        kernel = RBFKernel(bandwidth=h)
        K, grad_K = kernel.forward(X)
        
        # dist_sq = [[0, 1], [1, 0]]
        # gamma = 1/(1+1e-8) approx 1
        # K = exp(-dist_sq) -> [[1, exp(-1)], [exp(-1), 1]]
        
        expected_K = torch.exp(-torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        self.assertTrue(torch.allclose(K, expected_K, atol=1e-6))
        
        # Gradient check
        # grad_K[0] = sum_j 2 gamma k(x_0, x_j) (x_0 - x_j)
        # j=0: 2 * 1 * 1 * (0-0) = 0
        # j=1: 2 * 1 * exp(-1) * (0-1) = -2 exp(-1)
        # grad_K[0] approx -0.7357
        
        val = 2.0 * 1.0 * torch.exp(torch.tensor(-1.0)) * (-1.0)
        self.assertTrue(torch.allclose(grad_K[0], val, atol=1e-6))

    def test_median_heuristic(self):
        # Test median heuristic calculation
        X = torch.tensor([[0.0], [1.0], [2.0]])
        # Distances: |0-1|=1, |0-2|=2, |1-2|=1
        # Squared distances: 1, 4, 1
        # Median of [1, 4, 1] is 1
        # N=3
        # h = 1 / log(3)
        
        kernel = RBFKernel(bandwidth=None)
        # We need to access private method or infer from result, 
        # but _median_heuristic is a method we can test directly if we want
        # or we check if bandwidth is computed correctly during forward pass.
        
        # Let's test _median_heuristic directly since we are in white-box testing mode
        dist_sq = torch.tensor([1.0, 4.0, 1.0])
        h = kernel._median_heuristic(dist_sq)
        expected_h = 1.0 / (torch.log(torch.tensor(3.0)) + 1e-8)
        self.assertTrue(torch.allclose(h, expected_h))
        
    def test_median_heuristic_single_particle(self):
        # Edge case: 1 particle
        dist_sq = torch.tensor([0.0]) # Distance to self is 0
        kernel = RBFKernel()
        h = kernel._median_heuristic(dist_sq)
        self.assertEqual(h.item(), 1.0)
        
    def test_median_heuristic_zero_median(self):
        # Edge case: particles are identical
        dist_sq = torch.tensor([0.0, 0.0, 0.0])
        kernel = RBFKernel()
        h = kernel._median_heuristic(dist_sq)
        self.assertEqual(h.item(), 1.0)
