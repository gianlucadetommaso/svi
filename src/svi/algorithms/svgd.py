import torch
from torch import Tensor
from svi.algorithms.base import SVIAlgorithm

class SVGD(SVIAlgorithm):
    """
    Stein Variational Gradient Descent (SVGD) implementation.
    """
    
    def compute_direction(self, particles: Tensor, score: Tensor | None = None, **kwargs) -> Tensor:
        """
        Computes the SVGD update direction.
        
        Args:
            particles: (N, D)
            score: (N, D) - Required for SVGD.
            
        Returns:
            direction: (N, D)
        """
        if score is None:
            raise ValueError("SVGD requires 'score' (gradient of log-probability) to be provided.")

        # K_XY: (N, N) where K_XY[i, j] = k(particles[i], particles[j])
        # grad_K: (N, D) where grad_K[i] = \sum_j \nabla_{x_j} k(x_j, particles[i])
        K_XX, grad_K = self.kernel.forward(particles)
        
        # Term 1: 1/N \sum_j k(x_j, x_i) \nabla_{x_j} log p(x_j)
        # Matrix multiplication: (N, N) @ (N, D) -> (N, D)
        term1 = torch.matmul(K_XX, score)
        
        # Term 2: 1/N \sum_j \nabla_{x_j} k(x_j, x_i)
        term2 = grad_K
        
        direction = (term1 + term2) / particles.size(0)
        
        return direction
