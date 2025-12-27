import torch
from torch import Tensor
from svi.kernels.base import Kernel

class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) Kernel implementation.
    
    Computes the Gaussian kernel k(x, y) = exp(-||x - y||^2 / h) and its gradients.
    Optionally uses the median heuristic to determine the bandwidth h.
    """
    def __init__(self, bandwidth: float | None = None):
        """
        Args:
            bandwidth: Optional fixed bandwidth parameter h. 
                       If None, the median heuristic is used to estimate h from the data.
        """
        self.bandwidth = bandwidth

    def forward(self, X: Tensor, Y: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Computes the kernel matrix and the SVGD gradient term.

        The gradient term computed is \sum_j \nabla_{x_j} k(x_j, x_i).
        For the RBF kernel, \nabla_{x_j} k(x_j, x_i) = 2/h * k(x_j, x_i) * (x_i - x_j).
        
        Args:
            X: Tensor of shape (N, D).
            Y: Optional Tensor of shape (M, D). If None, Y = X.
            
        Returns:
            K_XY: Kernel matrix of shape (N, M).
            grad_K: Gradient term of shape (N, D) corresponding to sum_j grad_{x_j} k(x_j, x_i).
        """
        if Y is None:
            Y = X
            
        diff = X.unsqueeze(1) - Y.unsqueeze(0)  # Shape: (N, M, D)
        dist_sq = torch.sum(diff ** 2, dim=-1)   # Shape: (N, M)
        
        if self.bandwidth is None:
            h = self._median_heuristic(dist_sq)
        else:
            h = self.bandwidth
            
        gamma = 1.0 / (h + 1e-8)
        
        K_XY = torch.exp(-dist_sq * gamma)
        
        # Compute sum_j \nabla_{x_j} k(x_j, x_i)
        # Using the identity derived for RBF: sum_j 2 * gamma * k(x_i, x_j) * (x_i - x_j)
        grad_K = 2 * gamma * (diff * K_XY.unsqueeze(-1)).sum(dim=1)
        
        return K_XY, grad_K

    def _median_heuristic(self, dist_sq: Tensor) -> Tensor:
        """
        Estimates the bandwidth h using the median heuristic:
        h = median(dist_sq) / log(N)
        """
        N = dist_sq.size(0)
        if N == 1:
            return torch.tensor(1.0, device=dist_sq.device)
            
        d = dist_sq.view(-1)
        median = torch.median(d)
        
        if median == 0.0:
             return torch.tensor(1.0, device=dist_sq.device) 

        return median / (torch.log(torch.tensor(N)) + 1e-8)
