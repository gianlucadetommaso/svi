from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Kernel(ABC):
    """
    Abstract base class for Kernels used in Stein Variational Inference.
    """

    @abstractmethod
    def forward(self, X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the kernel matrix and its gradient with respect to the first argument.

        Args:
            X: Tensor of shape (N, D) containing N particles of dimension D.
            Y: Tensor of shape (M, D) containing M particles of dimension D.

        Returns:
            K_XY: Tensor of shape (N, M) containing the kernel values k(x_i, y_j).
            grad_K: Tensor of shape (N, D) containing sum_j grad_x k(x, y_j)|_{x=x_i}.
                    Note: The specific shape/summation might depend on the SVGD implementation needs,
                    but typically for SVGD we need \sum_j \nabla_{x_j} k(x_j, x) which is symmetric.
                    Let's clarify the standard SVGD requirement:
                    Phi(x) = 1/N \sum_{j=1}^N [ k(x_j, x) \nabla_{x_j} log p(x_j) + \nabla_{x_j} k(x_j, x) ]

                    Here, we usually compute the matrix K(X, X).
                    And we need the gradient of the kernel w.r.t the first argument or second argument?
                    \nabla_{x_j} k(x_j, x).
        """
        pass
