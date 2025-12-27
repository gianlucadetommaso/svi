from abc import ABC, abstractmethod

from torch import Tensor

from svi.kernels.base import Kernel


class SVIAlgorithm(ABC):
    """
    Abstract base class for Stein Variational Inference algorithms.
    """

    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def compute_direction(
        self, particles: Tensor, score: Tensor | None = None, **kwargs
    ) -> Tensor:
        """
        Computes the update direction for the particles.

        Args:
            particles: Tensor of shape (N, D) representing N particles of dimension D.
            score: Optional Tensor of shape (N, D) containing the score function
                   (gradient of log-probability) evaluated at each particle.
                   \nabla_{x_j} log p(x_j).
            **kwargs: Additional arguments that might be required by specific algorithms
                      (e.g. Hessian information, loss closures, etc).

        Returns:
            direction: Tensor of shape (N, D) representing the update direction.
                       x_new = x + epsilon * direction
        """
        pass
