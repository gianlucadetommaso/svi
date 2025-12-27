from typing import Iterable, Any, Callable
import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from svi.algorithms.base import SVIAlgorithm

class SVIModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Stein Variational Inference.
    
    This module manages a set of particles (copies of the model) and updates them
    using a specified SVI algorithm (e.g., SVGD).
    
    It supports two modes:
    1. Supervised Learning: Providing a model and data (x, y). The module optimizes
       posterior p(theta | x, y) ~ p(y | x, theta) * p(theta).
    2. Density Estimation: Providing a target log-probability function. The module
       optimizes particles to match the target distribution p(theta).
    """
    
    def __init__(
        self,
        model: nn.Module,
        svi_algorithm: SVIAlgorithm,
        num_particles: int,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        freeze_previous_layers: bool = False,
        prior_std: float | None = None,
        target_log_prob_fn: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            model: The base model to perform inference on.
            svi_algorithm: The SVI algorithm to use (e.g., SVGD).
            num_particles: Number of particles to maintain.
            optimizer_cls: Optimizer class to use for updating particles.
            optimizer_kwargs: Arguments for the optimizer.
            freeze_previous_layers: If True, only the last layer of the particles will be optimized.
                                    The base model's previous layers will be frozen and shared.
            prior_std: Standard deviation for the Gaussian prior. If None (default), no prior is applied (Uniform prior).
                       Prior: p(theta) ~ N(0, prior_std^2 * I)
            target_log_prob_fn: Optional function for density estimation mode. 
                                Takes a particle tensor (shape like parameters) and returns scalar log probability.
                                If provided, training_step will ignore batch data and optimize this target.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "svi_algorithm", "target_log_prob_fn"])
        
        self._svi_algorithm = svi_algorithm
        self._num_particles = num_particles
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._prior_std = prior_std
        self._target_log_prob_fn = target_log_prob_fn
        
        # Prepare model
        self.base_model = model
        
        if freeze_previous_layers:
            self._freeze_all_except_last_layer(self.base_model)
            
        # Initialize particles
        self.trainable_param_names = [n for n, p in self.base_model.named_parameters() if p.requires_grad]
        trainable_params = [p for p in self.base_model.parameters() if p.requires_grad]
        initial_params = parameters_to_vector(trainable_params)
        
        # Store metadata for reconstruction (vector -> dict)
        self.param_shapes = {n: p.shape for n, p in self.base_model.named_parameters() if p.requires_grad}
        self.param_numels = {n: p.numel() for n, p in self.base_model.named_parameters() if p.requires_grad}
        
        # Store particles as a Tensor (num_particles, num_params)
        self.particles = nn.Parameter(
            initial_params.unsqueeze(0).repeat(num_particles, 1),
            requires_grad=True
        )
        
        # Perturb particles slightly
        with torch.no_grad():
            noise = torch.randn_like(self.particles) * 1e-2
            self.particles.add_(noise)

    def _freeze_all_except_last_layer(self, model: nn.Module):
        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False
            
        # Unfreeze the parameters of the last module with parameters.
        modules_with_params = [m for m in model.modules() if len(list(m.parameters(recurse=False))) > 0]
        if modules_with_params:
            last_module = modules_with_params[-1]
            for p in last_module.parameters():
                p.requires_grad = True
        else:
            # Fallback
            params = list(model.named_parameters())
            if len(params) >= 2:
                for _, p in params[-2:]:
                    p.requires_grad = True
            elif len(params) == 1:
                params[0][1].requires_grad = True

    def forward(self, x: Tensor, particle_idx: int) -> Tensor:
        """
        Forward pass using a specific particle.
        """
        self._load_particle_params(particle_idx)
        return self.base_model(x)
        
    def _load_particle_params(self, particle_idx: int):
        # Helper for legacy/manual access, though training_step now uses functional approach
        trainable_params = [p for p in self.base_model.parameters() if p.requires_grad]
        vector_to_parameters(self.particles[particle_idx], trainable_params)

    def _unflatten_params(self, param_vec: Tensor) -> dict[str, Tensor]:
        """
        Reconstructs the parameter dictionary from a flattened parameter vector.
        """
        params_dict = {}
        offset = 0
        for name in self.trainable_param_names:
            numel = self.param_numels[name]
            shape = self.param_shapes[name]
            # Slice and reshape
            param_flat = param_vec[offset : offset + numel]
            params_dict[name] = param_flat.view(shape)
            offset += numel
        return params_dict

    def training_step(self, batch, batch_idx):
        grads = []
        log_probs = [] # Used for logging in density mode
        
        if self._target_log_prob_fn is not None:
            # --- Density Estimation Mode (Vectorized) ---
            p = self.particles.detach().requires_grad_(True)
            log_probs = self._target_log_prob_fn(p)
            score_stack = torch.autograd.grad(log_probs.sum(), p)[0]
            
            if self._prior_std is not None:
                 prior_grad = - self.particles / (self._prior_std ** 2)
                 score_stack += prior_grad
                 
            self.log("mean_log_prob", log_probs.detach().mean())

        else:
            # --- Supervised Learning Mode (Vectorized) ---
            x, y = batch
            
            # Prepare fixed parameters (buffers + non-trainable params)
            # We want a full state dict for functional_call.
            # We start with the current model state (which has buffers and fixed params correct)
            # and override the trainable params.
            base_params = dict(self.base_model.named_parameters())
            base_buffers = dict(self.base_model.named_buffers())
            full_state = {**base_params, **base_buffers}
            
            def compute_loss_single_particle(p_vec, x_batch, y_batch):
                # 1. Unflatten trainable params
                trainable_dict = self._unflatten_params(p_vec)
                # 2. Merge with fixed params/buffers
                # efficient merge: create a new dict with overrides
                # Note: This shallow copy approach assumes full_state values are not modified in place
                # which is true for functional_call usually.
                current_state = {**full_state, **trainable_dict}
                
                # 3. Functional Call
                y_hat = torch.func.functional_call(self.base_model, current_state, (x_batch,))
                
                # 4. Loss
                return self._compute_loss(y_hat, y_batch)

            # Define gradient function
            # We want gradients w.r.t p_vec (arg 0)
            grad_fn = torch.func.grad(compute_loss_single_particle, argnums=0)
            
            # Vectorize over particles (dim 0 of p_vec)
            # in_dims: (0, None, None) -> p_vec is batched, x/y are broadcasted
            vmap_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, None, None))
            
            # Compute gradients for all particles
            # self.particles: (N_particles, N_params)
            grad_vecs = vmap_grad_fn(self.particles.detach(), x, y)
            
            # Compute Score
            # Score = - grad(Loss) + grad(Prior)
            if self._prior_std is not None:
                prior_grad = - self.particles / (self._prior_std ** 2)
            else:
                prior_grad = torch.zeros_like(self.particles)
                
            score_stack = -grad_vecs + prior_grad
            
            # For logging, we can compute the mean loss. 
            # We could vmap the loss fn too if we want exact metrics.
            # let's skip for now to save compute or implement if needed.
            
        # --- Common Update Step ---
        direction = self._svi_algorithm.compute_direction(self.particles.detach(), score_stack)
        
        self.particles.grad = -direction
        
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        
        return None

    def _compute_loss(self, y_hat, y):
        if isinstance(y, torch.Tensor) and (y.dtype == torch.long or y.dtype == torch.int):
             return torch.nn.functional.cross_entropy(y_hat, y)
        else:
             # Use manual MSE to avoid "impl: target and input tensors must have identical shapes"
             # error on MPS backend when using torch.func.vmap
             return (y_hat - y).pow(2).mean()

    def configure_optimizers(self):
        return self._optimizer_cls([self.particles], **self._optimizer_kwargs)

    @property
    def automatic_optimization(self) -> bool:
        return False
