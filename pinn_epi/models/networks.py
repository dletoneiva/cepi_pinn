"""Neural network architectures for PINNs."""

from typing import Optional, List
import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    """Standard Multi-Layer Perceptron backbone."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: type = nn.Tanh
    ):
        super().__init__()
        layers = []
        last_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                activation()
            ])
            last_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(last_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HardICAnsatz(nn.Module):
    """Hard initial condition ansatz: u(t) = u0 + (t-t0) * NN(t)"""
    
    def __init__(self, initial_conditions: torch.Tensor, t0: float, t1: float):
        super().__init__()
        self.register_buffer("u0", initial_conditions.clone().detach())
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, t: torch.Tensor, raw_output: torch.Tensor) -> torch.Tensor:
        # Apply the ansatz
        phi = (t - self.t0)
        return self.u0 + phi * raw_output


class ModularPINN(nn.Module):
    """Composable PINN architecture with backbone and ansatz."""
    
    def __init__(
        self,
        backbone: nn.Module,
        ansatz: Optional[nn.Module] = None,
        t0: float = 0.0,
        t1: float = 1.0
    ):
        super().__init__()
        self.backbone = backbone
        self.ansatz = ansatz
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize time to [-1, 1] to prevent Tanh saturation (mimicking notebook behavior)
        t_norm = 2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0
        
        # Apply backbone
        raw_output = self.backbone(t_norm)
        
        # Apply ansatz if present
        if self.ansatz:
            output = self.ansatz(t, raw_output)
        else:
            output = raw_output
            
        return output
