"""Neural network architectures for PINNs."""

from typing import Optional, List
import torch
import torch.nn as nn
import logging
from typing import List, Dict
from pinn_epi.utils.device_utils import DEVICE  # Import the centralized device

logger = logging.getLogger(__name__)

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


class TimeNormalizationEncoder(nn.Module):
    """Encoder that normalizes time inputs to [-1, 1]."""
    
    def __init__(self, t0: float, t1: float):
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize time to [-1, 1] to prevent Tanh saturation
        return 2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0


class HardICHead(nn.Module):
    """Hard initial condition head: u(t) = u0 + (t-t0) * NN(t)"""
    
    def __init__(self, initial_conditions: torch.Tensor, t0: float, t1: float):
        super().__init__()
        self.register_buffer("u0", initial_conditions.clone().detach())
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, t: torch.Tensor, raw_output: torch.Tensor) -> torch.Tensor:
        # Apply the head
        phi = (t - self.t0)
        return self.u0 + phi * raw_output


class ModularPINN(nn.Module):
    """Composable PINN architecture with encoder, backbone, and head."""
    
    def __init__(
        self,
        backbone: nn.Module,
        head: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.head = head
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Apply encoder if present
        features = self.encoder(t) if self.encoder else t
        
        # Apply backbone
        raw_output = self.backbone(features)
        
        # Apply head if present
        if self.head:
            output = self.head(t, raw_output)
        else:
            output = raw_output
            
        return output

def create_pinn_model(network_config: dict, compartment_names: list, initial_conditions: list, t_span: list, output_size: int) -> ModularPINN:
    """Create a PINN model from network configuration.
    
    Args:
        network_config: Configuration for the neural network
        compartment_names: Names of compartments in the model
        initial_conditions: Initial conditions for each compartment
        t_span: Time span [t_start, t_end]
        output_size: Number of output dimensions (compartments)
        
    Returns:
        Configured ModularPINN model
    """
    
    # Create network backbone
    backbone_config = network_config.get("backbone", {})
    input_dim = 1  # time dimension
    output_dim = output_size
    hidden_dims = [backbone_config.get("layer_size", 50)] * backbone_config.get("num_layers", 3)
    
    # Handle activation function - convert string to class
    activation_name = backbone_config.get("activation", "Tanh")
    if isinstance(activation_name, str):
        # Convert string to actual activation class
        activation = getattr(torch.nn, activation_name)
    else:
        activation = activation_name
    
    backbone = BaseMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation
    )
    
    # Create encoder if specified
    encoder = None
    if network_config.get("encoder") == "time_normalization":
        encoder = TimeNormalizationEncoder(
            t0=float(t_span[0]),
            t1=float(t_span[1])
        )
    
    # Create initial condition head if specified
    head = None
    if network_config.get("head") == "hardIC":
        initial_conditions_tensor = torch.tensor(initial_conditions, dtype=torch.float32, device=DEVICE)
        head = HardICHead(
            initial_conditions=initial_conditions_tensor,
            t0=float(t_span[0]),
            t1=float(t_span[1])
        )
    
    # Create modular PINN
    pinn_model = ModularPINN(
        backbone=backbone,
        head=head,
        encoder=encoder
    )
    
    # Set the device for the model
    pinn_model.to(DEVICE)
    
    # Add device property to access the device
    pinn_model.device = DEVICE
    
    # Log model creation details
    logger.info(f"Created PINN model with architecture:")
    logger.info(f"  - Backbone: {backbone_config}")
    logger.info(f"  - Encoder: {network_config.get('encoder', 'None')}")
    logger.info(f"  - Head: {network_config.get('head', 'None')}")
    logger.info(f"  - Device: {DEVICE}")
    
    return pinn_model
