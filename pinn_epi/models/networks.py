"""Neural network architectures for PINNs."""

from typing import Optional, List
import torch
import torch.nn as nn
import logging
from typing import List, Dict
from pinn_epi.utils.device_utils import DEVICE  # Import the centralized device

logger = logging.getLogger(__name__)

class BaseMLP(nn.Module):
    """Standard Multi-Layer Perceptron backbone.
    
    A flexible MLP implementation that can be used as the backbone for PINNs.
    It consists of fully connected layers with configurable activation functions
    between hidden layers.
    
    Attributes:
        network: The sequential neural network layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: type = nn.Tanh
    ):
        """Initialize the MLP with specified architecture.
        
        Args:
            input_dim: Number of input features (typically 1 for time)
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output dimensions (compartments)
            activation: Activation function class (default: Tanh)
        """
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
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all layers
        """
        return self.network(x)


class TimeNormalizationEncoder(nn.Module):
    """Encoder that normalizes time inputs to [-1, 1].
    
    This encoder helps prevent activation saturation in neural networks by
    normalizing the time input to the range [-1, 1]. This is particularly
    important for Tanh activation functions.
    
    Attributes:
        t0: Start time for normalization
        t1: End time for normalization
    """

    def __init__(self, t0: float, t1: float):
        """Initialize the time normalization encoder.
        
        Args:
            t0: Start time for normalization
            t1: End time for normalization
        """
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize time inputs to [-1, 1].
        
        Args:
            t: Time tensor to normalize
            
        Returns:
            Normalized time tensor in range [-1, 1]
        """
        # Normalize time to [-1, 1] to prevent Tanh saturation
        return 2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0


class HardICHead(nn.Module):
    """Hard initial condition head: u(t) = u0 + (t-t0) * NN(t).
    
    This head enforces initial conditions by structurally embedding them
    into the network architecture. The neural network learns only the
    deviation from the initial conditions, ensuring that the PINN exactly
    satisfies u(t0) = u0.
    
    Attributes:
        u0: Initial conditions tensor
        t0: Initial time
        t1: Final time (for reference)
    """

    def __init__(self, initial_conditions: torch.Tensor, t0: float, t1: float):
        """Initialize the hard initial condition head.
        
        Args:
            initial_conditions: Initial values for each compartment
            t0: Initial time
            t1: Final time
        """
        super().__init__()
        self.register_buffer("u0", initial_conditions.clone().detach())
        self.t0 = t0
        self.t1 = t1
        
    def forward(self, t: torch.Tensor, raw_output: torch.Tensor) -> torch.Tensor:
        """Apply the hard initial condition constraint.
        
        Args:
            t: Time tensor
            raw_output: Raw output from the neural network backbone
            
        Returns:
            Output tensor that exactly satisfies initial conditions
        """
        # Apply the head
        phi = (t - self.t0)
        return self.u0 + phi * raw_output


class ModularPINN(nn.Module):
    """Composable PINN architecture with encoder, backbone, and head.
    
    This modular architecture allows for flexible PINN construction by
    combining:
    
    1. Encoder: Preprocesses inputs (e.g., time normalization)
    2. Backbone: Main neural network (e.g., MLP)
    3. Head: Post-processes outputs (e.g., hard initial conditions)
    
    This design enables easy experimentation with different components
    while maintaining a consistent interface.
    
    Attributes:
        encoder: Optional input encoder module
        backbone: Main neural network module
        head: Optional output head module
        device: Computing device for the model
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None
    ):
        """Initialize the modular PINN architecture.
        
        Args:
            backbone: Main neural network module
            head: Optional output head module
            encoder: Optional input encoder module
        """
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.head = head
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the modular PINN.
        
        Args:
            t: Time tensor input
            
        Returns:
            Model predictions for all compartments
        """
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
    
    This factory function builds a PINN model by combining components
    specified in the configuration. It handles the creation of the
    backbone network, optional encoder, and optional head.
    
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
