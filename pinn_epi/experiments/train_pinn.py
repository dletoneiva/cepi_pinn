"""Train PINN models from configuration files."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
import os
from pinn_epi.configs.model_loader import (
    create_model_from_config, 
    get_model_parameters, 
    get_initial_conditions,
    get_output_size
)
from pinn_epi.models.physics import SIRModel, SEIRModel, SIModel  # Import other models as needed
from pinn_epi.models.networks import BaseMLP, HardICHead, TimeNormalizationEncoder, ModularPINN
from pinn_epi.training.trainer import PINNTrainer
from pinn_epi.analysis.data_wrangler import load_trajectories
from pinn_epi.constants import COMPARTMENTAL_MODEL_MAP

# Set up logging
logger = logging.getLogger(__name__)

def load_data(config: DictConfig) -> dict:
    """Load data based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary containing time series data
    """
    if config.data.source == "generated":
        # Data will be generated during training setup
        return {}
    elif config.data.source == "file":
        # Load from file
        data = load_trajectories(config.data.file_path)
        return data
    else:
        raise ValueError(f"Unknown data source: {config.data.source}")

def create_pinn_model(config: DictConfig, output_dim: int, initial_conditions: list) -> ModularPINN:
    """Create PINN model based on configuration.
    
    Args:
        config: Model configuration
        output_dim: Output dimension of the model
        initial_conditions: Initial conditions for the compartments
        
    Returns:
        Configured ModularPINN model
    """
    # Create backbone
    # Use default values if not specified in config
    input_dim = getattr(config.network, 'input_dim', 1)  # Time is typically 1D input
    hidden_dims = getattr(config.network, 'hidden_dims', [16, 16])  # Default hidden layers
    activation_name = getattr(config.network, 'activation', 'Tanh')  # Default activation
    
    # Convert activation string to actual class
    activation = getattr(torch.nn, activation_name) if hasattr(torch.nn, activation_name) else torch.nn.Tanh
    
    backbone = BaseMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation
    )
    
    # Create encoder if needed
    encoder = None
    if getattr(config.network, 'use_time_normalization', False):
        # These would come from data or config
        t0, t1 = 0.0, 30.0  # Default values, should be configurable
        encoder = TimeNormalizationEncoder(t0=t0, t1=t1)
    
    # Create head
    head = None
    if getattr(config.network, 'use_hard_ic', False):
        ic_tensor = torch.tensor(initial_conditions, dtype=torch.float32)
        # These would come from data or config
        t0, t1 = 0.0, 30.0  # Default values, should be configurable
        head = HardICHead(initial_conditions=ic_tensor, t0=t0, t1=t1)
    
    # Create modular PINN
    model = ModularPINN(
        backbone=backbone,
        head=head,
        encoder=encoder
    )
    
    return model

@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Train PINN model based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    logger.info("Starting PINN training")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Load or prepare data
    data = load_data(cfg.training)
    
    # Get model class and parameters
    model_type = cfg.compartmental.model.type.lower()
    
    # Handle case where model_type might be the full class name
    if model_type.endswith('model'):
        model_type = model_type[:-5]  # Remove 'model' suffix
    
    if model_type not in COMPARTMENTAL_MODEL_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = COMPARTMENTAL_MODEL_MAP[model_type]
    physics_model = model_class()
    
    # Get model parameters and initial conditions
    # Try to get parameters from the config directly first
    if hasattr(cfg.compartmental.model, 'parameters'):
        model_params = dict(cfg.compartmental.model.parameters)
    else:
        # Fallback to using the model loader function
        model_params = get_model_parameters(cfg.compartmental.model)
    
    # Try to get initial conditions from the config directly first
    if hasattr(cfg.compartmental.model, 'initial_conditions'):
        initial_conditions = list(cfg.compartmental.model.initial_conditions)
    else:
        # Fallback to using the model loader function
        initial_conditions = get_initial_conditions(cfg.compartmental.model)
    
    # Debug: Print initial conditions and parameters
    logger.info(f"Model type: {model_type}")
    logger.info(f"Initial conditions from config: {initial_conditions}")
    logger.info(f"Model compartments: {physics_model.compartment_names}")
    logger.info(f"Model parameters: {model_params}")
    
    # Ensure we have the right number of initial conditions
    if len(initial_conditions) != len(physics_model.compartment_names):
        raise ValueError(f"Mismatch between initial conditions ({len(initial_conditions)}) and model compartments ({len(physics_model.compartment_names)})")
    
    output_dim = get_output_size(cfg.compartmental.model)
    
    # Generate data if needed
    if cfg.training.data.source == "generated":
        from pinn_epi.analysis.evaluator import solve_compartmental_model
        solution = solve_compartmental_model(
            model=physics_model,
            t_span=cfg.training.data.t_span,
            y0=initial_conditions,  # Pass initial_conditions directly
            params=model_params,    # Pass model_params directly
            t_eval=np.linspace(cfg.training.data.t_span[0], cfg.training.data.t_span[1], cfg.training.data.num_points)
        )
        data = solution
    
    # Create PINN model
    pinn_model = create_pinn_model(cfg, output_dim, initial_conditions)
    
    # Initialize trainer
    trainer = PINNTrainer(
        model=pinn_model,
        physics_model=physics_model,
        data=data,
        config=OmegaConf.to_container(cfg.training, resolve=True)
    )
    
    # Train the model
    try:
        trainer.train()
        logger.info("PINN training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
