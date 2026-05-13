"""Train PINN models from configuration files."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
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
    backbone = BaseMLP(
        input_dim=config.network.input_dim,
        hidden_dims=config.network.hidden_dims,
        output_dim=output_dim,
        activation=getattr(torch.nn, config.network.activation) if hasattr(torch.nn, config.network.activation) else torch.nn.Tanh
    )
    
    # Create encoder if needed
    encoder = None
    if config.network.get('use_time_normalization', False):
        # These would come from data or config
        t0, t1 = 0.0, 30.0  # Default values, should be configurable
        encoder = TimeNormalizationEncoder(t0=t0, t1=t1)
    
    # Create head
    head = None
    if config.network.get('use_hard_ic', False):
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
    if model_type not in COMPARTMENTAL_MODEL_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = COMPARTMENTAL_MODEL_MAP[model_type]
    physics_model = model_class()
    
    # Get model parameters and initial conditions
    model_params = get_model_parameters(cfg.compartmental.model)
    initial_conditions = get_initial_conditions(cfg.compartmental.model)
    output_dim = get_output_size(cfg.compartmental.model)
    
    # Generate data if needed
    if cfg.training.data.source == "generated":
        from pinn_epi.analysis.evaluator import solve_compartmental_model
        solution = solve_compartmental_model(
            model=physics_model,
            t_span=cfg.training.data.t_span,
            y0=initial_conditions,
            params=model_params,
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
```

Now, to run a test training on the SIR model, you'll need to make sure you have the appropriate configuration files set up. Here's what you should check:

1. Make sure you have a training configuration file in `configs/training/` (e.g., `test.yaml`)
2. Ensure your base configuration (`configs/base.yaml`) properly references the training config
3. Make sure your compartmental model config (`configs/compartmental/sir.yaml`) is properly set up

You should be able to run the training with:
```bash
python pinn_epi/experiments/train_pinn.py
```

The current implementation should work with the existing codebase, but you might need to adjust some configuration parameters based on your specific setup.