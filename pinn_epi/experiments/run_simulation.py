"""Run simulations from configuration files."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
from pinn_epi.configs.model_loader import run_simulation_from_config

# Set up logging
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Run simulation based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    logger.info("Starting simulation run")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Run simulation
    try:
        result = run_simulation_from_config(cfg)
        logger.info(f"Simulation completed successfully for {cfg.compartmental.model.type}")
        logger.info(f"Model parameters: {OmegaConf.to_container(cfg.compartmental.model.parameters, resolve=True)}")
        logger.info(f"Initial conditions: {cfg.compartmental.model.initial_conditions}")
    except Exception as e:
        logger.error(f"Simulation failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
