"""Run compartmental model simulations from configuration files."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pinn_epi.configs.model_loader import run_simulation_from_config

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
    
    # Run simulation
    result = run_simulation_from_config(cfg)
    
    # Print summary
    print(f"\nSimulation completed for {cfg.model.type}")
    print(f"Model parameters: {OmegaConf.to_container(cfg.model.parameters, resolve=True)}")
    print(f"Initial conditions: {cfg.model.initial_conditions}")

if __name__ == "__main__":
    main()
