"""Run compartmental model simulations from configuration files."""

import hydra
from omegaconf import DictConfig
from pinn_epi.configs.model_loader import run_simulation_from_config

@hydra.main(version_base=None, config_path="../../configs", config_name="sir_base")
def main(cfg: DictConfig) -> None:
    """Run simulation based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert OmegaConf to regular dict
    config_dict = hydra.utils.instantiate(cfg, _convert_="dict")
    
    # Run simulation
    result = run_simulation_from_config(config_dict)
    
    # Print summary
    print(f"\nSimulation completed for {cfg.model.type}")
    print(f"Model parameters: {cfg.model.parameters}")
    print(f"Initial conditions: {cfg.simulation.y0}")
    if "figure_path" in result:
        print(f"Figure saved to: {result['figure_path']}")

if __name__ == "__main__":
    main()