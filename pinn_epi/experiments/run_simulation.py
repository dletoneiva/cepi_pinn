"""Run compartmental model simulations from configuration files."""

import argparse
import hydra
from omegaconf import DictConfig
from pinn_epi.configs.model_loader import run_simulation_from_config


def main_cli():
    """Command line interface to run simulation with specified config file."""
    parser = argparse.ArgumentParser(description="Run compartmental model simulation")
    parser.add_argument(
        "config_name", 
        help="Name of the configuration file (without .yaml extension)",
        default="sir_base",
        nargs="?"
    )
    args = parser.parse_args()
    
    # Run the Hydra main function with the specified config
    _run_with_config(args.config_name)


@hydra.main(version_base=None, config_path="../../configs", config_name="sir_base")
def _run_with_config(cfg: DictConfig, config_name: str = "sir_base") -> None:
    """Run simulation based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        config_name: Name of the configuration file
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
    main_cli()
