"""Run compartmental model simulations from configuration files."""

import sys
import hydra
from omegaconf import DictConfig
from pinn_epi.configs.model_loader import run_simulation_from_config


def get_config_name():
    """Get config name from command line arguments."""
    # Check if a config name was provided as a command line argument
    if len(sys.argv) > 1:
        # If it's not a hydra override (doesn't start with + or ~), treat it as config name
        first_arg = sys.argv[1]
        if not first_arg.startswith(('+', '~', 'hydra/', 'experiment=', 'model=', 'simulation=', 'plotting=')):
            return first_arg
    # Default to sir_base if no config name provided
    return "sir_base"


# Get the config name before Hydra initialization
config_name = get_config_name()


@hydra.main(version_base=None, config_path="../../configs", config_name=config_name)
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
