"""Load compartmental models from configuration."""

from typing import Dict, Any, Union
import numpy as np
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from pinn_epi.models.physics import CompartmentalModel, SIRModel, SEIRModel, SIModel
from pinn_epi.analysis.plotting import plot_compartmental_solution
from pinn_epi.analysis.evaluator import solve_compartmental_model


# Mapping of model names to classes
MODEL_REGISTRY = {
    "SIRModel": SIRModel,
    "SEIRModel": SEIRModel,
    "SIModel": SIModel,
}


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate the model configuration.
    
    Args:
        config: Configuration dictionary containing model information
        
    Raises:
        ValueError: If configuration is invalid
        KeyError: If required keys are missing
    """
    # Check if model type exists
    model_type = config["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Check if required keys exist
    required_keys = ["model", "simulation", "plotting"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration section: {key}")
    
    # Check model section
    model_keys = ["type", "parameters", "initial_conditions", "output_size"]
    for key in model_keys:
        if key not in config["model"]:
            raise KeyError(f"Missing required model key: {key}")
    
    # Check simulation section
    sim_keys = ["t_span", "t_eval_points"]
    for key in sim_keys:
        if key not in config["simulation"]:
            raise KeyError(f"Missing required simulation key: {key}")


def create_model_from_config(config: Dict[str, Any]) -> CompartmentalModel:
    """Create a compartmental model instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CompartmentalModel instance
        
    Raises:
        ValueError: If model type is unknown
    """
    model_type = config["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class()


def run_simulation_from_config(config: DictConfig) -> Dict[str, Any]:
    """Run a simulation based on the provided configuration.
    
    Args:
        config: Configuration dictionary from Hydra
        
    Returns:
        Dictionary containing the model, trajectories, and figure path (if saved)
    """
    # Convert DictConfig to regular dict for compatibility
    if hasattr(config, '__dict__'):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    # Validate configuration
    validate_model_config(config_dict)
    
    # Create model
    model = create_model_from_config(config_dict)
    
    # Extract simulation parameters
    t_span = config_dict["simulation"]["t_span"]
    y0 = config_dict["model"]["initial_conditions"]  # Get from model config
    params = config_dict["model"]["parameters"]
    t_eval_points = config_dict["simulation"]["t_eval_points"]
    
    # Create t_eval
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
    
    # Solve ODE
    trajectories = solve_compartmental_model(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        t_eval=t_eval,
    )
    
    # Plot solution
    fig, ax = plot_compartmental_solution(
        t=t_eval,
        trajectories=trajectories,
        title=config_dict["plotting"]["title"],
        resolution=config_dict["plotting"]["resolution"],
        show=config_dict["plotting"]["show_plot"],
        model_params=params,
        initial_conditions=y0,
    )
    
    # Handle saving
    result = {
        "model": model,
        "trajectories": trajectories,
        "figure": fig,
        "axes": ax,
        "t": t_eval
    }
    
    if config_dict["experiment"]["save_figures"]:
        figures_dir = config_dict["experiment"]["figures_dir"]
        os.makedirs(figures_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config_dict["model"]["type"]
        filename = f"{timestamp}_{model_type.lower()}_simulation.png"
        filepath = os.path.join(figures_dir, filename)
        
        fig.savefig(filepath, bbox_inches='tight')
        result["figure_path"] = filepath
        print(f"Figure saved to: {filepath}")
    
    # Show plot if requested
    if config_dict["experiment"]["plot_results"]:
        import matplotlib.pyplot as plt
        plt.show()
    
    return result
