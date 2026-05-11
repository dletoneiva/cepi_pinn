"""Load compartmental models from configuration."""

from typing import Dict, Any, Union
import numpy as np
import os
from datetime import datetime

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
    required_keys = ["experiment", "model", "simulation", "plotting"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration section: {key}")
    
    # Check experiment section
    exp_keys = ["name", "plot_results", "save_figures", "figures_dir"]
    for key in exp_keys:
        if key not in config["experiment"]:
            raise KeyError(f"Missing required experiment key: {key}")
    
    # Check model section
    if "parameters" not in config["model"]:
        raise KeyError("Missing required model parameters")
    
    # Check simulation section
    sim_keys = ["t_span", "y0", "t_eval_points"]
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


def run_simulation_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a simulation based on the provided configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the model, trajectories, and figure path (if saved)
    """
    # Validate configuration
    validate_model_config(config)
    
    # Create model
    model = create_model_from_config(config)
    
    # Extract simulation parameters
    t_span = config["simulation"]["t_span"]
    y0 = config["simulation"]["y0"]
    params = config["model"]["parameters"]
    t_eval_points = config["simulation"]["t_eval_points"]
    
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
        title=config["plotting"]["title"],
        resolution=config["plotting"]["resolution"],
        show=config["plotting"]["show_plot"]
    )
    
    # Handle saving
    result = {
        "model": model,
        "trajectories": trajectories,
        "figure": fig,
        "axes": ax
    }
    
    if config["experiment"]["save_figures"]:
        figures_dir = config["experiment"]["figures_dir"]
        os.makedirs(figures_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config["model"]["type"]
        filename = f"{timestamp}_{model_type.lower()}_simulation.png"
        filepath = os.path.join(figures_dir, filename)
        
        fig.savefig(filepath, bbox_inches='tight')
        result["figure_path"] = filepath
        print(f"Figure saved to: {filepath}")
    
    # Show plot if requested
    if config["experiment"]["plot_results"]:
        import matplotlib.pyplot as plt
        plt.show()
    
    return result
