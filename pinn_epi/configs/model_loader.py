"""Load compartmental models from configuration."""

from typing import Dict, Any, Union
import numpy as np
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra

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
    # Check if compartmental section exists and contains model
    if "compartmental" not in config:
        raise KeyError("Missing required configuration section: compartmental")
    
    if "model" not in config["compartmental"]:
        raise KeyError("Missing required configuration section: compartmental.model")
    
    # Check if model type exists
    model_section = config["compartmental"]["model"]
    if "type" not in model_section:
        raise KeyError("Missing required model key: type")
    
    model_type = model_section["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Check model section
    model_keys = ["type", "parameters", "initial_conditions", "output_size"]
    for key in model_keys:
        if key not in config["compartmental"]["model"]:
            raise KeyError(f"Missing required model key: {key}")


def create_model_from_config(config: Dict[str, Any]) -> CompartmentalModel:
    """Create a compartmental model instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CompartmentalModel instance
        
    Raises:
        ValueError: If model type is unknown
    """
    model_type = config["compartmental"]["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class()


def get_model_parameters(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract model parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of model parameters
    """
    return config.get('compartmental', {}).get('model', {}).get('parameters', {})


def get_initial_conditions(config: Dict[str, Any]) -> list:
    """Extract initial conditions from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of initial conditions
    """
    return config.get('compartmental', {}).get('model', {}).get('initial_conditions', [])


def get_output_size(config: Dict[str, Any]) -> int:
    """Extract output size from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Output size (number of compartments)
    """
    return config.get('compartmental', {}).get('model', {}).get('output_size', 0)


def run_simulation_from_config(config: DictConfig) -> Dict[str, Any]:
    """Run a simulation based on the provided configuration.
    
    Args:
        config: Configuration dictionary from Hydra
        
    Returns:
        Dictionary containing the model, trajectories, and figure path (if saved)
    """
    # Convert DictConfig to regular dict for compatibility
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Validate configuration
    validate_model_config(config_dict)
    
    # Create model
    model = create_model_from_config(config_dict)
    
    # Extract simulation parameters
    t_span = config_dict["simulation"]["t_span"]
    y0 = config_dict["compartmental"]["model"]["initial_conditions"]  # Get from model config
    params = config_dict["compartmental"]["model"]["parameters"]
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
    
    # Plot solution if requested
    result = {
        "model": model,
        "trajectories": trajectories,
        "t": t_eval
    }
    
    # Handle plotting
    plotting_config = config_dict.get("plotting", {})
    if plotting_config.get("show_plot", True):
        # Determine save path if saving is requested
        save_path = None
        if plotting_config.get("save_plot", False):
            # Use Hydra's output directory
            hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = config_dict["compartmental"]["model"]["type"]
            save_path = f"{hydra_output_dir}/{model_type}_simulation_{timestamp}.png"
        
        fig, ax = plot_compartmental_solution(
            t=t_eval,
            trajectories=trajectories,
            title=plotting_config.get("title", "Compartmental Model Simulation"),
            resolution=plotting_config.get("resolution", "low"),
            show=plotting_config.get("show_plot", True),
            save_path=save_path,
            model_params=params,
            initial_conditions=y0,
        )
        result["figure"] = fig
        result["axes"] = ax
        
        if save_path:
            result["save_path"] = save_path
            print(f"Figure saved to: {save_path}")
    
    return result
