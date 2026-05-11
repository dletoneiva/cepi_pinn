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
    required_keys = ["model", "simulation"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration section: {key}")
    
    # Check model section
    model_keys = ["type", "parameters", "initial_conditions", "output_size"]
    for key in model_keys:
        if key not in config["model"]:
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
    model_type = config["model"]["type"]
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
    return config.get('model', {}).get('parameters', {})


def get_initial_conditions(config: Dict[str, Any]) -> list:
    """Extract initial conditions from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of initial conditions
    """
    return config.get('model', {}).get('initial_conditions', [])


def get_output_size(config: Dict[str, Any]) -> int:
    """Extract output size from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Output size (number of compartments)
    """
    return config.get('model', {}).get('output_size', 0)


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
    
    # Plot solution if requested
    result = {
        "model": model,
        "trajectories": trajectories,
        "t": t_eval
    }
    
    if config_dict.get("plotting", {}).get("show_plot", True):
        fig, ax = plot_compartmental_solution(
            t=t_eval,
            trajectories=trajectories,
            title=config_dict.get("plotting", {}).get("title", "Compartmental Model Simulation"),
            resolution=config_dict.get("plotting", {}).get("resolution", "low"),
            show=config_dict.get("plotting", {}).get("show_plot", True),
            model_params=params,
            initial_conditions=y0,
        )
        result["figure"] = fig
        result["axes"] = ax
    
    return result
