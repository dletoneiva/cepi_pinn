"""Load compartmental models from configuration."""

from typing import Dict, Any, Union
import numpy as np
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import importlib

from pinn_epi.analysis.plotting import plot_compartmental_solution
from pinn_epi.analysis.evaluator import solve_compartmental_model
from pinn_epi.analysis.data_wrangler import save_simulation_data
from pinn_epi.constants import MODEL_REGISTRY


def get_model_class(model_type: str) -> type:
    """Dynamically import and return the model class.
    
    Args:
        model_type: String name of the model class
        
    Returns:
        The model class
        
    Raises:
        ValueError: If model type is unknown
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_type].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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


def create_model_from_config(config: Dict[str, Any]) -> Any:
    """Create a compartmental model instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CompartmentalModel instance
        
    Raises:
        ValueError: If model type is unknown
    """
    model_type = config["compartmental"]["model"]["type"]
    model_class = get_model_class(model_type)
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
    show_plot = plotting_config.get("show_plot", True)
    save_plot = plotting_config.get("save_plot", False)
    
    # Determine save path if saving is requested
    save_path = None
    if save_plot:
        # Use Hydra's output directory
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config_dict["compartmental"]["model"]["type"]
        save_path = f"{hydra_output_dir}/{model_type}_simulation_{timestamp}.pdf"
        
        # Ensure the directory exists
        os.makedirs(hydra_output_dir, exist_ok=True)
    
    # Generate title based on model compartments if not provided
    title = plotting_config.get("title")
    if title == "Compartmental Model Simulation":  # Default title, replace with model-specific one
        # Get compartment names from the model
        compartment_names = model.compartment_names
        compartments_str = "".join(compartment_names)
        title = f"{compartments_str} differential equation solutions"
    
    fig, ax = plot_compartmental_solution(
        t=t_eval,
        trajectories=trajectories,
        title=title,
        show=show_plot,
        save_path=save_path,
        model_params=params,
        initial_conditions=y0,
    )
    result["figure"] = fig
    result["axes"] = ax
    
    if save_path:
        result["save_path"] = save_path
        print(f"Figure saved to: {save_path}")
    
    # Handle data saving
    data_config = config_dict.get("data", {})
    save_data = data_config.get("save_data", True)
    
    if save_data:
        # Use Hydra's output directory for data as well
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        timestamp = datetime.now().strftime("%Y%mdd_%H%M%S")
        model_type = config_dict["compartmental"]["model"]["type"]
        
        # Ensure the directory exists
        os.makedirs(hydra_output_dir, exist_ok=True)
        
        # Save simulation data
        save_simulation_data(
            trajectories=trajectories,
            model_params=params,
            initial_conditions=y0,
            compartment_names=model.compartment_names,
            save_dir=hydra_output_dir,
            t_eval=t_eval,
            file_prefix=f"{model_type}_simulation_{timestamp}"
        )
        
        result["data_save_path"] = hydra_output_dir
        print(f"Data saved to: {hydra_output_dir}")
    
    return result
