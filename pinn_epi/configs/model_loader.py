"""Load compartmental models from configuration."""

from typing import Dict, Any, Union
import importlib
import logging
from omegaconf import DictConfig, OmegaConf

from pinn_epi.constants import COMPARTMENTAL_MODEL_REGISTRY

# Set up logging
logger = logging.getLogger(__name__)


def get_model_class(model_type: str) -> type:
    """Dynamically import and return the model class.
    
    Args:
        model_type: String name of the model class
        
    Returns:
        The model class
        
    Raises:
        ValueError: If model type is unknown
    """
    if model_type not in COMPARTMENTAL_MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(COMPARTMENTAL_MODEL_REGISTRY.keys())}")
    
    module_path, class_name = COMPARTMENTAL_MODEL_REGISTRY[model_type].rsplit('.', 1)
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
    logger.debug("Validating model configuration")
    
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
    if model_type not in COMPARTMENTAL_MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(COMPARTMENTAL_MODEL_REGISTRY.keys())}")
    
    # Check model section
    model_keys = ["type", "parameters", "initial_conditions", "output_size"]
    for key in model_keys:
        if key not in config["compartmental"]["model"]:
            raise KeyError(f"Missing required model key: {key}")
    
    logger.debug(f"Model configuration validated successfully for {model_type}")
    
    # Validate that learnable parameters are in the physics model
    learnable_param_names = config.get("training", {}).get("learnable_parameters", [])
    model_params = config.get("compartmental", {}).get("model", {}).get("parameters", {})
    
    for param_name in learnable_param_names:
        if param_name not in model_params:
            raise ValueError(f"Learnable parameter '{param_name}' not found in model parameters. "
                             f"Available parameters: {list(model_params.keys())}")


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
    logger.info(f"Creating model instance for {model_type}")
    model_class = get_model_class(model_type)
    
    # Extract parameters from config
    model_params = get_model_parameters(config)
    return model_class(parameters=model_params)


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


def get_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract network configuration from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Network configuration dictionary
    """
    return config.get('network', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Training configuration dictionary
    """
    return config.get('training', {})


def get_simulation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract simulation configuration from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Simulation configuration dictionary
    """
    sim_config = config.get('numerical_ode_params', {})
    # Ensure solve_ode key exists with default value
    if 'solve_ode' not in sim_config:
        sim_config['solve_ode'] = True
    return sim_config


def get_synthetic_plotting_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract synthetic plotting configuration from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Synthetic plotting configuration dictionary
    """
    return config.get('plotting', {})


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data configuration from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Data configuration dictionary
    """
    return config.get('data', {})
