"""Load compartmental models from configuration."""

from typing import Dict, Any
import importlib
import logging

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
    model_keys = ["type", "parameters", "initial_conditions"]
    for key in model_keys:
        if key not in config["compartmental"]["model"]:
            raise KeyError(f"Missing required model key: {key}")
    
    # Validate parameters
    parameters = get_model_parameters(config)
    _validate_parameters(parameters, model_type)
    
    # Validate initial conditions length matches expected compartment count
    initial_conditions = get_initial_conditions(config)
    model_class = get_model_class(model_type)
    
    # Create a temporary instance to get compartment names (without parameters since we only need the count)
    try:
        temp_model = model_class()
        expected_compartments = len(temp_model.compartment_names)
    except Exception:
        # If we can't create a model without parameters, we'll skip this validation
        expected_compartments = None
    
    if expected_compartments is not None:
        if len(initial_conditions) != expected_compartments:
            raise ValueError(f"Initial conditions length ({len(initial_conditions)}) does not match "
                             f"expected number of compartments ({expected_compartments}) for {model_type}")
    
    # Validate initial conditions values
    _validate_initial_conditions(initial_conditions, model_type)
    
    # Validate conservation for all models (assuming all models conserve population)
    if expected_compartments is not None:
        ic_sum = sum(initial_conditions)
        if not abs(ic_sum - 1.0) < 1e-6:
            raise ValueError(f"Initial conditions must sum to 1.0 for {model_type} (sum={ic_sum:.6f})")
    
    logger.debug(f"Model configuration validated successfully for {model_type}")
    
    # Validate that learnable parameters are in the physics model
    learnable_param_dict = config.get("training", {}).get("learnable_parameters", {})
    model_params = config.get("compartmental", {}).get("model", {}).get("parameters", {})
    
    for param_name in learnable_param_dict.keys():
        if param_name not in model_params:
            raise ValueError(f"Learnable parameter '{param_name}' not found in model parameters. "
                             f"Available parameters: {list(model_params.keys())}")


def _validate_parameters(parameters: Dict[str, Any], model_type: str) -> None:
    """Validate model parameters.
    
    Args:
        parameters: Dictionary of model parameters
        model_type: Type of model being validated
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a dictionary")
    
    for param_name, param_value in parameters.items():
        # Check if parameter is numeric
        if not isinstance(param_value, (int, float)):
            raise ValueError(f"Parameter '{param_name}' must be numeric, got {type(param_value)}")
        
        # Check if parameter is non-negative
        if param_value < 0:
            raise ValueError(f"Parameter '{param_name}' must be non-negative, got {param_value}")
        
        # Check for reasonable parameter ranges (these are general guidelines)
        if param_name == 'beta' and param_value > 10.0:
            raise ValueError(f"Parameter 'beta' seems too large ({param_value}), typical values are < 10.0")
        
        if param_name in ['gamma', 'sigma', 'mu', 'nu'] and param_value > 5.0:
            raise ValueError(f"Parameter '{param_name}' seems too large ({param_value}), typical values are < 5.0")


def _validate_initial_conditions(initial_conditions: list, model_type: str) -> None:
    """Validate initial conditions.
    
    Args:
        initial_conditions: List of initial conditions
        model_type: Type of model being validated
        
    Raises:
        ValueError: If initial conditions are invalid
    """
    if not isinstance(initial_conditions, list):
        raise ValueError("Initial conditions must be a list")
    
    if len(initial_conditions) == 0:
        raise ValueError("Initial conditions list cannot be empty")
    
    for i, ic_value in enumerate(initial_conditions):
        # Check if initial condition is numeric
        if not isinstance(ic_value, (int, float)):
            raise ValueError(f"Initial condition at index {i} must be numeric, got {type(ic_value)}")
        
        # Check if initial condition is within valid range [0, 1]
        if ic_value < 0 or ic_value > 1:
            raise ValueError(f"Initial condition at index {i} must be between 0 and 1, got {ic_value}")


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
    # First check if network config is directly in config
    network_config = config.get('network', {})
    
    # Handle potential nested network configuration (the bug we're fixing)
    # If the network config has a 'network' key, that means it's nested incorrectly
    if isinstance(network_config, dict) and 'network' in network_config:
        network_config = network_config['network']
    
    # If network_config is still nested, try to flatten it
    while isinstance(network_config, dict) and 'network' in network_config:
        network_config = network_config['network']
        
    return network_config


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
