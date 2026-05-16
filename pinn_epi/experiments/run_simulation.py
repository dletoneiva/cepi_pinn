"""Run simulations from configuration files."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, Any
import mlflow

from pinn_epi.configs.model_loader import (
    validate_model_config, 
    create_model_from_config, 
    get_model_parameters,
    get_initial_conditions,
    get_network_config,
    get_training_config,
    get_simulation_config,
    get_plotting_config,
    get_data_config
)
from pinn_epi.analysis.evaluator import solve_compartmental_model
from pinn_epi.analysis.plotting import plot_compartmental_solution
from pinn_epi.analysis.data_wrangler import save_simulation_data
from pinn_epi.training.trainer import PINNTrainer
from pinn_epi.models.networks import ModularPINN

# Set up logging
logger = logging.getLogger(__name__)

def run_simulation_from_config(config: DictConfig) -> Dict[str, Any]:
    """Run a simulation based on the provided configuration.
    
    Args:
        config: Configuration dictionary from Hydra
        
    Returns:
        Dictionary containing the model, trajectories, and figure path (if saved)
    """
    logger.info("Starting simulation from configuration")
    
    # Start MLflow run for tracking simulation
    if not mlflow.active_run():
        mlflow.start_run(run_name=f"simulation_{config.compartmental.model.type}")
    
    try:
        # Convert DictConfig to regular dict for compatibility
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # Log configuration parameters
        mlflow.log_params({
            "model_type": config.compartmental.model.type,
            "t_span_start": config.compartmental.simulation.t_span[0],
            "t_span_end": config.compartmental.simulation.t_span[1],
            "t_eval_points": config.compartmental.simulation.t_eval_points
        })
        
        # Log model parameters
        model_params = get_model_parameters(config_dict)
        mlflow.log_params(model_params)
        
        # Log initial conditions
        initial_conditions = get_initial_conditions(config_dict)
        mlflow.log_param("initial_conditions", str(initial_conditions))
        
        # Validate configuration
        validate_model_config(config_dict)
        
        # Create model
        model = create_model_from_config(config_dict)
        logger.info(f"Created model: {type(model).__name__}")
        
        # Extract parameters using helper functions
        params = get_model_parameters(config_dict)
        y0 = get_initial_conditions(config_dict)
        simulation_config = get_simulation_config(config_dict)
        
        # Extract simulation parameters
        t_span = simulation_config["t_span"]
        t_eval_points = simulation_config["t_eval_points"]
        
        logger.info(f"Simulation parameters - t_span: {t_span}, t_eval_points: {t_eval_points}")
        logger.info(f"Model parameters: {params}")
        logger.info(f"Initial conditions: {y0}")
        
        # Create t_eval
        t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
        
        # Solve ODE
        logger.info("Solving ODE system")
        trajectories = solve_compartmental_model(
            model=model,
            t_span=t_span,
            y0=y0,
            params=params,
            t_eval=t_eval,
        )
        logger.info("ODE system solved successfully")
        
        # Plot solution if requested
        result = {
            "model": model,
            "trajectories": trajectories,
            "t": t_eval
        }
        
        # Handle plotting
        plotting_config = get_plotting_config(config_dict)
        show_plot = plotting_config.get("show_plot", True)
        save_plot = plotting_config.get("save_plot", False)
        
        logger.info(f"Plotting configuration - show_plot: {show_plot}, save_plot: {save_plot}")
        
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
            logger.info(f"Figure will be saved to: {save_path}")
        
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
            logger.info(f"Figure saved to: {save_path}")
        
        # Handle data saving
        data_config = get_data_config(config_dict)
        save_data = data_config.get("save_data", True)
        
        logger.info(f"Data saving configuration - save_data: {save_data}")
        
        if save_data:
            # Use Hydra's output directory for data as well
            hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            timestamp = datetime.now().strftime("%Y%mdd_%H%M%S")
            model_type = config_dict["compartmental"]["model"]["type"]
            
            # Ensure the directory exists
            os.makedirs(hydra_output_dir, exist_ok=True)
            
            # Save simulation data
            logger.info(f"Saving simulation data to: {hydra_output_dir}")
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
            logger.info(f"Data saved to: {hydra_output_dir}")
        
        # Extract network and training configurations
        network_config = get_network_config(config_dict)
        training_config = get_training_config(config_dict)
        
        result["network_config"] = network_config
        result["training_config"] = training_config
        
        # Run PINN training if requested
        if training_config.get("run_training", False):
            logger.info("Starting PINN training with generated simulation data")
            
            # Prepare data for training
            training_data = {
                't': t_eval
            }
            
            # Add compartment data
            target_compartments = training_config.get("target_compartments", model.compartment_names)
            for comp in target_compartments:
                training_data[comp] = trajectories[comp].flatten()
            
            # Create PINN model
            input_dim = 1  # time dimension
            output_dim = len(model.compartment_names)
            
            # Create network backbone
            backbone_config = network_config.get("backbone", {})
            hidden_dims = backbone_config.get("hidden_dims", [50, 50, 50])
            activation_name = backbone_config.get("activation", "Tanh")
            
            # Map activation name to class
            activation_map = {
                "Tanh": torch.nn.Tanh,
                "ReLU": torch.nn.ReLU,
                "Sigmoid": torch.nn.Sigmoid
            }
            activation = activation_map.get(activation_name, torch.nn.Tanh)
            
            from pinn_epi.models.networks import BaseMLP
            
            backbone = BaseMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation
            )
            
            # Create encoder if specified
            encoder = None
            if network_config.get("use_time_normalization", False):
                from pinn_epi.models.networks import TimeNormalizationEncoder
                encoder = TimeNormalizationEncoder(
                    t0=float(t_span[0]),
                    t1=float(t_span[1])
                )
            
            # Create initial condition head if specified
            head = None
            if network_config.get("use_hard_ic", False):
                from pinn_epi.models.networks import HardICHead
                initial_conditions_tensor = torch.tensor(y0, dtype=torch.float32)
                head = HardICHead(
                    initial_conditions=initial_conditions_tensor,
                    t0=float(t_span[0]),
                    t1=float(t_span[1])
                )
            
            # Create modular PINN
            pinn_model = ModularPINN(
                backbone=backbone,
                head=head,
                encoder=encoder
            )
            
            # Create trainer
            trainer = PINNTrainer(
                model=pinn_model,
                physics_model=model,  # The compartmental model for physics loss
                data=training_data,
                config=training_config
            )
            
            # Run training
            trainer.train()
            
            result["trained_pinn"] = pinn_model
            logger.info("PINN training completed successfully")
        
        logger.info("Simulation run completed successfully")
        return result
        
    finally:
        # End MLflow run if we started it
        if mlflow.active_run():
            mlflow.end_run()


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Run simulation based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    logger.info("Starting simulation run")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Run simulation
    try:
        result = run_simulation_from_config(cfg)
        logger.info(f"Simulation completed successfully for {cfg.compartmental.model.type}")
        logger.info(f"Model parameters: {OmegaConf.to_container(cfg.compartmental.model.parameters, resolve=True)}")
        logger.info(f"Initial conditions: {cfg.compartmental.model.initial_conditions}")
    except Exception as e:
        logger.error(f"Simulation failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
