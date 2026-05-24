"""Training orchestrator for PINN models."""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pinn_epi.models.networks import ModularPINN
from pinn_epi.models.physics import CompartmentalModel
import mlflow
import mlflow.pytorch
import hydra
import os

# Set up logging
logger = logging.getLogger(__name__)

class PINNTrainer:
    def __init__(self, model: ModularPINN, physics_model: CompartmentalModel, data: dict, config: dict):
        """
        Initialize the PINNTrainer.

        Args:
            model: The PINN model to train.
            physics_model: The physics model used for calculating residuals.
            data: A dictionary containing the empirical data with compartment names as keys.
            config: Configuration dictionary for training parameters.
        """
        self.model = model
        self.physics_model = physics_model
        self.data = data
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check MLflow configuration
        self.mlflow_config = self.config.get('mlflow', {})
        self.mlflow_enabled = self.mlflow_config.get('enabled', True)
        
        # Initialize learnable parameters
        self.learnable_params = nn.ParameterDict()
        learnable_param_names = self.config.get("learnable_parameters", [])

        # Validate and process learnable parameters
        for param_name in learnable_param_names:
            if param_name in self.physics_model.parameters:
                # Extract the initial value from the model's parameters
                base_value = self.physics_model.parameters[param_name]
                # Create a learnable parameter
                p = nn.Parameter(torch.tensor(base_value, dtype=torch.float32, device=self.device))
                # Store in our learnable parameters
                self.learnable_params[param_name] = p
                # Replace the value in the physics model's parameters dictionary
                self.physics_model.parameters[param_name] = p
                logger.info(f"Parameter '{param_name}' is learnable with initial value {base_value}")
            else:
                # Parameter not found in model but marked as learnable - raise error
                raise ValueError(f"Parameter '{param_name}' marked as learnable but not found in model parameters. "
                                 f"Available parameters: {list(self.physics_model.parameters.keys())}")

    def sample_collocation_points(self, t_range: Tuple[float, float], n_points: int) -> torch.Tensor:
        """
        Sample collocation points for physics loss evaluation.

        Args:
            t_range: Tuple (t_min, t_max) specifying the time range.
            n_points: Number of collocation points to sample.

        Returns:
            torch.Tensor: Collocation points with shape (n_points, 1).
        """
        t_min, t_max = t_range
        # Sample random points in the interval and reshape to (n_points, 1)
        points = (t_min + (t_max - t_min) * torch.rand(n_points)).to(self.device)
        return points.view(-1, 1)

    def compute_loss(self, t_data: torch.Tensor, y_data_dict: dict, collocation_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the total loss for the PINN model.

        Args:
            t_data: Time points for the empirical data with shape (N, 1).
            y_data_dict: Dictionary mapping compartment names to ground truth data.
            collocation_points: Collocation points for physics loss with shape (M, 1).

        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        # Data loss - compare model predictions with observed data
        y_pred_data = self.model(t_data)
        
        # Check if model produced valid output
        if y_pred_data.nelement() == 0:
            raise ValueError("Model produced empty output. Check model initialization and input tensor dimensions.")
        
        # Get the compartment names from the physics model
        compartment_names = self.physics_model.compartment_names
        
        # Calculate data loss (MSE between predicted and actual values for observed compartments)
        data_losses = []
        for i, compartment in enumerate(compartment_names):
            if compartment in y_data_dict:
                # Only compute loss for observed compartments
                pred = y_pred_data[:, i]
                true = y_data_dict[ compartment].to(self.device)  # Ensure tensor is on the correct device
                compartment_loss = torch.mean((pred - true) ** 2)
                data_losses.append(compartment_loss)
        
        if not data_losses:
            raise ValueError("No observed compartments found in data for loss computation.")
            
        data_loss = torch.mean(torch.stack(data_losses))
        
        # Physics loss
        t_phys = collocation_points.clone().requires_grad_(True)
        y_pred_phys = self.model(t_phys)
        
        # Check if physics model produced valid output
        if y_pred_phys.nelement() == 0:
            raise ValueError("Model produced empty output for physics points. Check model initialization.")
        
        # Compute derivatives using autograd
        du_dt = []
        for i in range(y_pred_phys.shape[1]):
            grad = torch.autograd.grad(
                y_pred_phys[:, i], 
                t_phys, 
                grad_outputs=torch.ones_like(y_pred_phys[:, i]),
                create_graph=True
            )[0]
            du_dt.append(grad)
        du_dt = torch.cat(du_dt, dim=1)
        
        # Get physics residuals - use the model's internal parameters dictionary
        # The parameters dictionary has been updated to point to learnable parameters
        physics_residuals = self.physics_model.get_derivatives(t_phys, y_pred_phys, self.physics_model.parameters)
        physics_loss = torch.mean((du_dt - physics_residuals) ** 2)

        # Total loss
        total_loss = self.config.get('data_weight', 1.0) * data_loss + self.config.get('physics_weight', 1.0) * physics_loss

        return total_loss, data_loss, physics_loss

    def save_model_for_inference(self):
        """Save the trained model for later inference."""
        try:
            # Get Hydra output directory
            hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            model_save_path = os.path.join(hydra_output_dir, "trained_pinn_model.pth")
            
            # Save model state dict
            torch.save(self.model.state_dict(), model_save_path)
            logger.info(f"Saved model for inference to: {model_save_path}")
            
            # Also save the full model for easier loading
            full_model_path = os.path.join(hydra_output_dir, "trained_pinn_model_full.pth")
            torch.save(self.model, full_model_path)
            logger.info(f"Saved full model to: {full_model_path}")
            
            return model_save_path
        except Exception as e:
            logger.warning(f"Failed to save model for inference: {e}")
            return None

    def train(self):
        """
        Train the PINN model using Adam and L-BFGS optimizers.
        """
        # MLflow setup
        should_end_run = False
        if self.mlflow_enabled:
            try:
                # Start MLflow run if not already active
                if not mlflow.active_run():
                    experiment_name = self.mlflow_config.get('experiment_name', 'pinn_training')
                    mlflow.set_experiment(experiment_name)
                    mlflow.start_run()
                    should_end_run = True
                    logger.info(f"Started new MLflow run in experiment: {experiment_name}")
                else:
                    logger.info("Using existing MLflow run")
                
                # Log detailed model and training parameters
                try:
                    # Physics model information
                    mlflow.log_params({
                        "model_type": type(self.physics_model).__name__,
                        "compartment_names": str(self.physics_model.compartment_names),
                        "ground_truth_parameters": str(self.physics_model.parameters),
                        "initial_conditions": str(self.config.get('initial_conditions', [])),
                    })
                    
                    # Network architecture details
                    network_config = self.config.get('network', {})
                    mlflow.log_params({
                        "network_layers": len(network_config.get('hidden_dims', [])),
                        "layer_size": str(network_config.get('hidden_dims', [])),
                        "dropout": network_config.get('dropout', 0.0),
                        "encoder_type": type(self.model.encoder).__name__ if self.model.encoder else "None",
                        "head_type": type(self.model.head).__name__ if self.model.head else "None",
                    })
                    
                    # Observables configuration
                    observables_config = self.config.get('observables', {})
                    mlflow.log_params({
                        "observable_type": observables_config.get('type', 'synthetic'),
                        "t_max": observables_config.get('t_max', 'N/A'),
                        "n_points": observables_config.get('n_points', 'N/A'),
                        "noise": observables_config.get('noise', 0.0),
                        "t_span": str(observables_config.get('t_span', 'N/A')),
                        "t_eval": str(observables_config.get('t_eval', 'N/A')),
                        "observed_variables": str(observables_config.get('observed_variables', [])),
                    })
                    
                    # Training parameters
                    mlflow.log_params({
                        "adam_lr": self.config.get('adam_lr', 1e-3),
                        "adam_epochs": self.config.get('adam_epochs', 5000),
                        "lbfgs_max_iter": self.config.get('lbfgs_max_iter', 100),
                        "n_collocation_points": self.config.get('n_collocation_points', 100),
                        "data_weight": self.config.get('data_weight', 1.0),
                        "physics_weight": self.config.get('physics_weight', 1.0),
                        "learnable_parameters": str(self.config.get('learnable_parameters', []))
                    })
                    logger.info("Logged detailed training parameters to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log parameters to MLflow: {e}")
            except Exception as e:
                logger.warning(f"Failed to set up MLflow: {e}")
                self.mlflow_enabled = False  # Disable MLflow for the rest of training
        else:
            logger.info("MLflow logging is disabled")
            
        try:
            # Prepare data - extract time tensor and observed data dictionary
            t_array = self.data['t']
            
            # Check if we have data
            if len(t_array) == 0:
                raise ValueError("No time data provided for training.")
            
            # Convert to tensors and ensure they're on the correct device - fix tensor construction warning
            if torch.is_tensor(t_array):
                t_tensor = t_array.clone().detach().to(dtype=torch.float32).view(-1, 1).to(self.device)
            else:
                t_tensor = torch.as_tensor(t_array, dtype=torch.float32).view(-1, 1).to(self.device)
            
            # Check tensor dimensions
            if t_tensor.nelement() == 0:
                raise ValueError("Time tensor is empty after conversion.")
                
            # Optimizers - include both network parameters and learnable physics parameters
            trainable_vars = list(self.model.parameters()) + list(self.learnable_params.parameters())
            optimizer_adam = torch.optim.Adam(trainable_vars, lr=self.config.get('adam_lr', 1e-3))
            optimizer_lbfgs = torch.optim.LBFGS(trainable_vars, 
                                              max_iter=self.config.get('lbfgs_max_iter', 100), 
                                              line_search_fn="strong_wolfe")

            # Training parameters
            adam_epochs = self.config.get('adam_epochs', 5000)
            n_collocation_points = self.config.get('n_collocation_points', 100)
            log_interval = 50  # Fixed log interval for MLflow
            
            # Sample collocation points
            t_min, t_max = t_array.min(), t_array.max()
            collocation_points = self.sample_collocation_points((t_min, t_max), n_collocation_points)
            
            # Check collocation points
            if collocation_points.nelement() == 0:
                raise ValueError("No collocation points generated.")

            # Adam phase
            for epoch in range(adam_epochs):
                optimizer_adam.zero_grad()
                total_loss, data_loss, physics_loss = self.compute_loss(t_tensor, self.data, collocation_points)
                total_loss.backward()
                optimizer_adam.step()

                if epoch % log_interval == 0:
                    logger.info(f"Epoch {epoch}/{adam_epochs} - Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}")
                    # Log metrics with MLflow
                    if self.mlflow_enabled:
                        try:
                            if mlflow.active_run():
                                mlflow.log_metrics({
                                    "total_loss": total_loss.item(),
                                    "data_loss": data_loss.item(),
                                    "physics_loss": physics_loss.item()
                                }, step=epoch)
                                logger.debug(f"Logged metrics to MLflow at epoch {epoch}")
                        except Exception as e:
                            logger.warning(f"Failed to log metrics to MLflow at epoch {epoch}: {e}")
                    
                    # Log learnable parameters if MLflow is enabled
                    if self.mlflow_enabled and len(self.learnable_params) > 0:
                        try:
                            if mlflow.active_run():
                                param_dict = {name: param.item() for name, param in self.learnable_params.items()}
                                mlflow.log_metrics(param_dict, step=epoch)
                                logger.debug(f"Logged learnable parameters to MLflow at epoch {epoch}")
                        except Exception as e:
                            logger.warning(f"Failed to log learnable parameters to MLflow at epoch {epoch}: {e}")

            # L-BFGS phase
            def closure():
                optimizer_lbfgs.zero_grad()
                total_loss, _, _ = self.compute_loss(t_tensor, self.data, collocation_points)
                total_loss.backward()
                return total_loss

            optimizer_lbfgs.step(closure)
            
            # Log the final model
            if self.mlflow_enabled:
                try:
                    if mlflow.active_run():
                        mlflow.pytorch.log_model(self.model, "final_model")
                        logger.info("Logged final model to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log model to MLflow: {e}")
            
            # Save model for inference
            model_save_path = self.save_model_for_inference()
            
            # Log model save path to MLflow if enabled
            if self.mlflow_enabled and model_save_path:
                try:
                    if mlflow.active_run():
                        mlflow.log_param("model_save_path", model_save_path)
                except Exception as e:
                    logger.warning(f"Failed to log model save path to MLflow: {e}")

            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # End MLflow run if we started it
            if self.mlflow_enabled and should_end_run and mlflow.active_run():
                try:
                    mlflow.end_run()
                    logger.info("Ended MLflow run")
                except Exception as e:
                    logger.warning(f"Failed to end MLflow run: {e}")
