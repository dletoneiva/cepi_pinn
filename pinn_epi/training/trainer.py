"""Training orchestrator for PINN models."""

import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pinn_epi.models.networks import ModularPINN
from pinn_epi.models.physics import CompartmentalModel
import mlflow
import mlflow.pytorch

# Set up logging
logger = logging.getLogger(__name__)

class PINNTrainer:
    def __init__(self, model: ModularPINN, physics_model: CompartmentalModel, data: dict, config: dict):
        """
        Initialize the PINNTrainer.

        Args:
            model: The PINN model to train.
            physics_model: The physics model used for calculating residuals.
            data: A dictionary containing the empirical data.
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

    def compute_loss(self, t_data: torch.Tensor, y_data: torch.Tensor, collocation_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the total loss for the PINN model using data mapping.

        Args:
            t_data: Time points for the empirical data with shape (N, 1).
            y_data: Ground truth data with shape (N, C) where C is number of mapped observables.
            collocation_points: Collocation points for physics loss with shape (M, 1).

        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        # Get data mapping from config
        data_mapping = self.config.get('data_mapping', {})
        
        if not data_mapping:
            raise ValueError("No data mapping provided in configuration")
        
        # Data loss
        # Get full state prediction from the model
        y_pred_data = self.model(t_data)
        
        # Check if model produced valid output
        if y_pred_data.nelement() == 0:
            raise ValueError("Model produced empty output. Check model initialization and input tensor dimensions.")
        
        # Get observables from physics model
        physics_params = self.config.get('physics_params', {})
        observables = self.physics_model.get_observables(t_data, y_pred_data, physics_params)
        
        # Calculate data loss using data mapping
        data_losses = []
        for i, (csv_column, observable_name) in enumerate(data_mapping.items()):
            if observable_name not in observables:
                raise ValueError(f"Observable '{observable_name}' not found in model observables")
            
            # Get the observable tensor and ensure proper shape
            observable_tensor = observables[observable_name]
            if observable_tensor.dim() == 1:
                observable_tensor = observable_tensor.view(-1, 1)
            
            # Get corresponding data column
            y_data_column = y_data[:, i:i+1]  # Keep as column vector
            
            # Calculate MSE for this observable
            data_losses.append(torch.mean((observable_tensor - y_data_column) ** 2))
        
        # Average data loss across all mapped observables
        data_loss = torch.stack(data_losses).mean()
        
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
        
        # Get physics residuals - use physics_params from config if provided, otherwise use empty dict
        physics_params = self.config.get('physics_params', {})
        physics_residuals = self.physics_model.get_derivatives(t_phys, y_pred_phys, physics_params)
        physics_loss = torch.mean((du_dt - physics_residuals) ** 2)

        # Total loss
        total_loss = self.config.get('data_weight', 1.0) * data_loss + self.config.get('physics_weight', 1.0) * physics_loss

        return total_loss, data_loss, physics_loss

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
                
                # Log model and training parameters
                try:
                    mlflow.log_params({
                        "model_type": type(self.physics_model).__name__,
                        "compartment_names": str(self.physics_model.compartment_names),
                        "adam_lr": self.config.get('adam_lr', 1e-3),
                        "adam_epochs": self.config.get('adam_epochs', 5000),
                        "n_collocation_points": self.config.get('n_collocation_points', 100),
                        "data_weight": self.config.get('data_weight', 1.0),
                        "physics_weight": self.config.get('physics_weight', 1.0),
                        "data_mapping": str(self.config.get('data_mapping', {}))
                    })
                    logger.info("Logged training parameters to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log parameters to MLflow: {e}")
                
                # Log physics parameters if available
                try:
                    physics_params = self.config.get('physics_params', {})
                    if physics_params:
                        mlflow.log_params(physics_params)
                        logger.info("Logged physics parameters to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log physics parameters to MLflow: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to set up MLflow: {e}")
                self.mlflow_enabled = False  # Disable MLflow for the rest of training
        else:
            logger.info("MLflow logging is disabled")
            
        try:
            # Prepare data
            t_array = self.data['t']
            
            # Get data mapping from config
            data_mapping = self.config.get('data_mapping', {})
            
            if not data_mapping:
                raise ValueError("No data mapping provided in configuration")
            
            # Check if we have data
            if len(t_array) == 0:
                raise ValueError("No time data provided for training.")
            
            # Extract data columns based on data mapping keys
            y_true_list = [self.data[csv_column] for csv_column in data_mapping.keys()]
            y_true_array = np.column_stack(y_true_list)
            
            # Check if we have target data
            if y_true_array.size == 0:
                raise ValueError("No target data found based on data mapping.")
            
            # Convert to tensors
            t_tensor = torch.tensor(t_array, dtype=torch.float32).view(-1, 1).to(self.device)
            y_true_tensor = torch.tensor(y_true_array, dtype=torch.float32).to(self.device)
            
            # Check tensor dimensions
            if t_tensor.nelement() == 0:
                raise ValueError("Time tensor is empty after conversion.")
                
            if y_true_tensor.nelement() == 0:
                raise ValueError("Target data tensor is empty after conversion.")
            
            # Optimizers
            optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=self.config.get('adam_lr', 1e-3))
            optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(), 
                                              max_iter=self.config.get('lbfgs_max_iter', 100), 
                                              line_search_fn="strong_wolfe")

            # Training parameters
            adam_epochs = self.config.get('adam_epochs', 5000)
            n_collocation_points = self.config.get('n_collocation_points', 100)
            log_interval = self.config.get('log_interval', 100)
            
            # Sample collocation points
            t_min, t_max = t_array.min(), t_array.max()
            collocation_points = self.sample_collocation_points((t_min, t_max), n_collocation_points)
            
            # Check collocation points
            if collocation_points.nelement() == 0:
                raise ValueError("No collocation points generated.")

            # Adam phase
            for epoch in range(adam_epochs):
                optimizer_adam.zero_grad()
                total_loss, data_loss, physics_loss = self.compute_loss(t_tensor, y_true_tensor, collocation_points)
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

            # L-BFGS phase
            def closure():
                optimizer_lbfgs.zero_grad()
                total_loss, _, _ = self.compute_loss(t_tensor, y_true_tensor, collocation_points)
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
