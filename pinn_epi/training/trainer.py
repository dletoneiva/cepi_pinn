"""Training orchestrator for PINN models."""

import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pinn_epi.models.networks import ModularPINN
from pinn_epi.models.physics import CompartmentalModel
import mlflow

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

    def sample_collocation_points(self, t_range: Tuple[float, float], n_points: int) -> torch.Tensor:
        """
        Sample collocation points for physics loss evaluation.

        Args:
            t_range: Tuple (t_min, t_max) specifying the time range.
            n_points: Number of collocation points to sample.

        Returns:
            torch.Tensor: Collocation points.
        """
        t_min, t_max = t_range
        # Sample random points in the interval
        return (t_min + (t_max - t_min) * torch.rand(n_points)).to(self.device)

    def compute_loss(self, t_data: torch.Tensor, y_data: torch.Tensor, collocation_points: torch.Tensor, target_compartments: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the total loss for the PINN model.

        Args:
            t_data: Time points for the empirical data.
            y_data: Ground truth data for target compartments.
            collocation_points: Collocation points for physics loss.
            target_compartments: List of compartment names to train on.

        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        # Data loss
        y_pred_data = self.model(t_data)
        
        # Select only the target compartments for data loss calculation
        compartment_indices = [self.physics_model.compartment_names.index(comp) for comp in target_compartments]
        y_pred_selected = y_pred_data[:, compartment_indices]
        data_loss = torch.mean((y_pred_selected - y_data) ** 2)

        # Physics loss
        t_phys = collocation_points.clone().requires_grad_(True)
        y_pred_phys = self.model(t_phys)
        
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
        
        # Get physics residuals
        physics_residuals = self.physics_model.get_derivatives(t_phys, y_pred_phys, self.config.get('physics_params', {}))
        physics_loss = torch.mean((du_dt - physics_residuals) ** 2)

        # Total loss
        total_loss = self.config.get('data_weight', 1.0) * data_loss + self.config.get('physics_weight', 1.0) * physics_loss

        return total_loss, data_loss, physics_loss

    def train(self):
        """
        Train the PINN model using Adam and L-BFGS optimizers.
        """
        # Prepare data
        t_array = self.data['t']
        target_compartments = self.config.get('target_compartments', self.physics_model.compartment_names)
        
        # Extract target compartment data
        y_true_list = [self.data[comp] for comp in target_compartments]
        y_true_array = np.column_stack(y_true_list)
        
        # Convert to tensors
        t_tensor = torch.tensor(t_array, dtype=torch.float32).view(-1, 1).to(self.device)
        y_true_tensor = torch.tensor(y_true_array, dtype=torch.float32).to(self.device)
        
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

        # Adam phase
        for epoch in range(adam_epochs):
            optimizer_adam.zero_grad()
            total_loss, data_loss, physics_loss = self.compute_loss(t_tensor, y_true_tensor, collocation_points, target_compartments)
            total_loss.backward()
            optimizer_adam.step()

            if epoch % log_interval == 0:
                logger.info(f"Epoch {epoch}/{adam_epochs} - Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}")
                # Log metrics with MLflow
                try:
                    mlflow.log_metric("total_loss", total_loss.item(), step=epoch)
                    mlflow.log_metric("data_loss", data_loss.item(), step=epoch)
                    mlflow.log_metric("physics_loss", physics_loss.item(), step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log metrics to MLflow: {e}")

        # L-BFGS phase
        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _ = self.compute_loss(t_tensor, y_true_tensor, collocation_points, target_compartments)
            total_loss.backward()
            return total_loss

        optimizer_lbfgs.step(closure)

        logger.info("Training completed successfully")

# Example usage moved to experiments/train_pinn.py
