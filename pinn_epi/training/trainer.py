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
            torch.Tensor: Collocation points with shape (n_points, 1).
        """
        t_min, t_max = t_range
        # Sample random points in the interval and reshape to (n_points, 1)
        points = (t_min + (t_max - t_min) * torch.rand(n_points)).to(self.device)
        return points.view(-1, 1)

    def compute_loss(self, t_data: torch.Tensor, y_data: torch.Tensor, collocation_points: torch.Tensor, target_compartments: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the total loss for the PINN model.

        Args:
            t_data: Time points for the empirical data with shape (N, 1).
            y_data: Ground truth data for target compartments with shape (N, C).
            collocation_points: Collocation points for physics loss with shape (M, 1).
            target_compartments: List of compartment names to train on.

        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        # Data loss
        y_pred_data = self.model(t_data)
        
        # Check if model produced valid output
        if y_pred_data.nelement() == 0:
            raise ValueError("Model produced empty output. Check model initialization and input tensor dimensions.")
        
        # Select only the target compartments for data loss calculation
        compartment_indices = [self.physics_model.compartment_names.index(comp) for comp in target_compartments]
        
        # Check if indices are valid
        if not compartment_indices:
            raise ValueError("No valid compartment indices found. Check target_compartments configuration.")
            
        # Check if y_pred_data has the expected number of columns
        if y_pred_data.shape[1] < max(compartment_indices) + 1:
            raise ValueError(f"Model output has {y_pred_data.shape[1]} compartments, but requested index {max(compartment_indices)}")
            
        y_pred_selected = y_pred_data[:, compartment_indices]
        data_loss = torch.mean((y_pred_selected - y_data) ** 2)

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
        
        # Check if we have data
        if len(t_array) == 0:
            raise ValueError("No time data provided for training.")
        
        # Extract target compartment data
        y_true_list = [self.data[comp] for comp in target_compartments]
        y_true_array = np.column_stack(y_true_list)
        
        # Check if we have target data
        if y_true_array.size == 0:
            raise ValueError("No target compartment data found.")
        
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
