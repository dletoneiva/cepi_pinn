"""Training orchestrator for PINN models."""

import torch
import numpy as np
import logging
from pinn_epi.models.networks import ModularPINN
from pinn_epi.models.physics import CompartmentalModel
from pinn_epi.data.generator import ODESimulator
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

    def sample_collocation_points(self, t_range, n_points):
        """
        Sample collocation points for physics loss evaluation.

        Args:
            t_range: Tuple (t_min, t_max) specifying the time range.
            n_points: Number of collocation points to sample.

        Returns:
            torch.Tensor: Collocation points.
        """
        t_min, t_max = t_range
        return torch.linspace(t_min, t_max, n_points).to(self.device)

    def compute_loss(self, t, y_true, collocation_points):
        """
        Compute the total loss for the PINN model.

        Args:
            t: Time points for the empirical data.
            y_true: Ground truth data.
            collocation_points: Collocation points for physics loss.

        Returns:
            torch.Tensor: Total loss.
        """
        t = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(self.device)
        y_true = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        collocation_points = collocation_points.view(-1, 1)

        # Data loss
        y_pred = self.model(t)
        data_loss = torch.mean((y_pred - y_true) ** 2)

        # Physics loss
        y_pred_colloc = self.model(collocation_points)
        t_colloc = collocation_points.clone().requires_grad_(True)
        y_pred_colloc = self.model(t_colloc)
        y_pred_colloc = y_pred_colloc.view(-1, y_pred_colloc.shape[-1])
        residuals = self.physics_model.get_derivatives(t_colloc, y_pred_colloc, self.config['physics_params'])
        physics_loss = torch.mean(residuals ** 2)

        # Total loss
        total_loss = data_loss + self.config['lambda_phys'] * physics_loss

        return total_loss, data_loss, physics_loss

    def train(self):
        """
        Train the PINN model using Adam and L-BFGS optimizers.
        """
        optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=self.config['adam_lr'])
        optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(), max_iter=self.config['lbfgs_max_iter'], line_search_fn="strong_wolfe")

        t = self.data['t']
        y_true = self.data['y_true']
        collocation_points = self.sample_collocation_points((t[0], t[-1]), self.config['n_collocation_points'])

        # Adam phase
        for epoch in range(self.config['adam_epochs']):
            optimizer_adam.zero_grad()
            total_loss, data_loss, physics_loss = self.compute_loss(t, y_true, collocation_points)
            total_loss.backward()
            optimizer_adam.step()

            if epoch % self.config['log_interval'] == 0:
                logger.info(f"Epoch {epoch}/{self.config['adam_epochs']} - Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}")
                mlflow.log_metric("total_loss", total_loss.item(), step=epoch)
                mlflow.log_metric("data_loss", data_loss.item(), step=epoch)
                mlflow.log_metric("physics_loss", physics_loss.item(), step=epoch)

        # L-BFGS phase
        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _ = self.compute_loss(t, y_true, collocation_points)
            total_loss.backward()
            return total_loss

        optimizer_lbfgs.step(closure)

        logger.info("Training completed successfully")

if __name__ == "__main__":
    # Example usage
    from pinn_epi.models.networks import BaseMLP, HardICAnsatz, ModularPINN
    from pinn_epi.models.physics import SIRModel
    from pinn_epi.data.generator import ODESimulator

    # Example configuration
    config = {
        'adam_lr': 1e-3,
        'adam_epochs': 5000,
        'lbfgs_max_iter': 500,
        'n_collocation_points': 100,
        'lambda_phys': 1.0,
        'log_interval': 100,
        'physics_params': {'beta': 0.4, 'gamma': 0.1},
        'seed': 42
    }

    # Example data
    t_max = 30
    t_eval = np.linspace(0, t_max, 150)
    y0 = [0.99, 0.01, 0.00]
    physics_model = SIRModel()
    simulator = ODESimulator(physics_model)
    data = simulator.generate([0, t_max], y0, config['physics_params'], t_eval)

    # Example model
    backbone = BaseMLP(input_dim=1, output_dim=3, hidden_dims=[128, 128, 128])
    ansatz = HardICAnsatz(t0=0.0, initial_conditions=y0)
    model = ModularPINN(encoder=None, backbone=backbone, ansatz=ansatz)

    # Initialize and train the PINNTrainer
    trainer = PINNTrainer(model, physics_model, data, config)
    trainer.train()
```

### Updated `pinn_epi/experiments/train_pinn.py`

pinn_epi/experiments/train_pinn.py
