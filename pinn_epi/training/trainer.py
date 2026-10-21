"""Training orchestrator for PINN models."""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, Any, Optional
from pinn_epi.models.networks import ModularPINN
from pinn_epi.models.physics import CompartmentalModel
import mlflow
import mlflow.pytorch
import hydra
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pinn_epi.utils.device_utils import DEVICE  # Import the centralized device

# Set up logging
logger = logging.getLogger(__name__)

class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks (PINNs) in epidemiological modeling.
    
    This class orchestrates the training of PINNs by combining empirical data loss
    with physics-based loss terms derived from compartmental models. The training
    process involves two phases:
    
    1. Adam optimization for a large number of epochs to find a good starting point
    2. L-BFGS optimization for fine-tuning the solution
    
    The physics loss is computed by ensuring that the neural network's derivatives
    match the right-hand side of the ODE system defined by the compartmental model.
    This approach allows the PINN to learn the underlying dynamics without requiring
    extensive training data.
    
    Attributes:
        model: The neural network model to be trained
        physics_model: The compartmental model defining the physics constraints
        data: Dictionary containing empirical training data
        config: Configuration parameters for training
        device: Computing device (CPU/GPU) for training
        learnable_params: Dictionary of parameters that are being optimized
        mlflow_enabled: Flag indicating if MLflow logging is active
    """

    _RESERVED_COLUMNS = frozenset({
        'total_loss',
        'data_loss',
        'physics_loss',
        'parameter',
        'value',
        'is_current',
        'phase',
        'epoch',
    })

    def __init__(self, model: ModularPINN, physics_model: CompartmentalModel, data: dict, config: dict):
        """Initialize the PINN trainer with model, physics constraints, and training data.
        
        Args:
            model: The PINN model to train
            physics_model: The physics model used for calculating residuals
            data: A dictionary containing the empirical data with compartment names as keys
            config: Configuration dictionary for training parameters
        """
        self.model = model
        self.physics_model = physics_model
        self.data = data
        self.config = config
        self.device = DEVICE  # Use the centralized device
        self.model.to(self.device)
        
        # Check MLflow configuration
        self.mlflow_config = self.config.get('mlflow', {})
        self.mlflow_enabled = self.mlflow_config.get('enabled', True)
        
        # Initialize learnable parameters
        self.learnable_params = nn.ParameterDict()
        learnable_param_dict = self.config.get("learnable_parameters", {})

        # Validate and process learnable parameters
        for param_name, initial_value in learnable_param_dict.items():
            if param_name in self.physics_model.parameters:
                logger.info(f"Using configured initial value {initial_value} for parameter '{param_name}'")
                
                # Create a learnable parameter
                p = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32, device=self.device))
                # Store in our learnable parameters
                self.learnable_params[param_name] = p
                # Replace the value in the physics model's parameters dictionary
                self.physics_model.parameters[param_name] = p
                logger.info(f"Parameter '{param_name}' is learnable with initial value {initial_value}")
            else:
                # Parameter not found in model but marked as learnable - raise error
                raise ValueError(f"Parameter '{param_name}' marked as learnable but not found in model parameters. "
                                 f"Available parameters: {list(self.physics_model.parameters.keys())}")

        # Combined loss-landscape artifact state for a single training run.
        self._loss_landscape_rows: list = []
        self._loss_landscape_fieldnames: list = []
        self._loss_landscape_plot_tasks: list = []
        self._loss_landscape_output_dir: Optional[str] = None
        self._loss_landscape_csv_path: Optional[str] = None
        self._loss_landscape_pdf_path: Optional[str] = None

    @staticmethod
    def _as_scalar_float(value: Any) -> float:
        """Convert a scalar-like parameter value to float, preserving NaN for invalid values."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float('nan')
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().cpu().reshape(-1)[0])

        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                array = np.asarray(value)
                if array.size == 0:
                    return float('nan')
                return float(array.reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                return float('nan')

    def _get_parameter_names(self) -> list:
        """Return ordered compartmental parameter names suitable for CSV/plot axes.

        For SIR this should return names such as ['beta', 'gamma'], preserving
        the order in which they are stored in the physics model.
        """
        names = []
        for name in self.physics_model.parameters.keys():
            if name in self._RESERVED_COLUMNS or name in names:
                continue
            names.append(name)
        return names

    def _get_parameter_values(self) -> Dict[str, float]:
        """Return the current full compartmental parameter set as scalar floats."""
        values: Dict[str, float] = {}
        for name, value in self.physics_model.parameters.items():
            if name in self._RESERVED_COLUMNS:
                continue
            values[name] = self._as_scalar_float(value)
        return values

    def _get_unique_parameter_names_from_results(self, results: list) -> list:
        """Fallback parameter names extracted from result rows."""
        names = []
        for result in results:
            name = result.get('parameter')
            if name and name not in self._RESERVED_COLUMNS and name not in names:
                names.append(name)
        return names

    def _close_loss_landscape_figures(self) -> None:
        """Clear any pending loss-landscape plot tasks."""
        self._loss_landscape_plot_tasks = []

    def _reset_loss_landscape_artifacts(self) -> None:
        """Reset combined loss-landscape artifact state for a new training run."""
        self._close_loss_landscape_figures()
        self._loss_landscape_rows = []
        self._loss_landscape_fieldnames = []
        self._loss_landscape_plot_tasks = []
        self._loss_landscape_output_dir = None
        self._loss_landscape_csv_path = None
        self._loss_landscape_pdf_path = None

    def _ensure_loss_landscape_paths(self) -> None:
        """Ensure combined loss-landscape output paths exist."""
        if self._loss_landscape_output_dir is None:
            hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            self._loss_landscape_output_dir = os.path.join(hydra_output_dir, "loss_landscape")
            os.makedirs(self._loss_landscape_output_dir, exist_ok=True)
            self._loss_landscape_csv_path = os.path.join(
                self._loss_landscape_output_dir,
                "loss_landscape.csv"
            )
            self._loss_landscape_pdf_path = os.path.join(
                self._loss_landscape_output_dir,
                "loss_landscape.pdf"
            )

    def _finalize_loss_landscape_artifacts(self) -> None:
        """Write all pending loss-landscape plots into one multi-page PDF.

        Plots are organized into two columns per PDF page. Each plot keeps the
        same full parallel-coordinate content; only the page organization changes.
        """
        if not self._loss_landscape_plot_tasks:
            return

        self._ensure_loss_landscape_paths()

        if self._loss_landscape_pdf_path is None:
            return

        try:
            with PdfPages(self._loss_landscape_pdf_path) as pdf:
                tasks = self._loss_landscape_plot_tasks

                for start_idx in range(0, len(tasks), 2):
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8), squeeze=False)

                    for col_idx in range(2):
                        ax = axes[0][col_idx]
                        task_idx = start_idx + col_idx

                        if task_idx < len(tasks):
                            task = tasks[task_idx]
                            self._draw_loss_landscape_axis(
                                ax,
                                task['results'],
                                task['checkpoint_info']
                            )
                        else:
                            ax.axis('off')

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            logger.info(f"Combined loss landscape PDF saved to: {self._loss_landscape_pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to save combined loss landscape PDF: {e}")
        finally:
            self._loss_landscape_plot_tasks = []

    def sample_collocation_points(self, t_range: Tuple[float, float], n_points: int) -> torch.Tensor:
        """Sample collocation points for physics loss evaluation.
        
        Collocation points are randomly sampled time points within the specified
        range where the physics loss (residual of the ODE system) is evaluated.
        These points are crucial for ensuring the PINN satisfies the differential
        equations across the entire time domain.
        
        Args:
            t_range: Tuple (t_min, t_max) specifying the time range
            n_points: Number of collocation points to sample
            
        Returns:
            torch.Tensor: Collocation points with shape (n_points, 1)
        """
        t_min, t_max = t_range
        # Sample random points in the interval and reshape to (n_points, 1)
        points = (t_min + (t_max - t_min) * torch.rand(n_points)).to(self.device)
        return points.view(-1, 1)

    def compute_loss(self, t_data: torch.Tensor, y_data_dict: dict, collocation_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the total loss for the PINN model.
        
        The total loss is a weighted combination of data loss and physics loss:
        
        - Data loss: Mean squared error between model predictions and observed data
        - Physics loss: Mean squared error of the residual of the ODE system
          (difference between neural network derivatives and model derivatives)
          
        This combined loss ensures the model both fits the data and respects the
        underlying physics.
        
        Args:
            t_data: Time points for the empirical data with shape (N, 1)
            y_data_dict: Dictionary mapping compartment names to ground truth data
            collocation_points: Collocation points for physics loss with shape (M, 1)
            
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
                true = y_data_dict[compartment].to(self.device)  # Ensure tensor is on the correct device
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

    def compute_residual_loss_landscape(self, t_data: torch.Tensor, collocation_points: torch.Tensor, 
                                     param_name: str, param_values: np.ndarray, 
                                     current_param_value: float, checkpoint_info: dict):
        """Compute loss landscape for a specific parameter around its current value.
        
        Args:
            t_data: Time points for the empirical data
            collocation_points: Collocation points for physics loss
            param_name: Name of the parameter to vary
            param_values: Array of parameter values to test
            current_param_value: Current value of the parameter
            checkpoint_info: Dictionary with phase and epoch information
        """
        # Store original parameter value
        original_value = self.physics_model.parameters[param_name]
        if isinstance(original_value, torch.Tensor):
            original_value = original_value.clone()
        
        # Get the original parameter tensor (could be learnable or fixed)
        param_tensor = self.physics_model.parameters[param_name]

        # Capture the full compartmental parameter set for each CSV row.
        all_param_names = self._get_parameter_names()
        
        # Results storage
        results = []
        
        # Evaluate loss at each parameter value
        for val in param_values:
            # Update parameter value
            if isinstance(param_tensor, nn.Parameter):
                param_tensor.data.fill_(val)
            else:
                self.physics_model.parameters[param_name] = torch.tensor(val, dtype=torch.float32, device=self.device)
            
            # Compute loss
            try:
                total_loss, data_loss, physics_loss = self.compute_loss(t_data, self.data, collocation_points)
                row = {
                    'parameter': param_name,
                    'value': float(val),
                    'total_loss': total_loss.item(),
                    'data_loss': data_loss.item(),
                    'physics_loss': physics_loss.item(),
                    'is_current': abs(val - current_param_value) < 1e-8,
                    'phase': checkpoint_info['phase'],
                    'epoch': checkpoint_info['epoch']
                }
                row.update(self._get_parameter_values())
                results.append(row)
            except Exception as e:
                logger.warning(f"Failed to compute loss for {param_name}={val}: {e}")
                row = {
                    'parameter': param_name,
                    'value': float(val),
                    'total_loss': float('nan'),
                    'data_loss': float('nan'),
                    'physics_loss': float('nan'),
                    'is_current': abs(val - current_param_value) < 1e-8,
                    'phase': checkpoint_info['phase'],
                    'epoch': checkpoint_info['epoch']
                }
                row.update(self._get_parameter_values())
                results.append(row)
        
        # Restore original parameter value
        if isinstance(param_tensor, nn.Parameter):
            if isinstance(original_value, torch.Tensor):
                param_tensor.data.copy_(original_value.data)
            else:
                param_tensor.data.fill_(original_value)
        else:
            self.physics_model.parameters[param_name] = original_value
            
        return results

    def save_loss_landscape(self, results: list, checkpoint_info: dict):
        """Append loss landscape results to one combined CSV and queue one PDF plot.
        
        Args:
            results: List of dictionaries containing loss landscape data
            checkpoint_info: Dictionary with phase and epoch information
        """
        if not results:
            return

        self._ensure_loss_landscape_paths()
        
        parameter_names = self._get_parameter_names()

        if not parameter_names:
            parameter_names = self._get_unique_parameter_names_from_results(results)

        desired_fields = (
            parameter_names
            + [
                'total_loss',
                'data_loss',
                'physics_loss',
                'parameter',
                'value',
                'is_current',
                'phase',
                'epoch',
            ]
        )

        for field in desired_fields:
            if field not in self._loss_landscape_fieldnames:
                self._loss_landscape_fieldnames.append(field)

        # Preserve any extra keys that may appear in individual result rows.
        for result in results:
            for field in result.keys():
                if field not in self._loss_landscape_fieldnames:
                    self._loss_landscape_fieldnames.append(field)

        self._loss_landscape_rows.extend(results)

        with open(self._loss_landscape_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=self._loss_landscape_fieldnames,
                extrasaction='ignore'
            )
            writer.writeheader()
            for result in self._loss_landscape_rows:
                row = {field: result.get(field, float('nan')) for field in self._loss_landscape_fieldnames}
                writer.writerow(row)

        logger.info(f"Combined loss landscape CSV updated at: {self._loss_landscape_csv_path}")

        self._loss_landscape_plot_tasks.append({
            'results': results,
            'checkpoint_info': checkpoint_info
        })

    def _draw_loss_landscape_axis(self, ax, results: list, checkpoint_info: dict):
        """Draw a full parallel-coordinate loss landscape on a given axis.

        Args:
            ax: Matplotlib axis to draw on
            results: List of dictionaries containing loss landscape data
            checkpoint_info: Dictionary with phase and epoch information
        """
        if not results:
            ax.axis('off')
            return

        def _finite(values):
            finite_values = []
            for value in values:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    finite_values.append(numeric_value)
            return finite_values

        parameter_names = self._get_parameter_names()

        if not parameter_names:
            parameter_names = self._get_unique_parameter_names_from_results(results)

        loss_keys = ['physics_loss', 'data_loss', 'total_loss']
        loss_labels = ['Physics Loss', 'Data Loss', 'Total Loss']
        loss_key_by_label = {label: key for label, key in zip(loss_labels, loss_keys)}

        dimensions = parameter_names + loss_labels
        n_dims = len(dimensions)
        n_param_axes = len(parameter_names)
        x_positions = np.arange(n_dims)

        def get_axis_value(axis_label: str, result: dict):
            if axis_label in parameter_names:
                if axis_label in result:
                    return result[axis_label]
                if result.get('parameter') == axis_label:
                    return result.get('value')
                return None

            return result.get(loss_key_by_label[axis_label])

        current_values = {}
        for axis_label in dimensions:
            for result in results:
                if not result.get('is_current', False):
                    continue

                val = get_axis_value(axis_label, result)
                if val is None:
                    continue

                try:
                    numeric_value = float(val)
                except (TypeError, ValueError):
                    continue

                if np.isfinite(numeric_value):
                    current_values[axis_label] = numeric_value
                    break

        axis_ranges = {}
        for axis_label in dimensions:
            vals = _finite([get_axis_value(axis_label, result) for result in results])
            if vals:
                lo, hi = min(vals), max(vals)
                if lo == hi:
                    lo -= 0.5
                    hi += 0.5
                axis_ranges[axis_label] = (lo, hi)
            else:
                axis_ranges[axis_label] = (0.0, 1.0)

        def value_to_y(axis_label, value):
            lo, hi = axis_ranges[axis_label]
            if hi == lo:
                return 0.5
            return (float(value) - lo) / (hi - lo)

        def line_points(result):
            ys = []
            for axis_label in dimensions:
                val = get_axis_value(axis_label, result)
                if val is None:
                    return None
                try:
                    numeric_value = float(val)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(numeric_value):
                    return None
                ys.append(value_to_y(axis_label, numeric_value))
            return ys

        # Draw vertical axis lines.
        for i in range(n_dims):
            ax.plot([i, i], [0, 1], color='black', linewidth=1.5, zorder=1)

        # Axis labels on top with a blank background so they stand out.
        for i, dim in enumerate(dimensions):
            ax.text(
                i,
                1.04,
                dim,
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='black',
                zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0),
            )

        # Add raw-value tick labels to each axis.
        if n_dims > 10:
            n_ticks = 2
        elif n_dims > 6:
            n_ticks = 3
        else:
            n_ticks = 5

        tick_fontsize = 11
        current_tick_fontsize = 13

        for i, axis_label in enumerate(dimensions):
            lo, hi = axis_ranges[axis_label]
            current_value = current_values.get(axis_label)
            tick_fracs = []

            for t in range(n_ticks):
                frac = t / (n_ticks - 1) if n_ticks > 1 else 0.5
                val = lo + frac * (hi - lo)
                tick_fracs.append(frac)
                ax.text(
                    i - 0.03,
                    frac,
                    f'{val:.4g}',
                    ha='right',
                    va='center',
                    fontsize=tick_fontsize,
                    color='gray'
                )

            if current_value is not None:
                current_y = value_to_y(axis_label, current_value)
                if 0.0 <= current_y <= 1.0 and not any(abs(current_y - frac) < 0.02 for frac in tick_fracs):
                    ax.text(
                        i - 0.03,
                        current_y,
                        f'{current_value:.4g}',
                        ha='right',
                        va='center',
                        fontsize=current_tick_fontsize,
                        fontweight='bold',
                        color='darkred'
                    )

        # Color mapping for swept parameters.
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_param_axes, 1)))
        param_color_map = {name: colors[idx] for idx, name in enumerate(parameter_names)}

        # Draw lines for each result.
        for result in results:
            ys = line_points(result)
            if ys is None:
                continue

            color = param_color_map.get(result.get('parameter'), 'gray')
            ax.plot(x_positions, ys, color=color, alpha=0.3, linewidth=1, zorder=2)

        # Highlight current parameter values.
        for result in results:
            if not result.get('is_current', False):
                continue

            ys = line_points(result)
            if ys is None:
                continue

            color = param_color_map.get(result.get('parameter'), 'gray')
            ax.plot(
                x_positions,
                ys,
                color=color,
                linewidth=3,
                marker='o',
                markersize=6,
                zorder=3,
            )

        ax.set_xlim(-0.5, n_dims - 0.5)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f'Loss Landscape - {checkpoint_info["phase"].upper()} Epoch {checkpoint_info["epoch"]}',
            fontsize=12
        )

    def generate_loss_landscape(self, t_tensor: torch.Tensor, collocation_points: torch.Tensor, 
                               checkpoint_info: dict):
        """Generate loss landscape around current parameter values.
        
        Args:
            t_tensor: Time points for the empirical data
            collocation_points: Collocation points for physics loss
            checkpoint_info: Dictionary with phase and epoch information
        """
        # Get parameters to analyze - use learnable parameters if available, otherwise use all model parameters
        learnable_param_dict = self.config.get("learnable_parameters", {})
        parameter_names = self._get_parameter_names()
        
        # If no learnable parameters, use all parameters from the physics model
        if not learnable_param_dict:
            logger.info("No learnable parameters found, using all model parameters for loss landscape")
            param_dict = {
                name: self._as_scalar_float(self.physics_model.parameters[name])
                for name in parameter_names
            }
        else:
            param_dict = {}
            for name, initial_value in learnable_param_dict.items():
                if name in self.physics_model.parameters and name not in self._RESERVED_COLUMNS:
                    param_dict[name] = self._as_scalar_float(initial_value)
            
            if not param_dict:
                param_dict = {
                    name: self._as_scalar_float(self.physics_model.parameters[name])
                    for name in parameter_names
                }
            
        if not param_dict:
            logger.info("No parameters to analyze for loss landscape")
            return
            
        # For each parameter, generate loss landscape
        all_results = []
        for param_name, current_value in param_dict.items():
            if param_name in self.physics_model.parameters:
                # Create parameter grid from -300% to +300% of current value
                if current_value == 0:
                    # Handle zero case
                    param_range = np.linspace(-1.0, 1.0, 25)
                else:
                    param_range = np.linspace(-3 * abs(current_value), 3 * abs(current_value), 25)
                
                # Compute loss landscape for this parameter
                results = self.compute_residual_loss_landscape(
                    t_tensor, collocation_points, param_name, param_range, current_value, checkpoint_info)
                all_results.extend(results)
        
        # Save results
        if all_results:
            self.save_loss_landscape(all_results, checkpoint_info)

    def save_model_for_inference(self):
        """Save the trained model for later inference.
        
        Saves both the model state dictionary and the full model object to enable
        easy loading for inference. The files are saved in the Hydra output directory.
        
        Returns:
            str: Path to the saved model file, or None if saving failed
        """
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
        """Train the PINN model using Adam and L-BFGS optimizers.
        
        The training process consists of two phases:
        
        1. Adam optimization: Performs a large number of iterations with adaptive
           learning rates to find a good approximate solution
        2. L-BFGS optimization: Fine-tunes the solution with a quasi-Newton method
           for higher accuracy
           
        Throughout training, metrics are logged to MLflow if enabled, including
        loss values and parameter estimates.
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
                        "learnable_parameters": str(self.config.get('learnable_parameters', {}))
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
            self._reset_loss_landscape_artifacts()

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
            lbfgs_max_iter = self.config.get('lbfgs_max_iter', 100)
            n_collocation_points = self.config.get('n_collocation_points', 100)
            log_interval = 50  # Fixed log interval for MLflow
            
            # Sample collocation points
            t_min, t_max = t_array.min(), t_array.max()
            collocation_points = self.sample_collocation_points((t_min, t_max), n_collocation_points)
            
            # Check collocation points
            if collocation_points.nelement() == 0:
                raise ValueError("No collocation points generated.")

            # Setup CSV logging
            hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            csv_file_path = os.path.join(hydra_output_dir, "training_metrics.csv")
            
            # Initialize CSV file with headers including phase column and all compartmental parameters
            parameter_names = self._get_parameter_names()
            fieldnames = ['phase', 'epoch', 'total_loss', 'data_loss', 'physics_loss'] + parameter_names
            fieldnames = list(dict.fromkeys(fieldnames))
            
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
            
            # Adam phase
            adam_checkpoint_interval = max(1, adam_epochs // 4)  # Every 25% of training
            for epoch in range(adam_epochs):
                optimizer_adam.zero_grad()
                total_loss, data_loss, physics_loss = self.compute_loss(t_tensor, self.data, collocation_points)
                total_loss.backward()
                optimizer_adam.step()

                if epoch % log_interval == 0:
                    current_parameter_values = self._get_parameter_values()

                    # Create log message with loss values and parameter values
                    log_message = f"[Adam] Epoch {epoch}/{adam_epochs} - Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}"
                    
                    # Add parameter values to log message
                    if len(current_parameter_values) > 0:
                        param_values = ", ".join([f"{name}: {value:.4f}" for name, value in current_parameter_values.items()])
                        log_message += f", Parameters: {param_values}"
                    
                    logger.info(log_message)
                    
                    # Log metrics with MLflow (use numeric phase identifier)
                    if self.mlflow_enabled:
                        try:
                            if mlflow.active_run():
                                metrics_dict = {
                                    "phase": 0.0,  # 0 for Adam phase
                                    "epoch": float(epoch),
                                    "total_loss": total_loss.item(),
                                    "data_loss": data_loss.item(),
                                    "physics_loss": physics_loss.item()
                                }
                                
                                # Add all compartmental parameter values to metrics dict
                                metrics_dict.update(current_parameter_values)
                                
                                mlflow.log_metrics(metrics_dict, step=epoch)
                                logger.debug(f"Logged metrics to MLflow at epoch {epoch}")
                        except Exception as e:
                            logger.warning(f"Failed to log metrics to MLflow at epoch {epoch}: {e}")
                    
                    # Log metrics to CSV (keep string phase for readability)
                    try:
                        row_data = {
                            'phase': 'adam',
                            'epoch': epoch,
                            'total_loss': total_loss.item(),
                            'data_loss': data_loss.item(),
                            'physics_loss': physics_loss.item()
                        }
                        
                        row_data.update(current_parameter_values)
                        
                        with open(csv_file_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                            writer.writerow(row_data)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to CSV at epoch {epoch}: {e}")
                
                # Generate loss landscape at checkpoints
                if epoch % adam_checkpoint_interval == 0 and epoch > 0:
                    logger.info(f"Generating loss landscape at Adam epoch {epoch}")
                    checkpoint_info = {'phase': 'adam', 'epoch': epoch}
                    self.generate_loss_landscape(t_tensor, collocation_points, checkpoint_info)

            # L-BFGS phase with iteration tracking
            lbfgs_iteration = [0]  # Use list to make it mutable in closure
            lbfgs_checkpoint_interval = max(1, lbfgs_max_iter // 4)  # Every 25% of training
            
            def closure():
                optimizer_lbfgs.zero_grad()
                total_loss, data_loss, physics_loss = self.compute_loss(t_tensor, self.data, collocation_points)
                total_loss.backward()
                
                current_parameter_values = self._get_parameter_values()

                # Log metrics for L-BFGS iterations
                current_iter = lbfgs_iteration[0]
                log_message = f"[L-BFGS] Iteration {current_iter}/{lbfgs_max_iter} - Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}"
                
                # Add parameter values to log message
                if len(current_parameter_values) > 0:
                    param_values = ", ".join([f"{name}: {value:.4f}" for name, value in current_parameter_values.items()])
                    log_message += f", Parameters: {param_values}"
                
                logger.info(log_message)
                
                # Log metrics with MLflow (use numeric phase identifier)
                if self.mlflow_enabled:
                    try:
                        if mlflow.active_run():
                            metrics_dict = {
                                "phase": 1.0,  # 1 for L-BFGS phase
                                "epoch": float(current_iter),
                                "total_loss": total_loss.item(),
                                "data_loss": data_loss.item(),
                                "physics_loss": physics_loss.item()
                            }
                            
                            # Add all compartmental parameter values to metrics dict
                            metrics_dict.update(current_parameter_values)
                            
                            mlflow.log_metrics(metrics_dict, step=adam_epochs + current_iter)
                            logger.debug(f"Logged metrics to MLflow at L-BFGS iteration {current_iter}")
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to MLflow at L-BFGS iteration {current_iter}: {e}")
                
                # Log metrics to CSV (keep string phase for readability)
                try:
                    row_data = {
                        'phase': 'lbfgs',
                        'epoch': current_iter,
                        'total_loss': total_loss.item(),
                        'data_loss': data_loss.item(),
                        'physics_loss': physics_loss.item()
                    }
                    
                    row_data.update(current_parameter_values)
                    
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                        writer.writerow(row_data)
                except Exception as e:
                    logger.warning(f"Failed to log metrics to CSV at L-BFGS iteration {current_iter}: {e}")
                
                # Generate loss landscape at checkpoints
                if current_iter % lbfgs_checkpoint_interval == 0 and current_iter > 0:
                    logger.info(f"Generating loss landscape at L-BFGS iteration {current_iter}")
                    checkpoint_info = {'phase': 'lbfgs', 'epoch': current_iter}
                    self.generate_loss_landscape(t_tensor, collocation_points, checkpoint_info)
                
                lbfgs_iteration[0] += 1
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
            logger.info(f"Training metrics saved to: {csv_file_path}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self._finalize_loss_landscape_artifacts()

            # End MLflow run if we started it
            if self.mlflow_enabled and should_end_run and mlflow.active_run():
                try:
                    mlflow.end_run()
                    logger.info("Ended MLflow run")
                except Exception as e:
                    logger.warning(f"Failed to end MLflow run: {e}")
