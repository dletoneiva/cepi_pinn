"""Data collection (sweeps, CSV) for loss-landscape analysis."""

import csv
import logging
import os
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinn_epi.constants import RESERVED_COLUMNS
from pinn_epi.analysis import loss_landscape_plotting as llp

logger = logging.getLogger(__name__)


class LossLandscapeAnalyzer:
    """Collects loss-landscape sweep data and queues plots for the PDF artifact.

    The analyzer owns the mutable per-run state of the loss-landscape
    artifact: the accumulated CSV rows, the CSV fieldnames, and the queued
    plot tasks. Rendering is delegated to
    :mod:`pinn_epi.analysis.loss_landscape_plotting`. The sweep CSV and the
    multi-page PDF are routed to explicit directories so they land alongside
    the other CSVs and figures of the run.
    """

    def __init__(
        self,
        physics_model,
        loss_fn: Callable,                       # (t_data, y_data_dict, collocation) -> (total, data, physics)
        device,
        network_params_provider: Callable = lambda: [],
        learnable_params: Optional[nn.ParameterDict] = None,
        output_dir: Optional[str] = None,
        csv_dir: Optional[str] = None,
        pdf_dir: Optional[str] = None,
    ):
        self.physics_model = physics_model
        self.loss_fn = loss_fn
        self.device = device
        self.network_params_provider = network_params_provider
        self.learnable_params = learnable_params
        self.output_dir = output_dir

        # Artifact routing: the sweep CSV is written to ``csv_dir`` and the
        # multi-page PDF to ``pdf_dir``. When not provided, both fall back to
        # the previous ``<output_dir>/loss_landscape`` location.
        legacy_landscape_dir = (
            os.path.join(output_dir, "loss_landscape") if output_dir else None
        )
        self.csv_dir = csv_dir if csv_dir is not None else legacy_landscape_dir
        self.pdf_dir = pdf_dir if pdf_dir is not None else legacy_landscape_dir

        # Resolve a stable run identifier once: prefer the Hydra output
        # directory name (unique per run); fall back to a timestamp when
        # running outside Hydra (e.g. unit tests).
        if output_dir:
            self.run_id = os.path.basename(os.path.normpath(output_dir))
        else:
            self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.rows: list = []
        self.fieldnames: list = []
        self.plot_tasks: list = []
        self.csv_path: Optional[str] = None
        self.pdf_path: Optional[str] = None
        self.last_gradient_cosine: Optional[float] = None
        self.last_checkpoint_info: Optional[dict] = None

    def get_parameter_names(self) -> list:
        """Return ordered compartmental parameter names suitable for CSV/plot axes.

        For SIR this should return names such as ['beta', 'gamma'], preserving
        the order in which they are stored in the physics model.
        """
        names = []
        for name in self.physics_model.parameters.keys():
            if name in RESERVED_COLUMNS or name in names:
                continue
            names.append(name)
        return names

    def get_parameter_values(self) -> Dict[str, float]:
        """Return the current full compartmental parameter set as scalar floats."""
        values: Dict[str, float] = {}
        for name, value in self.physics_model.parameters.items():
            if name in RESERVED_COLUMNS:
                continue
            values[name] = llp.as_scalar_float(value)
        return values

    def compute_residual_loss_landscape(
        self,
        t_data: torch.Tensor,
        data: dict,
        collocation_points: torch.Tensor,
        param_name: str,
        param_values: np.ndarray,
        current_param_value: float,
        checkpoint_info: dict,
    ):
        """Compute loss landscape for a specific parameter around its current value.

        Args:
            t_data: Time points for the empirical data
            data: Ground truth data dictionary passed through to the loss function
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
        all_param_names = self.get_parameter_names()

        # Results storage
        results = []

        try:
            # Evaluate loss at each parameter value
            for val in param_values:
                # Update parameter value
                if isinstance(param_tensor, nn.Parameter):
                    param_tensor.data.fill_(val)
                else:
                    self.physics_model.parameters[param_name] = torch.tensor(val, dtype=torch.float32, device=self.device)

                # Compute loss
                try:
                    total_loss, data_loss, physics_loss = self.loss_fn(t_data, data, collocation_points)
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
                    row.update(self.get_parameter_values())
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
                    row.update(self.get_parameter_values())
                    results.append(row)
        finally:
            # Restore original parameter value
            if isinstance(param_tensor, nn.Parameter):
                if isinstance(original_value, torch.Tensor):
                    param_tensor.data.copy_(original_value.data)
                else:
                    param_tensor.data.fill_(original_value)
            else:
                self.physics_model.parameters[param_name] = original_value

        return results

    @staticmethod
    def _deduplicate_tensors(tensors: list) -> list:
        """Return unique tensors by object identity."""
        seen = set()
        unique = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor_id = id(tensor)
            if tensor_id not in seen:
                seen.add(tensor_id)
                unique.append(tensor)
        return unique

    @staticmethod
    def _flatten_autograd_result(grads, tensors: list) -> torch.Tensor:
        """Flatten autograd results into a single 1D float tensor."""
        flat = []
        for grad, tensor in zip(grads, tensors):
            if grad is None:
                flat.append(torch.zeros(tensor.numel(), dtype=torch.float32, device=tensor.device))
            else:
                flat.append(grad.detach().float().reshape(-1))

        if not flat:
            return torch.empty(0, dtype=torch.float32)

        return torch.cat(flat)

    def _cosine_from_grads(self, grads_data, grads_physics, tensors: list) -> Optional[float]:
        """Compute cosine similarity between two flattened gradient vectors."""
        if not tensors:
            return None

        try:
            flat_data = self._flatten_autograd_result(grads_data, tensors)
            flat_physics = self._flatten_autograd_result(grads_physics, tensors)

            if flat_data.numel() == 0 or flat_physics.numel() == 0:
                return None

            norm_data = flat_data.norm()
            norm_physics = flat_physics.norm()

            if norm_data == 0 or norm_physics == 0:
                return None

            cosine = (torch.dot(flat_data, flat_physics) / (norm_data * norm_physics)).item()

            if not np.isfinite(cosine):
                return None

            return float(np.clip(cosine, -1.0, 1.0))
        except Exception:
            return None

    def _prepare_compartmental_tensors_for_gradient(self):
        """Return tensors for compartmental parameters and a restore mapping.

        Fixed parameters are temporarily replaced by leaf tensors that require
        gradients so autograd can differentiate the loss with respect to them.
        """
        original = {}
        tensors = []

        for name, value in list(self.physics_model.parameters.items()):
            if name in RESERVED_COLUMNS:
                continue

            if isinstance(value, nn.Parameter):
                if value.requires_grad:
                    tensors.append(value)
                else:
                    leaf = value.detach().clone().to(dtype=torch.float32, device=self.device).requires_grad_(True)
                    original[name] = value
                    self.physics_model.parameters[name] = leaf
                    tensors.append(leaf)
            elif isinstance(value, torch.Tensor):
                if value.requires_grad:
                    tensors.append(value)
                else:
                    leaf = value.detach().clone().to(dtype=torch.float32, device=self.device).requires_grad_(True)
                    original[name] = value
                    self.physics_model.parameters[name] = leaf
                    tensors.append(leaf)
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue

                leaf = torch.tensor(numeric, dtype=torch.float32, device=self.device).requires_grad_(True)
                original[name] = value
                self.physics_model.parameters[name] = leaf
                tensors.append(leaf)

        return original, tensors

    def _restore_compartmental_parameters(self, original: dict) -> None:
        """Restore compartmental parameters after temporary gradient setup."""
        for name, value in original.items():
            self.physics_model.parameters[name] = value

    def compute_loss_gradient_cosine(
        self,
        t_data: torch.Tensor,
        data: dict,
        collocation_points: torch.Tensor,
    ) -> Optional[float]:
        """Compute cosine similarity between data-loss and physics-loss gradients.

        The preferred scope is the compartmental parameter set. If that gradient
        pair is degenerate (for example, if one gradient norm is zero), the
        computation falls back to all trainable parameters so the conflict bar
        remains informative.
        """
        original_params, compartmental_tensors = self._prepare_compartmental_tensors_for_gradient()

        try:
            network_tensors = [p for p in self.network_params_provider() if p.requires_grad]
            learnable_tensors = [
                p for p in (self.learnable_params.parameters() if self.learnable_params is not None else [])
                if p.requires_grad
            ]

            compartmental_tensors = self._deduplicate_tensors(compartmental_tensors)
            all_tensors = self._deduplicate_tensors(network_tensors + learnable_tensors + compartmental_tensors)

            if not all_tensors:
                return None

            _, data_loss, physics_loss = self.loss_fn(t_data, data, collocation_points)

            grads_data_all = torch.autograd.grad(
                data_loss,
                all_tensors,
                retain_graph=True,
                allow_unused=True
            )
            grads_physics_all = torch.autograd.grad(
                physics_loss,
                all_tensors,
                retain_graph=True,
                allow_unused=True
            )

            grad_data_map = {id(tensor): grad for tensor, grad in zip(all_tensors, grads_data_all)}
            grad_physics_map = {id(tensor): grad for tensor, grad in zip(all_tensors, grads_physics_all)}

            compartmental_grads_data = [grad_data_map.get(id(tensor)) for tensor in compartmental_tensors]
            compartmental_grads_physics = [grad_physics_map.get(id(tensor)) for tensor in compartmental_tensors]

            cosine_compartmental = self._cosine_from_grads(
                compartmental_grads_data,
                compartmental_grads_physics,
                compartmental_tensors
            )

            if cosine_compartmental is not None:
                return cosine_compartmental

            return self._cosine_from_grads(
                grads_data_all,
                grads_physics_all,
                all_tensors
            )
        except Exception:
            return None
        finally:
            self._restore_compartmental_parameters(original_params)

    def save_results(self, results: list, checkpoint_info: dict) -> None:
        """Append loss landscape results to one combined CSV and queue one PDF plot.

        Every row is stamped with run-level provenance: a stable ``run_id``
        (resolved once from the Hydra output directory name, or a timestamp
        outside Hydra) and the continuous ``global_step`` training step, so
        the CSV stays self-contained for auditing alongside
        ``training_metrics.csv``.

        Args:
            results: List of dictionaries containing loss landscape data
            checkpoint_info: Dictionary with phase and epoch information
        """
        if not results:
            return

        # Stamp provenance onto every row before any early return so queued
        # plot tasks carry the same identifiers as the CSV rows.
        global_step = checkpoint_info.get('global_step', float('nan'))
        for result in results:
            result['run_id'] = self.run_id
            result['global_step'] = global_step

        if self.csv_path is None:
            if self.csv_dir is None:
                # No output directory available (e.g. outside Hydra): skip CSV
                # writing but still queue the plot task so nothing is lost.
                self.plot_tasks.append({
                    'results': results,
                    'checkpoint_info': checkpoint_info
                })
                return
            os.makedirs(self.csv_dir, exist_ok=True)
            self.csv_path = os.path.join(self.csv_dir, "loss_landscape.csv")
            if self.pdf_dir is not None:
                os.makedirs(self.pdf_dir, exist_ok=True)
                self.pdf_path = os.path.join(self.pdf_dir, "loss_landscape.pdf")

        parameter_names = self.get_parameter_names()

        if not parameter_names:
            parameter_names = llp.get_unique_parameter_names_from_results(results)

        desired_fields = (
            ['run_id', 'global_step']
            + parameter_names
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
            if field not in self.fieldnames:
                self.fieldnames.append(field)

        # Preserve any extra keys that may appear in individual result rows.
        for result in results:
            for field in result.keys():
                if field not in self.fieldnames:
                    self.fieldnames.append(field)

        self.rows.extend(results)

        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=self.fieldnames,
                extrasaction='ignore'
            )
            writer.writeheader()
            for result in self.rows:
                row = {field: result.get(field, float('nan')) for field in self.fieldnames}
                writer.writerow(row)

        logger.info(f"Combined loss landscape CSV updated at: {self.csv_path}")

        self.plot_tasks.append({
            'results': results,
            'checkpoint_info': checkpoint_info
        })

    def generate(
        self,
        t_data: torch.Tensor,
        data: dict,
        collocation_points: torch.Tensor,
        checkpoint_info: dict,
        learnable_param_dict: dict,
        config: dict,
    ):
        """Generate loss landscape around current parameter values.

        Args:
            t_data: Time points for the empirical data
            data: Ground truth data dictionary passed through to the loss function
            collocation_points: Collocation points for physics loss
            checkpoint_info: Dictionary with phase and epoch information
            learnable_param_dict: Learnable parameters from the training config
            config: Full training configuration dictionary
        """
        # Get parameters to analyze - use learnable parameters if available, otherwise use all model parameters
        parameter_names = self.get_parameter_names()

        # If no learnable parameters, use all parameters from the physics model
        if not learnable_param_dict:
            logger.info("No learnable parameters found, using all model parameters for loss landscape")
            param_dict = {
                name: llp.as_scalar_float(self.physics_model.parameters[name])
                for name in parameter_names
            }
        else:
            param_dict = {}
            for name, initial_value in learnable_param_dict.items():
                if name in self.physics_model.parameters and name not in RESERVED_COLUMNS:
                    param_dict[name] = llp.as_scalar_float(initial_value)

            if not param_dict:
                param_dict = {
                    name: llp.as_scalar_float(self.physics_model.parameters[name])
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
                    t_data, data, collocation_points, param_name, param_range, current_value, checkpoint_info)
                all_results.extend(results)

        # Save results
        if all_results:
            gradient_cosine = self.compute_loss_gradient_cosine(t_data, data, collocation_points)
            landscape_info = dict(checkpoint_info)

            if 'global_step' not in landscape_info:
                phase = str(landscape_info.get('phase', '')).lower().replace('-', '').replace('_', '')
                epoch = llp.as_scalar_float(landscape_info.get('epoch'))
                adam_epochs = llp.as_scalar_float(config.get('adam_epochs', 5000))
                if not np.isfinite(adam_epochs):
                    adam_epochs = 5000.0
                if np.isfinite(epoch):
                    landscape_info['global_step'] = epoch if phase != 'lbfgs' else adam_epochs + epoch

            landscape_info['gradient_cosine'] = gradient_cosine
            self.last_gradient_cosine = gradient_cosine
            self.last_checkpoint_info = landscape_info
            self.save_results(all_results, landscape_info)

    def reset(self) -> None:
        """Reset accumulated loss-landscape state for a new training run.

        Output paths (``output_dir``/``csv_path``/``pdf_path``) are kept so a
        repeated run in the same process reuses the same artifact locations,
        matching the previous path-caching behaviour.
        """
        self.rows = []
        self.fieldnames = []
        self.plot_tasks = []
        self.last_gradient_cosine = None
        self.last_checkpoint_info = None

    def finalize(self, config, physics_model, model) -> None:
        """Render the combined loss-landscape PDF from all queued plot tasks."""
        if not self.plot_tasks:
            return

        parameter_names = self.get_parameter_names() or llp.get_unique_parameter_names_from_results(self.rows)
        header_text = llp.build_experiment_header_text(config, physics_model, model)
        llp.finalize_loss_landscape_pdf(
            self.plot_tasks,
            self.rows,
            parameter_names,
            config,
            header_text,
            self.pdf_path,
        )
        self.plot_tasks = []
