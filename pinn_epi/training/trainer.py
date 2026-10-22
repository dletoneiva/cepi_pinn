"""Training orchestrator for PINN models."""

import torch
import torch.nn as nn
import logging
import textwrap
from typing import Tuple, Dict, Any, Optional
from pinn_epi.models.networks import ModularPINN
from pinn_epi.models.physics import CompartmentalModel
import mlflow
import mlflow.pytorch
import hydra
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pinn_epi.utils.device_utils import DEVICE  # Import the centralized device
import importlib

# Set up logging
logger = logging.getLogger(__name__)


# Try to reuse project-level constants if available. If not, fall back to
# local defaults so the trainer still renders Greek labels and equations.
_constants_module = None
for _module_name in (
    "pinn_epi.constants",
    "pinn_epi.configs.constants",
    "pinn_epi.utils.constants",
):
    try:
        _constants_module = importlib.import_module(_module_name)
        break
    except Exception:
        continue

if _constants_module is not None:
    _CONSTANT_PARAMETER_GREEK_MAP = getattr(
        _constants_module,
        "PARAMETER_GREEK_MAP",
        getattr(_constants_module, "GREEK_PARAMETER_MAP", {}),
    )
    _CONSTANT_PHYSICS_EQUATION_LATEX = getattr(
        _constants_module,
        "PHYSICS_EQUATION_LATEX",
        getattr(
            _constants_module,
            "MODEL_EQUATIONS_LATEX",
            getattr(_constants_module, "PHYSICS_EQUATIONS_LATEX", {}),
        ),
    )
else:
    _CONSTANT_PARAMETER_GREEK_MAP = {}
    _CONSTANT_PHYSICS_EQUATION_LATEX = {}

if not isinstance(_CONSTANT_PARAMETER_GREEK_MAP, dict):
    _CONSTANT_PARAMETER_GREEK_MAP = {}

if not isinstance(_CONSTANT_PHYSICS_EQUATION_LATEX, dict):
    _CONSTANT_PHYSICS_EQUATION_LATEX = {}


_DEFAULT_PARAMETER_GREEK_MAP = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "zeta": r"\zeta",
    "eta": r"\eta",
    "theta": r"\theta",
    "iota": r"\iota",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "pi": r"\pi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "upsilon": r"\upsilon",
    "phi": r"\phi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
    "v": r"\upsilon",
    "vaccination": r"\upsilon",
    "vaccination_rate": r"\upsilon",
    "vaccine": r"\upsilon",
    "vaccine_rate": r"\upsilon",
    "vax": r"\upsilon",
    "vax_rate": r"\upsilon",
    "waning": r"\nu",
    "waning_rate": r"\nu",
    "immunity_loss": r"\nu",
    "immunity_loss_rate": r"\nu",
    "exposure": r"\sigma",
    "exposure_rate": r"\sigma",
    "incubation": r"\sigma",
    "incubation_rate": r"\sigma",
    "latent": r"\sigma",
    "latent_rate": r"\sigma",
    "recovery": r"\gamma",
    "recovery_rate": r"\gamma",
    "death": r"\delta",
    "death_rate": r"\delta",
    "mortality": r"\delta",
    "mortality_rate": r"\delta",
    "fatality": r"\delta",
    "fatality_rate": r"\delta",
    "birth": r"\mu",
    "birth_rate": r"\mu",
    "natural_death": r"\mu",
    "natural_death_rate": r"\mu",
    "population_death": r"\mu",
    "population_death_rate": r"\mu",
    "transmission": r"\beta",
    "transmission_rate": r"\beta",
    "contact": r"\beta",
    "contact_rate": r"\beta",
    "infection": r"\beta",
    "infection_rate": r"\beta",
    "vaccine_efficacy": r"\epsilon",
    "efficacy": r"\epsilon",
}

# Matplotlib's mathtext parser does NOT support LaTeX environments such as
# \begin{array}{l} ... \end{array}. Each equation is therefore stored as a list
# of single-line mathtext expressions (no surrounding $ delimiters). They are
# rendered as a compact stacked block using \substack and wrapped in a box.
# The full set of equations is always shown (no truncation).
_DEFAULT_PHYSICS_EQUATION_LATEX = {
    "SI": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dI}{dt} = \beta S I",
    ],
    "SIR": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dI}{dt} = \beta S I - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
    ],
    "SIS": [
        r"\frac{dS}{dt} = -\beta S I + \gamma I",
        r"\frac{dI}{dt} = \beta S I - \gamma I",
    ],
    "SEIR": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
    ],
    "SIRD": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dI}{dt} = \beta S I - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
    ],
    "SIRV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dI}{dt} = \beta S I - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
    "SIRDV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dI}{dt} = \beta S I - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
    "SEIRD": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
    ],
    "SEIRV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
    "SEIRDV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
}

PARAMETER_GREEK_MAP = {
    **_DEFAULT_PARAMETER_GREEK_MAP,
    **_CONSTANT_PARAMETER_GREEK_MAP,
}

PHYSICS_EQUATION_LATEX = {
    **_DEFAULT_PHYSICS_EQUATION_LATEX,
    **_CONSTANT_PHYSICS_EQUATION_LATEX,
}


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

    _LOSS_LABELS = ("Physics Loss", "Data Loss", "Total Loss")

    _PARAMETER_GREEK_MAP = PARAMETER_GREEK_MAP
    _PHYSICS_EQUATION_LATEX = PHYSICS_EQUATION_LATEX

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

    @staticmethod
    def _ensure_math_delimiters(value: Any) -> str:
        """Ensure a single mathtext token is wrapped in $...$ delimiters."""
        if value is None:
            return ""

        text = str(value).strip()
        if not text:
            return ""

        if text.startswith("$") and text.endswith("$"):
            return text

        if text.startswith("$"):
            return text + "$"

        if text.endswith("$"):
            return "$" + text

        return f"${text}$"

    @staticmethod
    def _axis_range_from_values(values: list) -> Tuple[float, float]:
        """Compute a (lo, hi) axis range from a list of possibly-invalid values.

        Ignores non-numeric and non-finite entries. When all valid values are
        equal (or none exist) the range is widened so the axis stays visible.
        """
        finite_values = []
        for value in values:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                finite_values.append(numeric_value)

        if finite_values:
            lo, hi = min(finite_values), max(finite_values)
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            return (lo, hi)

        return (0.0, 1.0)

    def _normalize_equation_lines(self, value: Any) -> list:
        """Convert an equation value into a list of mathtext lines.

        Accepts either a list/tuple of lines or a single string. Handles the
        unsupported ``\\begin{array}{l} ... \\end{array}`` environment by
        stripping the environment markers and splitting on ``\\\\`` row
        separators, so the result is always a list of single-line mathtext
        expressions (without ``$`` delimiters).
        """
        if value is None:
            return []

        if isinstance(value, (list, tuple)):
            text = r" \\ ".join(str(item) for item in value)
        else:
            text = str(value)

        # Matplotlib mathtext does not support array environments; remove them
        # so the content can be split into independent lines instead.
        text = re.sub(r"\\begin\{array\}(?:\s*\{[^}]*\})?", " ", text)
        text = re.sub(r"\\end\{array\}", " ", text)

        text = text.strip()
        if text.startswith("$") and text.endswith("$"):
            text = text[1:-1]

        chunks = text.split(r"\\")

        lines = []
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk.startswith("$"):
                chunk = chunk[1:]
            if chunk.endswith("$"):
                chunk = chunk[:-1]
            chunk = chunk.strip()
            if chunk:
                lines.append(chunk)

        return lines

    def _get_parameter_display_label(self, name: str) -> str:
        """Return a display label for a parameter, using Greek letters when known."""
        key = str(name).lower()

        symbol = self._PARAMETER_GREEK_MAP.get(key)
        if symbol is None:
            for token in key.split("_"):
                if token in self._PARAMETER_GREEK_MAP:
                    symbol = self._PARAMETER_GREEK_MAP[token]
                    break

        if symbol is None:
            return str(name)

        text = str(symbol).strip()
        if not text:
            return str(name)

        if text.startswith("$") or "\\" in text:
            return self._ensure_math_delimiters(text)

        return text

    def _get_axis_display_label(self, axis_label: str) -> str:
        """Return display text for a parallel-coordinate axis label."""
        if axis_label in self._LOSS_LABELS:
            return axis_label

        return self._get_parameter_display_label(axis_label)

    def _get_physics_equation_lines(self) -> list:
        """Return the physics model's ODE equations as a list of mathtext lines.

        The lookup mirrors the previous behaviour (model class name, config
        ``model_type``, model-provided attributes, then a prefix fallback). The
        equations are returned in full; no truncation is applied so that every
        compartment equation is rendered inside the boxed block.
        """
        model_name = type(self.physics_model).__name__
        normalized_model_name = model_name.replace("Model", "").upper()

        model_cfg = self.config.get("model", {}) or {}
        if hasattr(model_cfg, "get"):
            model_type_from_cfg = model_cfg.get("type", "")
        else:
            model_type_from_cfg = ""

        config_model_type_raw = str(self.config.get("model_type", "") or model_type_from_cfg or "")
        config_model_type = config_model_type_raw.upper()

        candidate_keys = [
            model_name,
            normalized_model_name,
            config_model_type,
            config_model_type_raw,
        ]

        raw = None
        for key in candidate_keys:
            if not key:
                continue

            if key in self._PHYSICS_EQUATION_LATEX:
                raw = self._PHYSICS_EQUATION_LATEX[key]
                break

            lower_key = str(key).lower()
            if lower_key in self._PHYSICS_EQUATION_LATEX:
                raw = self._PHYSICS_EQUATION_LATEX[lower_key]
                break

        if raw is None:
            getter = getattr(self.physics_model, "get_equation_latex", None)
            if callable(getter):
                try:
                    raw = getter()
                except Exception:
                    raw = None

        if raw is None:
            for attr in ("equation_latex", "physics_equation_latex"):
                value = getattr(self.physics_model, attr, None)
                if value is not None:
                    raw = value
                    break

        if raw is None:
            for key in sorted(self._PHYSICS_EQUATION_LATEX.keys(), key=lambda item: len(str(item)), reverse=True):
                key_str = str(key)
                if key_str and key_str.upper() in normalized_model_name:
                    raw = self._PHYSICS_EQUATION_LATEX[key]
                    break

        return self._normalize_equation_lines(raw)

    @staticmethod
    def _format_equation_block(lines: list) -> str:
        """Format equation lines into a single mathtext block for display.

        A single line is returned as ``$...$``; multiple lines are wrapped in a
        ``\\substack`` so they render as a compact stacked block (suitable for
        drawing inside a box). No truncation is performed.
        """
        if not lines:
            return ""

        if len(lines) == 1:
            return f"${lines[0]}$"

        return r"$\substack{" + r" \\ ".join(lines) + r"}$"

    @staticmethod
    def _fmt_scalar(value: Any) -> str:
        """Format a numeric config value for the summary, keeping a decimal point.

        ``1.0`` renders as ``1.0`` (not ``1``), while ``0.001`` renders as
        ``0.001``. Non-numeric values fall back to their string form.
        """
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)

        text = f"{f:g}"
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """Convert a CamelCase class name to snake_case for the summary."""
        step1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        step2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', step1)
        return step2.lower()

    def _backbone_label(self, backbone: Any) -> str:
        """Return a short architecture label for the backbone component."""
        if backbone is None:
            return "None"

        name = type(backbone).__name__
        if "MLP" in name.upper():
            return "mlp"

        cleaned = name
        for suffix in ("Network", "Backbone"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break

        return self._camel_to_snake(cleaned) if cleaned else "mlp"

    def _encoder_label(self, encoder: Any) -> str:
        """Return a short architecture label for the encoder component."""
        if encoder is None:
            return "None"

        name = type(encoder).__name__
        if name.endswith("Encoder"):
            name = name[: -len("Encoder")]

        return self._camel_to_snake(name) if name else "encoder"

    def _head_label(self, head: Any) -> str:
        """Return a short architecture label for the head component."""
        if head is None:
            return "None"

        name = type(head).__name__
        if name.endswith("Head"):
            name = name[: -len("Head")]

        if name:
            return name[0].lower() + name[1:]

        return "head"

    def _build_experiment_summary(self) -> str:
        """Build a single-line experiment summary for the top of each PDF page.

        Mirrors the style of the network-prediction summary, packing the model
        equations name, loss weights, optimizer settings, collocation count,
        learned parameters, observed variables and architecture into one line.
        """
        model_name = type(self.physics_model).__name__.replace("Model", "")

        data_weight = self._fmt_scalar(self.config.get('data_weight', 1.0))
        physics_weight = self._fmt_scalar(self.config.get('physics_weight', 1.0))

        adam_epochs = self.config.get('adam_epochs', 5000)
        adam_lr = self._fmt_scalar(self.config.get('adam_lr', 1e-3))
        lbfgs_max_iter = self.config.get('lbfgs_max_iter', 100)
        n_collocation_points = self.config.get('n_collocation_points', 100)

        learnable = self.config.get('learnable_parameters', {}) or {}
        if learnable:
            learned = "[" + ", ".join(str(key) for key in learnable.keys()) + "]"
        else:
            learned = "None"

        observables_config = self.config.get('observables', {}) or {}
        observed_vars = observables_config.get('observed_variables', []) or []
        if observed_vars:
            observed = "[" + ", ".join(str(var) for var in observed_vars) + "]"
        else:
            observed = "None"

        backbone = self._backbone_label(getattr(self.model, "backbone", None))
        encoder = self._encoder_label(getattr(self.model, "encoder", None))
        head = self._head_label(getattr(self.model, "head", None))

        return (
            f"Equations: {model_name} | "
            f"Loss: Data={data_weight}, Physics={physics_weight} | "
            f"Adam: {adam_epochs}ep, lr={adam_lr} | "
            f"LBFGS: {lbfgs_max_iter}iter | "
            f"Collocation: {n_collocation_points}pts | "
            f"Learned: {learned} | "
            f"Observed: {observed} | "
            f"Architecture: Backbone: {backbone}, Encoder: {encoder}, Head: {head}"
        )

    def _build_experiment_header_text(self) -> str:
        """Build the top-of-page description for loss-landscape PDF pages.

        This is intentionally closer to the network-prediction summary in
        ``plotting.plot_actual_vs_predicted``: a readable wrapped block with a
        boxed background, plus the physics equations for the current model.
        """
        summary_text = self._build_experiment_summary()
        wrapped_summary = textwrap.fill(summary_text, width=110)

        equation_lines = self._get_physics_equation_lines()
        if not equation_lines:
            return wrapped_summary

        equation_text = "\n".join(
            self._ensure_math_delimiters(line) for line in equation_lines
        )
        return f"{wrapped_summary}\n\nPhysics equations:\n{equation_text}"

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

    def _get_axis_value(self, axis_label: str, result: dict, parameter_names: list, loss_key_by_label: dict):
        """Extract the numeric value for a given axis label from a result row."""
        if axis_label in parameter_names:
            if axis_label in result:
                return result[axis_label]
            if result.get('parameter') == axis_label:
                return result.get('value')
            return None

        return result.get(loss_key_by_label[axis_label])

    def _compute_global_axis_ranges(self, rows: list) -> Dict[str, Tuple[float, float]]:
        """Compute one (lo, hi) range per dimension across ALL loss-landscape rows.

        Using a single global range per dimension (parameter axes and loss axes)
        guarantees that every subplot on every page shares the same scale. For
        example, the "Data Loss" axis spans the same min/max in every snapshot,
        making the plots directly comparable.
        """
        parameter_names = self._get_parameter_names()
        if not parameter_names:
            parameter_names = self._get_unique_parameter_names_from_results(rows)

        loss_labels = list(self._LOSS_LABELS)
        loss_keys = ['physics_loss', 'data_loss', 'total_loss']
        loss_key_by_label = {label: key for label, key in zip(loss_labels, loss_keys)}

        dimensions = parameter_names + loss_labels

        ranges: Dict[str, Tuple[float, float]] = {}
        for axis_label in dimensions:
            values = [
                self._get_axis_value(axis_label, result, parameter_names, loss_key_by_label)
                for result in rows
            ]
            ranges[axis_label] = self._axis_range_from_values(values)

        return ranges

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

        Plots are organized into two columns per PDF page. Each page carries a
        single general experiment summary at the very top (instead of repeating
        the physics equations under every subplot). All subplots share a single
        global scale per dimension so that, e.g., the Data Loss axis uses the
        same min/max across every snapshot.
        """
        if not self._loss_landscape_plot_tasks:
            return

        self._ensure_loss_landscape_paths()

        if self._loss_landscape_pdf_path is None:
            return

        try:
            global_axis_ranges = self._compute_global_axis_ranges(self._loss_landscape_rows)
            header_text = self._build_experiment_header_text()
            header_line_count = max(1, len(header_text.splitlines()))
            rect_top = max(0.58, 0.94 - 0.045 * header_line_count)

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
                                task['checkpoint_info'],
                                global_axis_ranges=global_axis_ranges
                            )
                        else:
                            ax.axis('off')

                    fig.text(
                        0.5,
                        0.985,
                        header_text,
                        ha='center',
                        va='top',
                        fontsize=11,
                        linespacing=1.35,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        transform=fig.transFigure,
                    )
                    fig.tight_layout(rect=[0, 0, 1, rect_top])
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

    def _draw_loss_landscape_axis(self, ax, results: list, checkpoint_info: dict, global_axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """Draw a full parallel-coordinate loss landscape on a given axis.

        The physics equations are no longer rendered per subplot; they have been
        replaced by a single page-level experiment summary added in
        ``_finalize_loss_landscape_artifacts``.

        Args:
            ax: Matplotlib axis to draw on
            results: List of dictionaries containing loss landscape data
            checkpoint_info: Dictionary with phase and epoch information
            global_axis_ranges: Optional shared (lo, hi) range per dimension. When
                provided, every subplot uses the same scale for each axis so the
                charts are directly comparable.
        """
        if not results:
            ax.axis('off')
            return

        parameter_names = self._get_parameter_names()

        if not parameter_names:
            parameter_names = self._get_unique_parameter_names_from_results(results)

        loss_labels = list(self._LOSS_LABELS)
        loss_keys = ['physics_loss', 'data_loss', 'total_loss']
        loss_key_by_label = {label: key for label, key in zip(loss_labels, loss_keys)}

        dimensions = parameter_names + loss_labels
        n_dims = len(dimensions)
        n_param_axes = len(parameter_names)
        x_positions = np.arange(n_dims)

        def get_axis_value(axis_label: str, result: dict):
            return self._get_axis_value(axis_label, result, parameter_names, loss_key_by_label)

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
            axis_ranges[axis_label] = self._axis_range_from_values(
                [get_axis_value(axis_label, result) for result in results]
            )

        # Override local ranges with the shared global ranges so every subplot on
        # every page uses the same scale for each dimension (e.g. Data Loss).
        if global_axis_ranges:
            for axis_label in dimensions:
                if axis_label in global_axis_ranges:
                    axis_ranges[axis_label] = global_axis_ranges[axis_label]

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
            display_label = self._get_axis_display_label(dim)
            ax.text(
                i,
                1.04,
                display_label,
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='black',
                zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0),
            )

        # Add raw-value tick labels to each axis. These are drawn on top of the
        # sweep lines with a white background so they remain legible regardless
        # of how many lines cross behind them.
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
                    color='gray',
                    zorder=4,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.0),
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
                        color='darkred',
                        zorder=5,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1.0),
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

        title = f'Loss Landscape - {checkpoint_info["phase"].upper()} Epoch {checkpoint_info["epoch"]}'
        ax.set_title(title, fontsize=12)

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
