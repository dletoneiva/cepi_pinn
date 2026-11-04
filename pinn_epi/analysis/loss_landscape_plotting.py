"""Pure plotting/formatting helpers for loss-landscape artifacts."""

import logging
import re
import textwrap
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pinn_epi.constants import (
    PARAMETER_GREEK_MAP,
    PHYSICS_EQUATION_LATEX,
    PARAMETER_GRID_COLORS,
    HIGH_CONSPICUITY_PALETTE,
    TRAINER_PLOT_COLORS,
    RESERVED_COLUMNS,
    LOSS_LABELS,
)

logger = logging.getLogger(__name__)


def as_scalar_float(value: Any) -> float:
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


def ensure_math_delimiters(value: Any) -> str:
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


def axis_range_from_values(values: list, log_scale: bool = False) -> Tuple[float, float]:
    """Compute a (lo, hi) axis range from a list of possibly-invalid values.

    Ignores non-numeric and non-finite entries. When all valid values are
    equal (or none exist) the range is widened so the axis stays visible.

    If ``log_scale`` is True, the returned range is in original value space,
    but based only on positive finite values. Non-positive values are handled
    by the caller by clamping them to the bottom of the log axis.
    """
    finite_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            finite_values.append(numeric_value)

    if not finite_values:
        return (0.0, 1.0)

    if log_scale:
        positive_values = [v for v in finite_values if v > 0]
        if not positive_values:
            return (0.0, 1.0)

        lo = min(positive_values)
        hi = max(positive_values)

        if any(v <= 0 for v in finite_values):
            lo = lo / 10.0
            if lo <= 0:
                lo = min(positive_values)

        if lo == hi:
            if hi > 0:
                lo = hi / 10.0
                hi = hi * 10.0
                if lo <= 0:
                    lo = np.finfo(float).tiny
                    if hi <= lo:
                        hi = lo * 10.0
            else:
                return (0.0, 1.0)

        return (lo, hi)

    lo, hi = min(finite_values), max(finite_values)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return (lo, hi)


def normalize_equation_lines(value: Any) -> list:
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


def format_equation_block(lines: list) -> str:
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


def fmt_scalar(value: Any) -> str:
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


def camel_to_snake(name: str) -> str:
    """Convert a CamelCase class name to snake_case for the summary."""
    step1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    step2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', step1)
    return step2.lower()


def backbone_label(backbone: Any) -> str:
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

    return camel_to_snake(cleaned) if cleaned else "mlp"


def encoder_label(encoder: Any) -> str:
    """Return a short architecture label for the encoder component."""
    if encoder is None:
        return "None"

    name = type(encoder).__name__
    if name.endswith("Encoder"):
        name = name[: -len("Encoder")]

    return camel_to_snake(name) if name else "encoder"


def head_label(head: Any) -> str:
    """Return a short architecture label for the head component."""
    if head is None:
        return "None"

    name = type(head).__name__
    if name.endswith("Head"):
        name = name[: -len("Head")]

    if name:
        return name[0].lower() + name[1:]

    return "head"


def get_parameter_display_label(name: str, greek_map=PARAMETER_GREEK_MAP) -> str:
    """Return a display label for a parameter, using Greek letters when known."""
    key = str(name).lower()

    symbol = greek_map.get(key)
    if symbol is None:
        for token in key.split("_"):
            if token in greek_map:
                symbol = greek_map[token]
                break

    if symbol is None:
        return str(name)

    text = str(symbol).strip()
    if not text:
        return str(name)

    if text.startswith("$") or "\\" in text:
        return ensure_math_delimiters(text)

    return text


def get_axis_display_label(axis_label: str, greek_map=PARAMETER_GREEK_MAP) -> str:
    """Return display text for a parallel-coordinate axis label."""
    if axis_label in LOSS_LABELS:
        return axis_label

    return get_parameter_display_label(axis_label, greek_map)


def get_physics_equation_lines(physics_model, config) -> list:
    """Return the physics model's ODE equations as a list of mathtext lines.

    The lookup mirrors the previous behaviour (model class name, config
    ``model_type``, model-provided attributes, then a prefix fallback). The
    equations are returned in full; no truncation is applied so that every
    compartment equation is rendered inside the boxed block.
    """
    model_name = type(physics_model).__name__
    normalized_model_name = model_name.replace("Model", "").upper()

    model_cfg = config.get("model", {}) or {}
    if hasattr(model_cfg, "get"):
        model_type_from_cfg = model_cfg.get("type", "")
    else:
        model_type_from_cfg = ""

    config_model_type_raw = str(config.get("model_type", "") or model_type_from_cfg or "")
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

        if key in PHYSICS_EQUATION_LATEX:
            raw = PHYSICS_EQUATION_LATEX[key]
            break

        lower_key = str(key).lower()
        if lower_key in PHYSICS_EQUATION_LATEX:
            raw = PHYSICS_EQUATION_LATEX[lower_key]
            break

    if raw is None:
        getter = getattr(physics_model, "get_equation_latex", None)
        if callable(getter):
            try:
                raw = getter()
            except Exception:
                raw = None

    if raw is None:
        for attr in ("equation_latex", "physics_equation_latex"):
            value = getattr(physics_model, attr, None)
            if value is not None:
                raw = value
                break

    if raw is None:
        for key in sorted(PHYSICS_EQUATION_LATEX.keys(), key=lambda item: len(str(item)), reverse=True):
            key_str = str(key)
            if key_str and key_str.upper() in normalized_model_name:
                raw = PHYSICS_EQUATION_LATEX[key]
                break

    return normalize_equation_lines(raw)


def build_experiment_summary(config, physics_model, model) -> str:
    """Build a single-line experiment summary for the top of each PDF page.

    Mirrors the style of the network-prediction summary, packing the model
    equations name, loss weights, optimizer settings, collocation count,
    learned parameters, observed variables and architecture into one line.
    """
    model_name = type(physics_model).__name__.replace("Model", "")

    data_weight = fmt_scalar(config.get('data_weight', 1.0))
    physics_weight = fmt_scalar(config.get('physics_weight', 1.0))

    adam_epochs = config.get('adam_epochs', 5000)
    adam_lr = fmt_scalar(config.get('adam_lr', 1e-3))
    lbfgs_max_iter = config.get('lbfgs_max_iter', 100)
    n_collocation_points = config.get('n_collocation_points', 100)

    learnable = config.get('learnable_parameters', {}) or {}
    if learnable:
        learned = "[" + ", ".join(str(key) for key in learnable.keys()) + "]"
    else:
        learned = "None"

    observables_config = config.get('observables', {}) or {}
    observed_vars = observables_config.get('observed_variables', []) or []
    if observed_vars:
        observed = "[" + ", ".join(str(var) for var in observed_vars) + "]"
    else:
        observed = "None"

    backbone = backbone_label(getattr(model, "backbone", None))
    encoder = encoder_label(getattr(model, "encoder", None))
    head = head_label(getattr(model, "head", None))

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


def build_experiment_header_text(config, physics_model, model) -> str:
    """Build the top-of-page description for loss-landscape PDF pages.

    This is intentionally closer to the network-prediction summary in
    ``plotting.plot_actual_vs_predicted``: a readable wrapped block with a
    boxed background, plus the physics equations for the current model.
    """
    summary_text = build_experiment_summary(config, physics_model, model)
    wrapped_summary = textwrap.fill(summary_text, width=110)

    equation_lines = get_physics_equation_lines(physics_model, config)
    if not equation_lines:
        return wrapped_summary

    equation_text = "\n".join(
        ensure_math_delimiters(line) for line in equation_lines
    )
    return f"{wrapped_summary}\n\nPhysics equations:\n{equation_text}"


def get_unique_parameter_names_from_results(results: list) -> list:
    """Fallback parameter names extracted from result rows."""
    names = []
    for result in results:
        name = result.get('parameter')
        if name and name not in RESERVED_COLUMNS and name not in names:
            names.append(name)
    return names


def get_axis_value(axis_label: str, result: dict, parameter_names: list, loss_key_by_label: dict):
    """Extract the numeric value for a given axis label from a result row."""
    if axis_label in parameter_names:
        if axis_label in result:
            return result[axis_label]
        if result.get('parameter') == axis_label:
            return result.get('value')
        return None

    return result.get(loss_key_by_label[axis_label])


def compute_global_axis_ranges(rows: list, parameter_names: list) -> Dict[str, Tuple[float, float]]:
    """Compute one (lo, hi) range per dimension across ALL loss-landscape rows.

    Parameter axes keep their own linear scale. All loss axes share a single
    global log scale so Physics/Data/Total can be compared directly in the
    same chart and across snapshots.
    """
    if not parameter_names:
        parameter_names = get_unique_parameter_names_from_results(rows)

    loss_labels = list(LOSS_LABELS)
    loss_keys = ['physics_loss', 'data_loss', 'total_loss']
    loss_key_by_label = {label: key for label, key in zip(loss_labels, loss_keys)}

    dimensions = parameter_names + loss_labels
    loss_labels_set = set(loss_labels)

    ranges: Dict[str, Tuple[float, float]] = {}
    all_loss_values: list = []

    for axis_label in dimensions:
        values = [
            get_axis_value(axis_label, result, parameter_names, loss_key_by_label)
            for result in rows
        ]

        if axis_label in loss_labels_set:
            all_loss_values.extend(values)

        ranges[axis_label] = axis_range_from_values(
            values,
            log_scale=axis_label in loss_labels_set
        )

    common_loss_range = axis_range_from_values(all_loss_values, log_scale=True)
    for axis_label in loss_labels:
        ranges[axis_label] = common_loss_range

    return ranges


def find_best_result_row(results: list) -> Optional[dict]:
    """Return the loss-landscape row with the lowest finite total loss.

    When the current/ground-truth row is numerically close to the best
    minimum, prefer it so the highlighted curve corresponds to the true
    parameters unless a clearly better sweep point exists.
    """
    best_row = None
    best_loss = float('inf')
    current_row = None
    current_loss = float('inf')

    for result in results:
        try:
            total_loss = float(result.get('total_loss', float('nan')))
        except (TypeError, ValueError):
            continue

        if not np.isfinite(total_loss):
            continue

        is_current = bool(result.get('is_current', False))

        if is_current and current_row is None:
            current_row = result
            current_loss = total_loss

        if best_row is None or total_loss < best_loss:
            best_row = result
            best_loss = total_loss
        elif np.isclose(total_loss, best_loss, rtol=1e-12, atol=1e-12):
            if is_current and not bool(best_row.get('is_current', False)):
                best_row = result
                best_loss = total_loss

    if current_row is not None and np.isclose(current_loss, best_loss, rtol=1e-4, atol=1e-6):
        return current_row

    return best_row


def is_reddish_color(color: str) -> bool:
    """Return True for colors that are visually red or red-dominant."""
    if not isinstance(color, str) or not color.startswith("#"):
        return False

    hex_color = color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)

    if len(hex_color) != 6:
        return False

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return False

    return r > 180 and (r - max(g, b)) > 70 and g < 120


def get_sweep_colors(n_colors: int) -> list:
    """Return parameter-grid colors in the same order as constants."""
    if n_colors <= 0:
        return []

    base_palette = list(PARAMETER_GRID_COLORS)
    if not base_palette:
        base_palette = list(HIGH_CONSPICUITY_PALETTE)

    extended_palette = base_palette + [
        color for color in HIGH_CONSPICUITY_PALETTE
        if color not in base_palette
    ]

    if n_colors <= len(extended_palette):
        return extended_palette[:n_colors]

    try:
        cmap = plt.colormaps["hsv"]
    except AttributeError:
        cmap = plt.cm.get_cmap("hsv")

    return [cmap(i / max(n_colors, 1)) for i in range(n_colors)]


def get_best_curve_color(used_colors: Optional[list] = None) -> str:
    """Choose a high-conspicuity color for the best curve.

    The color is taken from the high-conspicuity palette and, when possible,
    avoids colors already assigned to swept parameters, phase indicators, and
    red tones so the highlighted best curve remains visually distinct.
    """
    used = set()
    if used_colors:
        for color in used_colors:
            if isinstance(color, str):
                used.add(color.lower())

    for color in HIGH_CONSPICUITY_PALETTE:
        if color.lower() in used or is_reddish_color(color):
            continue
        return color

    for color in [
        "#007AFF",
        "#FF9500",
        "#34C759",
        "#00C7BE",
        "#5856D6",
        "#AF52DE",
        "#FFCC00",
    ]:
        if color.lower() not in used and not is_reddish_color(color):
            return color

    return TRAINER_PLOT_COLORS['best_curve']


def select_global_best_curve_color(parameter_names: list, rows: Optional[list] = None) -> str:
    """Choose one best-curve color for the whole PDF artifact."""
    if not parameter_names and rows:
        parameter_names = get_unique_parameter_names_from_results(rows)

    sweep_colors = get_sweep_colors(max(len(parameter_names), 1))
    avoid_colors = [
        TRAINER_PLOT_COLORS['current'],
        TRAINER_PLOT_COLORS['sweep'],
        TRAINER_PLOT_COLORS['adam'],
        TRAINER_PLOT_COLORS['lbfgs'],
        TRAINER_PLOT_COLORS['transition'],
    ] + list(sweep_colors)

    return get_best_curve_color(avoid_colors)


def build_global_legend_handles(best_curve_color: str, tasks: Optional[list] = None) -> list:
    """Build a compact global legend for the loss-landscape PDF."""
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=TRAINER_PLOT_COLORS['current'],
            linewidth=3.5,
            label="Current / ground truth",
        ),
        plt.Line2D(
            [0],
            [0],
            color=best_curve_color,
            linewidth=4.5,
            alpha=0.95,
            label="Best total loss",
        ),
        plt.Line2D(
            [0],
            [0],
            color=TRAINER_PLOT_COLORS['sweep'],
            linewidth=1.3,
            alpha=0.55,
            label="Swept parameters",
        ),
    ]

    phases = set()
    if tasks:
        for task in tasks:
            phase = str(task.get('checkpoint_info', {}).get('phase', '')).lower().replace('-', '').replace('_', '')
            if phase:
                phases.add(phase)

    if 'adam' in phases:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=TRAINER_PLOT_COLORS['adam'],
                linewidth=2.5,
                label="Adam",
            )
        )

    if 'lbfgs' in phases:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=TRAINER_PLOT_COLORS['lbfgs'],
                linewidth=2.5,
                label="L-BFGS",
            )
        )

    if 'adam' in phases and 'lbfgs' in phases:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=TRAINER_PLOT_COLORS['transition'],
                linewidth=2,
                linestyle=':',
                label="Adam -> L-BFGS",
            )
        )

    return handles


def draw_cosine_similarity_development(ax, tasks: list, config: dict) -> None:
    """Plot cosine similarity over training development in a blank subplot.

    The y-axis is fixed from -1 to 1. A vertical dotted line marks the
    Adam -> L-BFGS transition whenever L-BFGS checkpoints are present.
    """
    if not tasks:
        ax.axis('off')
        return

    adam_epochs = as_scalar_float(config.get('adam_epochs', 5000))
    if not np.isfinite(adam_epochs):
        adam_epochs = 5000.0

    points = []
    for task in tasks:
        info = task.get('checkpoint_info', {})
        phase = str(info.get('phase', '')).lower().replace('-', '').replace('_', '')

        x = as_scalar_float(info.get('global_step'))
        if not np.isfinite(x):
            epoch = as_scalar_float(info.get('epoch'))
            if not np.isfinite(epoch):
                continue
            x = epoch if phase != 'lbfgs' else adam_epochs + epoch

        y = as_scalar_float(info.get('gradient_cosine'))
        if not np.isfinite(y):
            y = 0.0
        y = float(np.clip(y, -1.0, 1.0))

        points.append((float(x), y, phase))

    if not points:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel('Training step')
        ax.set_ylabel('Cosine similarity')
        ax.set_title(
            'Gradient Conflict Over Training\n(y-axis fixed from -1 to 1)',
            fontsize=12,
        )
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.text(
            0.5,
            0.5,
            'No cosine similarity data',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=12,
        )
        return

    points.sort(key=lambda item: item[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    adam_xs = [p[0] for p in points if p[2] == 'adam']
    adam_ys = [p[1] for p in points if p[2] == 'adam']
    lbfgs_xs = [p[0] for p in points if p[2] == 'lbfgs']
    lbfgs_ys = [p[1] for p in points if p[2] == 'lbfgs']

    def plot_phase_line(phase_xs: list, phase_ys: list, color: str) -> None:
        if not phase_xs:
            return

        if len(phase_xs) == 1:
            x0 = phase_xs[0]
            y0 = phase_ys[0]
            x_span = max(abs(x0), 1.0) * 0.01
            ax.plot(
                [x0 - x_span, x0 + x_span],
                [y0, y0],
                color=color,
                linewidth=2.5,
                solid_capstyle='round',
                zorder=2,
            )
        else:
            ax.plot(
                phase_xs,
                phase_ys,
                color=color,
                linewidth=2.5,
                solid_capstyle='round',
                zorder=2,
            )

    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5, zorder=1)

    plot_phase_line(adam_xs, adam_ys, TRAINER_PLOT_COLORS['adam'])
    plot_phase_line(lbfgs_xs, lbfgs_ys, TRAINER_PLOT_COLORS['lbfgs'])

    if not adam_xs and not lbfgs_xs:
        ax.plot(xs, ys, color='black', linewidth=2, alpha=0.75, zorder=2)

    transition_x = None
    if lbfgs_xs:
        if adam_xs and min(xs) <= adam_epochs <= max(xs):
            transition_x = adam_epochs
        else:
            transition_x = min(lbfgs_xs)

    if transition_x is not None:
        ax.axvline(
            transition_x,
            linestyle=':',
            color=TRAINER_PLOT_COLORS['transition'],
            linewidth=2,
            alpha=0.85,
            zorder=1,
        )
        ax.text(
            transition_x,
            0.96,
            'Adam -> L-BFGS',
            rotation=90,
            va='top',
            ha='left',
            fontsize=9,
            color=TRAINER_PLOT_COLORS['transition'],
            zorder=4,
        )

    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_min -= 1.0
        x_max += 1.0
    else:
        x_span = x_max - x_min
        x_min -= x_span * 0.05
        x_max += x_span * 0.05

    if transition_x is not None:
        span = x_max - x_min
        x_min = min(x_min, transition_x - span * 0.02)
        x_max = max(x_max, transition_x + span * 0.02)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Cosine similarity')
    ax.set_title(
        'Gradient Conflict Over Training\n(y-axis fixed from -1 to 1)',
        fontsize=12,
    )
    ax.grid(True, linestyle='--', alpha=0.35)


def draw_loss_landscape_axis(
    ax,
    results: list,
    checkpoint_info: dict,
    parameter_names: Optional[list] = None,
    global_axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    best_curve_color: Optional[str] = None,
):
    """Draw a full parallel-coordinate loss landscape on a given axis.

    Loss axes are rendered on a shared log scale across snapshots, while
    parameter axes remain linear. The current/actual value line is drawn in
    black to distinguish it from the colored grid-search sweep lines.

    The lowest-total-loss curve is highlighted in a globally selected color.

    Args:
        ax: Matplotlib axis to draw on
        results: List of dictionaries containing loss landscape data
        checkpoint_info: Dictionary with phase and epoch information
        parameter_names: Ordered compartmental parameter names. When None or
            empty, names are derived from the result rows.
        global_axis_ranges: Optional shared (lo, hi) range per dimension. When
            provided, every subplot uses the same scale for each axis so the
            charts are directly comparable.
        best_curve_color: Optional globally selected color for the best curve.
    """
    if not results:
        ax.axis('off')
        return

    if not parameter_names:
        parameter_names = get_unique_parameter_names_from_results(results)

    loss_labels = list(LOSS_LABELS)
    loss_keys = ['physics_loss', 'data_loss', 'total_loss']
    loss_key_by_label = {label: key for label, key in zip(loss_labels, loss_keys)}
    loss_labels_set = set(loss_labels)

    dimensions = parameter_names + loss_labels
    n_dims = len(dimensions)
    n_param_axes = len(parameter_names)
    x_positions = np.arange(n_dims)

    def axis_value(axis_label: str, result: dict):
        return get_axis_value(axis_label, result, parameter_names, loss_key_by_label)

    current_values = {}
    for axis_label in dimensions:
        for result in results:
            if not result.get('is_current', False):
                continue

            val = axis_value(axis_label, result)
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
    all_loss_values: list = []
    for axis_label in dimensions:
        values = [axis_value(axis_label, result) for result in results]

        if axis_label in loss_labels_set:
            all_loss_values.extend(values)

        axis_ranges[axis_label] = axis_range_from_values(
            values,
            log_scale=axis_label in loss_labels_set
        )

    common_loss_range = axis_range_from_values(all_loss_values, log_scale=True)
    for axis_label in loss_labels:
        axis_ranges[axis_label] = common_loss_range

    # Override local ranges with the shared global ranges so every subplot on
    # every page uses the same scale for each dimension (e.g. Data Loss).
    if global_axis_ranges:
        for axis_label in dimensions:
            if axis_label in global_axis_ranges:
                axis_ranges[axis_label] = global_axis_ranges[axis_label]

    use_log_axis = {}
    for axis_label in dimensions:
        lo, hi = axis_ranges[axis_label]
        use_log_axis[axis_label] = bool(
            axis_label in loss_labels_set and lo > 0 and hi > 0 and hi != lo
        )

    def value_to_y(axis_label: str, value: float) -> float:
        lo, hi = axis_ranges[axis_label]

        if use_log_axis.get(axis_label, False):
            if value <= 0:
                return 0.0

            try:
                log_value = np.log10(float(value))
                log_lo = np.log10(lo)
                log_hi = np.log10(hi)
            except (ValueError, FloatingPointError):
                return 0.5

            if log_hi == log_lo:
                return 0.5

            return (log_value - log_lo) / (log_hi - log_lo)

        if hi == lo:
            return 0.5

        return (float(value) - lo) / (hi - lo)

    def line_points(result):
        ys = []
        for axis_label in dimensions:
            val = axis_value(axis_label, result)
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

    best_result = find_best_result_row(results)

    # Draw vertical axis lines.
    for i in range(n_dims):
        ax.plot([i, i], [0, 1], color='black', linewidth=1.5, zorder=1)

    # Draw small bottom ticks for each dimension axis.
    for i in x_positions:
        ax.plot([i, i], [-0.012, 0.0], color='black', linewidth=1.0, zorder=1)

    # Axis labels on top with a blank background so they stand out.
    for i, dim in enumerate(dimensions):
        display_label = get_axis_display_label(dim)

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
        is_log_axis = use_log_axis.get(axis_label, False)

        for t in range(n_ticks):
            frac = t / (n_ticks - 1) if n_ticks > 1 else 0.5

            if is_log_axis:
                log_lo = np.log10(lo)
                log_hi = np.log10(hi)
                val = float(10 ** (log_lo + frac * (log_hi - log_lo)))
            else:
                val = lo + frac * (hi - lo)

            tick_fracs.append(frac)

            # Draw a tick mark on this axis.
            ax.plot(
                [i - 0.012, i + 0.012],
                [frac, frac],
                color='black',
                linewidth=1.0,
                zorder=1.5,
            )

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
            if 0.0 <= current_y <= 1.0:
                # Draw a distinct current-value tick mark on every axis.
                ax.plot(
                    [i - 0.022, i + 0.022],
                    [current_y, current_y],
                    color='black',
                    linewidth=2.0,
                    zorder=1.6,
                )

                if not any(abs(current_y - frac) < 0.02 for frac in tick_fracs):
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

    # High-contrast color mapping for swept parameters.
    colors = get_sweep_colors(max(n_param_axes, 1))
    if best_curve_color is None:
        best_curve_color = select_global_best_curve_color(parameter_names, results)

    param_color_map = {
        name: colors[idx % len(colors)]
        for idx, name in enumerate(parameter_names)
    }

    # Draw lines for each result.
    for result in results:
        ys = line_points(result)
        if ys is None:
            continue

        color = param_color_map.get(result.get('parameter'), TRAINER_PLOT_COLORS['sweep'])
        ax.plot(x_positions, ys, color=color, alpha=0.55, linewidth=1.3, zorder=2)

    # Highlight current parameter values with a distinct black line.
    for result in results:
        if not result.get('is_current', False):
            continue

        ys = line_points(result)
        if ys is None:
            continue

        ax.plot(
            x_positions,
            ys,
            color=TRAINER_PLOT_COLORS['current'],
            linewidth=3.5,
            zorder=3,
        )

    # Highlight the best available curve: the parameter combination with
    # the lowest finite total loss in this checkpoint's sweep.
    best_ys = None
    if best_result is not None:
        best_ys = line_points(best_result)

    if best_ys is not None:
        ax.plot(
            x_positions,
            best_ys,
            color=best_curve_color,
            linewidth=4.5,
            alpha=0.95,
            linestyle='-',
            zorder=3.5,
        )

    # Gradient-conflict bar: cosine similarity between data and physics
    # gradients. -1 means conflictuous optimization, +1 means coherent.
    cosine = checkpoint_info.get('gradient_cosine')
    try:
        cosine = float(cosine)
    except (TypeError, ValueError):
        cosine = 0.0

    if not np.isfinite(cosine):
        cosine = 0.0

    cosine = float(np.clip(cosine, -1.0, 1.0))

    bar_y = -0.12
    bar_half_height = 0.022
    bar_left = -0.45
    bar_right = n_dims - 0.55
    marker_x = bar_left + ((cosine + 1.0) / 2.0) * (bar_right - bar_left)

    gradient_values = np.linspace(-1.0, 1.0, 256).reshape(1, -1)
    ax.imshow(
        gradient_values,
        extent=(
            bar_left,
            bar_right,
            bar_y - bar_half_height,
            bar_y + bar_half_height,
        ),
        aspect='auto',
        cmap='RdYlGn',
        vmin=-1.0,
        vmax=1.0,
        zorder=1,
    )

    ax.plot(
        [marker_x, marker_x],
        [bar_y - bar_half_height, bar_y + bar_half_height],
        color='black',
        linewidth=1.5,
        zorder=2,
    )

    ax.text(
        marker_x,
        bar_y + 0.045,
        f'cos = {cosine:.2f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
        color='black',
        zorder=3,
        bbox=dict(facecolor='white', edgecolor='none', pad=1.0),
    )

    ax.text(
        bar_left,
        bar_y - 0.045,
        'Conflictuous optimization',
        ha='left',
        va='top',
        fontsize=9,
        color='darkred',
        zorder=3,
    )

    ax.text(
        bar_right,
        bar_y - 0.045,
        'Coherent optimization',
        ha='right',
        va='top',
        fontsize=9,
        color='darkgreen',
        zorder=3,
    )

    ax.set_xlim(-0.5, n_dims - 0.5)
    ax.set_ylim(-0.25, 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = (
        f'Loss Landscape - {checkpoint_info["phase"].upper()} Epoch {checkpoint_info["epoch"]}\n'
        'Losses are in log scale'
    )
    ax.set_title(title, fontsize=12)


def finalize_loss_landscape_pdf(tasks, rows, parameter_names, config, header_text, pdf_path):
    """Write all pending loss-landscape plots into one multi-page PDF.

    Plots are organized into two columns per PDF page. The first page carries
    the general experiment summary and physics equations at the very top.
    Subsequent pages omit the header to avoid repetition. All subplots share
    a single global scale per dimension so that, e.g., the Data Loss axis
    uses the same min/max across every snapshot.

    The cosine-similarity development plot is always appended as the final
    subplot of the PDF, opening a new page when the last landscape page has
    no blank slot.

    Output paths are computed by the caller and passed in via ``pdf_path``;
    clearing the queued ``tasks`` is also the caller's responsibility.
    """
    if not tasks or pdf_path is None:
        return

    try:
        global_axis_ranges = compute_global_axis_ranges(rows, parameter_names)
        best_curve_color = select_global_best_curve_color(parameter_names, rows)
        legend_handles = build_global_legend_handles(
            best_curve_color=best_curve_color,
            tasks=tasks
        )

        header_line_count = max(1, len(header_text.splitlines()))
        rect_top = max(0.50, 0.88 - 0.045 * header_line_count)

        with PdfPages(pdf_path) as pdf:
            # Reserve one final slot for the cosine-similarity plot so it is
            # always rendered, regardless of whether the last page has a
            # blank slot.
            n_slots = len(tasks) + 1

            for start_idx in range(0, n_slots, 2):
                first_page = start_idx == 0
                fig, axes = plt.subplots(1, 2, figsize=(16, 8), squeeze=False)

                for col_idx in range(2):
                    ax = axes[0][col_idx]
                    slot_idx = start_idx + col_idx

                    if slot_idx < len(tasks):
                        task = tasks[slot_idx]
                        draw_loss_landscape_axis(
                            ax,
                            task['results'],
                            task['checkpoint_info'],
                            parameter_names=parameter_names,
                            global_axis_ranges=global_axis_ranges,
                            best_curve_color=best_curve_color
                        )
                    elif slot_idx == len(tasks):
                        draw_cosine_similarity_development(ax, tasks, config)
                    else:
                        ax.axis('off')

                if first_page:
                    if legend_handles:
                        fig.legend(
                            handles=legend_handles,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 0.995),
                            ncol=min(len(legend_handles), 6),
                            frameon=False,
                            fontsize=9,
                            handlelength=2.2,
                            handletextpad=0.4,
                        )

                    fig.text(
                        0.5,
                        0.945,
                        header_text,
                        ha='center',
                        va='top',
                        fontsize=11,
                        linespacing=1.35,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        transform=fig.transFigure,
                    )

                rect_top_page = rect_top if first_page else 0.94
                fig.tight_layout(rect=[0, 0, 1, rect_top_page])
                pdf.savefig(fig)
                plt.close(fig)

        logger.info(f"Combined loss landscape PDF saved to: {pdf_path}")
    except Exception as e:
        logger.warning(f"Failed to save combined loss landscape PDF: {e}")
