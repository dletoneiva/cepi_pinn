"""Decoupled plotting functions for PINN model analysis.

All functions accept pre-computed arrays or model instances and never
compute gradients or run optimisation themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from pinn_epi.models.physics import CompartmentalModel

# ---------------------------------------------------------------------------
# Nature-style rcParams — apply once via apply_nature_style()
# ---------------------------------------------------------------------------

NATURE_STYLE: dict = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'figure.dpi': 100,
}

# Color-blind friendly palette
COMPARTMENT_COLORS = {
    'S': '#1f77b4',  # blue
    'E': '#ff7f0e',  # orange
    'I': '#d62728',  # red
    'R': '#2ca02c',  # green
}

def apply_nature_style() -> None:
    """Apply Nature-journal-style matplotlib rcParams globally."""
    plt.rcParams.update(NATURE_STYLE)


# ---------------------------------------------------------------------------
# Generic compartmental model solution plotter
# ---------------------------------------------------------------------------

def plot_compartmental_solution(
    model: "CompartmentalModel",
    t_span: list[float],
    y0: list[float],
    params: dict,
    t_eval: Optional[np.ndarray] = None,
    compartment_colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    **solve_ivp_kwargs,
) -> tuple[plt.Figure, plt.Axes, dict[str, np.ndarray]]:
    """Solve and plot the ODE system defined by any CompartmentalModel.

    This function is intentionally generic: it reads the compartment names
    directly from the model so it works for SIR, SEIR, SI, or any future
    model without modification.

    Args:
        model: Any instance of CompartmentalModel (SIRModel, SEIRModel, ...).
        t_span: [t_start, t_end] integration interval.
        y0: Initial conditions, one value per compartment in the same order
            as model.compartment_names.
        params: Parameter dict forwarded to model.get_derivatives.
        t_eval: Optional array of time points at which to store the solution.
            Defaults to 300 evenly-spaced points over t_span.
        compartment_colors: Optional mapping of compartment name to colour
            string, e.g. {'S': 'blue', 'I': 'red', 'R': 'green'}.
        title: Optional plot title.
        ax: Optional existing Axes to draw on.  If None a new figure is
            created.
        show: Whether to call plt.show() at the end.  Defaults to False so
            the caller controls display.
        save_path: Optional file path to save the figure (e.g.
            'figures/sir_sanity.png').  Directory must exist.
        **solve_ivp_kwargs: Extra keyword arguments forwarded to
            scipy.integrate.solve_ivp (e.g. method='RK45', rtol=1e-8).

    Returns:
        (fig, ax, trajectories) where trajectories is a dict mapping each
        compartment name to its solved numpy array — ready for downstream
        storage or further analysis.
    """
    import torch  # local import keeps plotting.py importable without torch

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 300)

    compartment_names: list[str] = model.compartment_names
    n_compartments = len(compartment_names)

    if len(y0) != n_compartments:
        raise ValueError(
            f"y0 has {len(y0)} entries but model has "
            f"{n_compartments} compartments: {compartment_names}"
        )

    # --- wrap model.get_derivatives for scipy ---
    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        u = torch.tensor(y, dtype=torch.float32)
        t_t = torch.tensor(t, dtype=torch.float32)
        du = model.get_derivatives(t_t, u, params)
        return du.detach().numpy()

    sol = solve_ivp(
        ode_rhs,
        t_span,
        y0,
        t_eval=t_eval,
        **solve_ivp_kwargs,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # --- build trajectories dict ---
    trajectories: dict[str, np.ndarray] = {
        name: sol.y[i] for i, name in enumerate(compartment_names)
    }

    # --- plotting ---
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    # Use default compartment colors if none provided
    if compartment_colors is None:
        compartment_colors = COMPARTMENT_COLORS

    for idx, name in enumerate(compartment_names):
        color = compartment_colors.get(name, f'C{idx}')
        ax.plot(sol.t, trajectories[name], label=f'{name}(t)', color=color, lw=2)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fraction of Population')
    if title:
        ax.set_title(title)

    # Add model information as text below the figure
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    y0_str = ", ".join([f"{name}(0)={y0[i]:.2f}" for i, name in enumerate(compartment_names)])
    info_text = f"Parameters: {param_str} | Initial: {y0_str}"
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if created_fig:
        # Add legend above the plot
        legend = ax.legend(loc='center', bbox_to_anchor=(0.5, 1.08), ncol=len(compartment_names))
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax, trajectories
