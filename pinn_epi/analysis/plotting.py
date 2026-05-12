"""Decoupled plotting functions for PINN model analysis.

All functions accept pre-computed arrays or model instances and never
compute gradients or run optimisation themselves.
"""

from __future__ import annotations

from typing import Optional
import datetime
import textwrap

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Nature-style rcParams — apply once via apply_nature_style()
# ---------------------------------------------------------------------------

NATURE_STYLE: dict = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,  # Base font size
    'axes.labelsize': 18,  # Axis labels
    'axes.titlesize': 22,  # Axis titles
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,  # Legend
    'figure.titlesize': 24,  # Figure title (largest)
    'figure.dpi': 300,  # Default resolution
}

# Font size hierarchy: figure title > axis labels > axis titles, legend and simulation description
FONT_HIERARCHY: dict = {
    'figure_title': 24,     # Largest
    'axis_labels': 18,      # Medium
    'axis_titles': 16,      # Smaller
    'legend': 16,           # Same as axis titles
    'description': 14,      # Smallest
    'description_min': 10   # Minimum for wrapped text
}

# Color-blind friendly palette
COMPARTMENT_COLORS = {                                                                
   'S': '#377EB8',
   'I': '#E41A1C',
   'R': '#4DAF4A',
   'E': '#984EA3',
   'D': '#A65628',
   'V': '#FF7F00',
   'H': '#FFFF33',              
   'C': '#F781BF',                  
   'Q': '#999999',
   'A': '#8DD3C7',             
   'P': '#BEBADA',                                                                                                
   'W': '#FDB462',
   'B': '#B3DE69',
 }                                 

# Resolution levels
RESOLUTION_LEVELS = {
    'low': 300,
    'medium': 600,
    'high': 1200
}

def apply_nature_style() -> None:
    """Apply Nature-journal-style matplotlib rcParams globally."""
    plt.rcParams.update(NATURE_STYLE)

def plot_compartmental_solution(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    compartment_colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    resolution: str = 'medium',
    model_params: Optional[dict] = None,
    initial_conditions: Optional[list] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the ODE system solution for any CompartmentalModel.

    This function is purely presentational and does not compute the solution.

    Args:
        t: Array of time points.
        trajectories: Dictionary mapping each compartment name to its solved numpy array.
        compartment_colors: Optional mapping of compartment name to colour string.
        title: Optional plot title.
        ax: Optional existing Axes to draw on.  If None a new figure is created.
        show: Whether to call plt.show() at the end.  Defaults to False so
            the caller controls display.
        save_path: Optional file path to save the figure (e.g.
            'figures/sir_sanity.png').  Directory must exist.
        resolution: Figure resolution level ('low'=300, 'medium'=600, 'high'=1200).
        model_params: Optional dictionary of model parameters for display.
        initial_conditions: Optional list of initial conditions for display.

    Returns:
        (fig, ax) where fig is the matplotlib Figure and ax is the Axes.
    """

    if resolution not in RESOLUTION_LEVELS:
        raise ValueError(f"Resolution must be one of {list(RESOLUTION_LEVELS.keys())}")
    dpi_value = RESOLUTION_LEVELS[resolution]

    compartment_names = list(trajectories.keys())

    # --- plotting ---
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi_value)
    else:
        fig = ax.get_figure()

    # Use default compartment colors if none provided
    if compartment_colors is None:
        compartment_colors = COMPARTMENT_COLORS

    for name in compartment_names:
        color = compartment_colors.get(name, f'C{compartment_names.index(name)}')
        ax.plot(t, trajectories[name], label=f'{name}(t)', color=color, lw=2)

    # Set axis labels with proper font sizing
    ax.set_xlabel('Time (days)', fontsize=FONT_HIERARCHY['axis_labels'])
    ax.set_ylabel('Fraction of Population', fontsize=FONT_HIERARCHY['axis_labels'])
    
    # Set title with model compartments if not provided
    if title is None:
        compartments = "".join(compartment_names)
        title = f"{compartments} Differential Equation Solutions"
    
    ax.set_title(title, pad=60, fontsize=FONT_HIERARCHY['figure_title'])

    if created_fig:
        # Add legend above the plot with proper spacing
        legend = ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=len(compartment_names), 
                          columnspacing=1.5, handletextpad=0.5, fontsize=FONT_HIERARCHY['legend'])
        # Use subplots_adjust instead of tight_layout for precise control
        plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.25)

    # Add model information as text below the figure with line wrapping
    # Only add info text if model_params and initial_conditions are provided
    if model_params is not None and initial_conditions is not None:
        # Convert parameter names to LaTeX where appropriate
        param_items = []
        for k, v in model_params.items():
            # Convert common Greek letters to LaTeX
            latex_k = k
            if k == 'beta':
                latex_k = r'$\beta$'
            elif k == 'gamma':
                latex_k = r'$\gamma$'
            elif k == 'sigma':
                latex_k = r'$\sigma$'
            elif k == 'mu':
                latex_k = r'$\mu$'
            elif k == 'rho':
                latex_k = r'$\rho$'
            elif k == 'alpha':
                latex_k = r'$\alpha$'
            elif k == 'delta':
                latex_k = r'$\delta$'
            elif k == 'theta':
                latex_k = r'$\theta$'
            elif k == 'lambda':
                latex_k = r'$\lambda$'
            elif k == 'epsilon':
                latex_k = r'$\epsilon$'
            elif k == 'omega':
                latex_k = r'$\omega$'
            elif k == 'tau':
                latex_k = r'$\tau$'
            elif k == 'phi':
                latex_k = r'$\phi$'
            elif k == 'chi':
                latex_k = r'$\chi$'
            elif k == 'psi':
                latex_k = r'$\psi$'
            elif k == 'eta':
                latex_k = r'$\eta$'
            elif k == 'nu':
                latex_k = r'$\nu$'
            elif k == 'xi':
                latex_k = r'$\xi$'
            elif k == 'pi':
                latex_k = r'$\pi$'
            elif k == 'kappa':
                latex_k = r'$\kappa$'
            param_items.append(f"{latex_k}={v}")
        
        param_str = ", ".join(param_items)
        y0_items = []
        for i, name in enumerate(compartment_names):
            y0_items.append(f"{name}(0)={initial_conditions[i]:.2f}")
        y0_str = ", ".join(y0_items)
        info_text = f"Parameters: {param_str} | Initial: {y0_str}"
        
        # Wrap text to fit within figure width
        wrapped_text = textwrap.fill(info_text, width=80)
        
        # Count lines in wrapped text
        lines = wrapped_text.count('\n') + 1
        
        # Adjust font size if text exceeds 2 lines
        font_size = FONT_HIERARCHY['description']
        if lines > 2:
            font_size = max(FONT_HIERARCHY['description_min'], 
                          FONT_HIERARCHY['description'] - (lines - 2) * 2)  # Reduce font size but keep minimum
        
        # Position text just below the x-axis label using axes coordinates
        ax.text(0.5, -0.25, wrapped_text, ha='center', va='top', fontsize=font_size,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                transform=ax.transAxes)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi_value)
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax
