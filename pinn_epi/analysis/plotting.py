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
# Nature-style rcParams with font hierarchy — apply once via apply_nature_style()
# ---------------------------------------------------------------------------

# Combined NATURE_STYLE with FONT_HIERARCHY
NATURE_STYLE: dict = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,  # Base font size
    'axes.labelsize': 18,  # Axis labels (medium)
    'axes.titlesize': 22,  # Axis titles (smaller)
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.labelsize': 14,  # Tick labels (slightly smaller than base)
    'ytick.labelsize': 14,  # Tick labels (slightly smaller than base)
    'legend.fontsize': 16,  # Legend (same as axis titles)
    'figure.titlesize': 24,  # Figure title (largest)
    'figure.dpi': 300,  # Default resolution
}

# Greek letters mapping for LaTeX rendering
GREEK_LETTERS: dict = {
    'alpha': r'$\alpha$',
    'beta': r'$\beta$',
    'gamma': r'$\gamma$',
    'delta': r'$\delta$',
    'epsilon': r'$\epsilon$',
    'zeta': r'$\zeta$',
    'eta': r'$\eta$',
    'theta': r'$\theta$',
    'iota': r'$\iota$',
    'kappa': r'$\kappa$',
    'lambda': r'$\lambda$',
    'mu': r'$\mu$',
    'nu': r'$\nu$',
    'xi': r'$\xi$',
    'omicron': r'$\omicron$',
    'pi': r'$\pi$',
    'rho': r'$\rho$',
    'sigma': r'$\sigma$',
    'tau': r'$\tau$',
    'upsilon': r'$\upsilon$',
    'phi': r'$\phi$',
    'chi': r'$\chi$',
    'psi': r'$\psi$',
    'omega': r'$\omega$',
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
    'low': 600,
    'medium': 600,
    'high': 600
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
        resolution: Figure resolution level ('low'=600, 'medium'=600, 'high'=600).
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
    ax.set_xlabel('Time (days)', fontsize=NATURE_STYLE['axes.labelsize'])
    ax.set_ylabel('Fraction of Population', fontsize=NATURE_STYLE['axes.labelsize'])
    
    # Set title with model compartments if not provided
    if title is None:
        compartments = "".join(compartment_names)
        title = f"{compartments} Differential Equation Solutions"
    
    ax.set_title(title, pad=60, fontsize=NATURE_STYLE['figure.titlesize'])

    if created_fig:
        # Add legend above the plot with proper spacing
        legend = ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=len(compartment_names), 
                          columnspacing=1.5, handletextpad=0.5, fontsize=NATURE_STYLE['legend.fontsize'])
        # Use subplots_adjust instead of tight_layout for precise control
        plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.25)

    # Add model information as text below the figure with line wrapping
    # Only add info text if model_params and initial_conditions are provided
    if model_params is not None and initial_conditions is not None:
        # Convert parameter names to LaTeX where appropriate
        param_items = []
        for k, v in model_params.items():
            # Convert common Greek letters to LaTeX using the mapping
            latex_k = GREEK_LETTERS.get(k, k)  # Use original if not found in mapping
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
        font_size = 14  # Base description size
        if lines > 2:
            font_size = max(10, 14 - (lines - 2) * 2)  # Reduce font size but keep minimum of 10
        
        # Position text just below the x-axis label using axes coordinates
        ax.text(0.5, -0.25, wrapped_text, ha='center', va='top', fontsize=font_size,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                transform=ax.transAxes)

    if save_path:
        # Change extension to PDF if not already specified
        if save_path and not save_path.endswith('.pdf'):
            if '.' in save_path:
                save_path = save_path.rsplit('.', 1)[0] + '.pdf'
            else:
                save_path = save_path + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax
