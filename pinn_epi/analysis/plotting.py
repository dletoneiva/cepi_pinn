from __future__ import annotations

from typing import Optional
import datetime
import textwrap

import matplotlib.pyplot as plt
import numpy as np

PLOT_STYLE: dict = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'figure.dpi': 300,
}

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

RESOLUTION_LEVELS = {
    'low': 300,
    'medium': 600,
    'high': 1200
}

def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)

def plot_compartmental_solution(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    compartment_colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    model_params: Optional[dict] = None,
    initial_conditions: Optional[list] = None,
) -> tuple[plt.Figure, plt.Axes]:

    apply_plot_style()
    compartment_names = list(trajectories.keys())

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
    else:
        fig = ax.get_figure()

    if compartment_colors is None:
        compartment_colors = COMPARTMENT_COLORS

    for name in compartment_names:
        color = compartment_colors.get(name, f'C{compartment_names.index(name)}')
        ax.plot(t, trajectories[name], label=f'{name}(t)', color=color, lw=2.5)

    ax.set_xlabel('Time (days)', fontsize=PLOT_STYLE['axes.labelsize'], labelpad=15)
    ax.set_ylabel('Fraction of Population', fontsize=PLOT_STYLE['axes.labelsize'], labelpad=15)
    
    ax.tick_params(axis='x', labelsize=PLOT_STYLE['xtick.labelsize'])
    ax.tick_params(axis='y', labelsize=PLOT_STYLE['ytick.labelsize'])

    if title is None:
        compartments = "".join(compartment_names)
        title = f"{compartments} Differential Equation Solutions"
    
    ax.set_title(title, pad=60, fontsize=PLOT_STYLE['figure.titlesize'])

    if created_fig:
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=len(compartment_names), 
                  columnspacing=1.5, handletextpad=0.5, fontsize=PLOT_STYLE['legend.fontsize'])
        plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.25)

    if model_params is not None and initial_conditions is not None:
        param_items = []
        for k, v in model_params.items():
            latex_k = GREEK_LETTERS.get(k, k)
            param_items.append(f"{latex_k}={v}")
        
        param_str = ", ".join(param_items)
        y0_items = []
        for i, name in enumerate(compartment_names):
            y0_items.append(f"{name}(0)={initial_conditions[i]:.2f}")
        y0_str = ", ".join(y0_items)
        info_text = f"Parameters: {param_str} | Initial: {y0_str}"
        
        wrapped_text = textwrap.fill(info_text, width=80)
        lines = wrapped_text.count('\n') + 1
        
        font_size = 14
        if lines > 2:
            font_size = max(10, 14 - (lines - 2) * 2)
        
        ax.text(0.5, -0.25, wrapped_text, ha='center', va='top', fontsize=font_size,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                transform=ax.transAxes)

    if save_path:
        fig.savefig(save_path, dpi=600)
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax