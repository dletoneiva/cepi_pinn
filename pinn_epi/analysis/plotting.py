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
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'figure.dpi': 100,
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

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, name in enumerate(compartment_names):
        color = (
            compartment_colors.get(name)
            if compartment_colors
            else default_colors[idx % len(default_colors)]
        )
        ax.plot(sol.t, trajectories[name], label=f'{name}(t)', color=color, lw=2)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fraction of Population')
    ax.legend(frameon=True)
    if title:
        ax.set_title(title)

    # conservation check annotation
    total = np.sum(sol.y, axis=0)
    max_deviation = np.max(np.abs(total - total[0]))
    ax.annotate(
        f'Max conservation error: {max_deviation:.2e}',
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        fontsize=10,
        color='gray',
    )

    if created_fig:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax, trajectories


# ---------------------------------------------------------------------------
# Figure 1 — identifiability / ambiguity plot
# ---------------------------------------------------------------------------

def plot_figure_1_corrected(
    t_max: float,
    t_eval: np.ndarray,
    y0: list[float],
    beta_true: float,
    gamma_true: float,
    ambiguous_pairs: Optional[list[tuple[float, float]]] = None,
    t_transition: float = 12,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot epidemic dynamics showing the identifiability problem.

    Args:
        t_max: End time for simulation.
        t_eval: Time points array.
        y0: Initial conditions [S0, I0, R0].
        beta_true: True transmission rate.
        gamma_true: True recovery rate.
        ambiguous_pairs: List of (beta, gamma) pairs with same r = beta-gamma.
            Defaults to four pairs around the true values.
        t_transition: Time at which to draw the exponential/nonlinear boundary.
        show: Whether to call plt.show().  Defaults to False.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure object.
    """
    def _sir(t, y, b, g):
        S, I, R = y
        return [-b * S * I, b * S * I - g * I, g * I]

    if ambiguous_pairs is None:
        ambiguous_pairs = [
            (beta_true - 0.05, gamma_true - 0.05),
            (beta_true,        gamma_true),
            (beta_true + 0.05, gamma_true + 0.05),
            (beta_true + 0.10, gamma_true + 0.10),
        ]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ambiguous_pairs)))

    for (b, g), color in zip(ambiguous_pairs, colors):
        sol = solve_ivp(_sir, [0, t_max], y0, args=(b, g), t_eval=t_eval)
        is_truth = (b == beta_true and g == gamma_true)
        lw = 3.0 if is_truth else 2.0
        alpha = 1.0 if is_truth else 0.4
        label_str = (
            rf'Truth: $\beta={b:.2f}, \gamma={g:.2f}$'
            if is_truth
            else rf'$\beta={b:.2f}, \gamma={g:.2f}$'
        )
        ax.plot(t_eval, sol.y[1], color=color, lw=lw, ls='--', alpha=alpha, label=label_str)

    ax.axvline(x=t_transition, color='gray', linestyle=':', lw=2, alpha=0.7)
    ax.text(t_transition / 2, 0.2, 'Exponential Phase\n(Unidentifiable)',
            ha='center', color='gray', fontsize=14)
    ax.text(t_transition + (t_max - t_transition) / 2, 0.06,
            'Non-linear Phase\n(Identifiable)', ha='center', color='gray', fontsize=14)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Infected Fraction $I(t)$')
    ax.legend(frameon=True, loc='upper left', facecolor='white')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Hessian scalarization plot
# ---------------------------------------------------------------------------

def plot_hessian_scalarization_r(
    t_max: float,
    t_eval: np.ndarray,
    y0: list[float],
    gamma_true: float,
    lambda_phys: float = 1.0,
    b_range: Optional[np.ndarray] = None,
    h_step: float = 0.005,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot normalised Hessian curves for data, physics, and total loss.

    Args:
        t_max: End time for simulation.
        t_eval: Time points array.
        y0: Initial conditions [S0, I0, R0].
        gamma_true: True recovery rate (marks epidemic threshold).
        lambda_phys: Weight on physics loss in total Hessian.
        b_range: Beta values to sweep.  Defaults to linspace(0, 1, 45).
        h_step: Finite-difference step size.
        show: Whether to call plt.show().  Defaults to False.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure object.
    """
    def _sir(t, y, b, g):
        S, I, R = y
        return [-b * S * I, b * S * I - g * I, g * I]

    if b_range is None:
        b_range = np.linspace(0.0, 1.0, 45)

    hn_data, hn_phys, hn_total = [], [], []

    print("  Calculating Hessians via finite differences...")
    for b_val in b_range:
        sol_ref = solve_ivp(_sir, [0, t_max], y0, args=(b_val, gamma_true), t_eval=t_eval)
        S_ref, I_ref = sol_ref.y[0], sol_ref.y[1]
        dI_dt_ref = b_val * S_ref * I_ref - gamma_true * I_ref
        energy = np.sum(I_ref ** 2)

        loss_d, loss_p = [], []
        for shift in [-h_step, 0, h_step]:
            b_shift = b_val + shift
            sol_s = solve_ivp(_sir, [0, t_max], y0, args=(b_shift, gamma_true), t_eval=t_eval)
            loss_d.append(np.sum((sol_s.y[1] - I_ref) ** 2))
            residual = dI_dt_ref - (b_shift * S_ref * I_ref - gamma_true * I_ref)
            loss_p.append(np.sum(residual ** 2))

        hd = (loss_d[2] - 2 * loss_d[1] + loss_d[0]) / h_step ** 2
        hp = (loss_p[2] - 2 * loss_p[1] + loss_p[0]) / h_step ** 2
        ht = hd + lambda_phys * hp

        if energy > 1e-8:
            hn_data.append(hd / energy)
            hn_phys.append(hp / energy)
            hn_total.append(ht / energy)
        else:
            hn_data.append(0)
            hn_phys.append(0)
            hn_total.append(0)

    def _format_panel(ax, y_data, title_suffix, show_ylabel):
        ax.plot(b_range, y_data, color='black', lw=3.0)
        ax.fill_between(b_range, 0, y_data, color='black', alpha=0.1)
        ax.axvline(x=gamma_true, color='black', linestyle=':', lw=2)
        max_idx = np.argmax(y_data)
        ax.axvline(x=b_range[max_idx], color='black', linestyle='--', lw=2, alpha=0.8)
        ax.set_xlabel(rf'{title_suffix} True $\beta$ Parameter Value')
        if show_ylabel:
            ax.set_ylabel(r'Normalized Hessian ($H_{norm}$)')

    fig = plt.figure(figsize=(16, 4), dpi=100)
    gs = gridspec.GridSpec(1, 3, wspace=0.1)

    _format_panel(fig.add_subplot(gs[0]), hn_data,
                  r'$H_{data}$ Data Hessian (a)' + '\n', show_ylabel=True)
    _format_panel(fig.add_subplot(gs[1]), hn_phys,
                  r'$H_{phys}$ Physics Hessian (b)' + '\n', show_ylabel=False)
    ax3 = fig.add_subplot(gs[2])
    _format_panel(ax3, hn_total,
                  r'$H_{total}$ Total Hessian (c)' + '\n', show_ylabel=False)

    max_idx_total = np.argmax(hn_total)
    h1 = ax3.axvline(x=gamma_true, color='black', linestyle=':', lw=2,
                     label=r'Epidemic Threshold ($r=0$)')
    h2 = ax3.axvline(x=b_range[max_idx_total], color='black', linestyle='--',
                     lw=2, alpha=0.8, label='Identifiability Peak')

    fig.legend([h1, h2], [r'Epidemic Threshold ($r=0$)', 'Identifiability Peak'],
               frameon=True, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, facecolor='white', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# SIR bifurcation diagram
# ---------------------------------------------------------------------------

def plot_sir_bifurcation(
    gamma: float = 0.4,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the SIR bifurcation diagram (DFE vs endemic equilibrium).

    Args:
        gamma: Recovery rate that sets the bifurcation point.
        show: Whether to call plt.show().  Defaults to False.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure object.
    """
    beta_range = np.linspace(0, 1.0, 500)
    dfe_branch = np.zeros_like(beta_range)
    ee_branch = 1 - (gamma / beta_range)
    ee_branch[beta_range <= gamma] = np.nan

    fig, ax = plt.subplots(figsize=(13, 4), dpi=100)
    ax.plot(beta_range[beta_range <= gamma], dfe_branch[beta_range <= gamma],
            'b-', lw=3, label='Stable DFE (Extinction)')
    ax.plot(beta_range[beta_range > gamma], dfe_branch[beta_range > gamma],
            'b--', lw=2, label='Unstable DFE')
    ax.plot(beta_range[beta_range > gamma], ee_branch[beta_range > gamma],
            'r-', lw=3, label='Stable EE (Epidemic)')
    ax.axvline(x=gamma, color='black', linestyle=':', alpha=0.7)
    ax.fill_between(beta_range[beta_range <= gamma], -0.05, 1, color='gray', alpha=0.1)
    ax.text(0.02, 0.8, "Near-degenerate Hessian", fontsize=16)
    ax.text(gamma + 0.05, 0.8, "Well-defined Hessian", fontsize=16)
    ax.set_xlabel(r'Transmission Rate ($\beta$)', fontsize=16)
    ax.set_ylabel(r'Equilibrium Infected Fraction ($I^*$)', fontsize=16)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(frameon=False, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Windowed PINN results
# ---------------------------------------------------------------------------

def plot_windowed_results(
    window_limits: np.ndarray,
    results_t: list[np.ndarray],
    results_d: list[np.ndarray],
    sol_ref,
    t_eval_ref: np.ndarray,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot windowed PINN predictions against ground truth.

    Args:
        window_limits: Array of window boundary times.
        results_t: List of prediction arrays for the 'match' model per window.
        results_d: List of prediction arrays for the 'drift' model per window.
        sol_ref: scipy ODE solution object for ground truth.
        t_eval_ref: Full time array corresponding to sol_ref.
        show: Whether to call plt.show().  Defaults to False.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 6))
    titles = ['Susceptibles (S) (a)', 'Infected (I) (b)',
              'Recovered (R) (c)', 'Sum (S+I+R) (d)']
    l_real = l_match = l_drift = None

    for i in range(len(window_limits) - 1):
        start, end = window_limits[i], window_limits[i + 1]
        mask = (t_eval_ref >= start) & (t_eval_ref <= end)
        t_win = t_eval_ref[mask]
        res_t, res_d = results_t[i], results_d[i]

        for j in range(3):
            ax = axs[j // 2, j % 2]
            if i == 0:
                l_real, = ax.plot(t_eval_ref, sol_ref.y[j], 'k-',
                                  alpha=0.15, lw=6, label='Real')
                if j % 2 == 0:
                    ax.set_ylabel('Value')
                if j // 2 == 1:
                    ax.set_xlabel('Time (days)')
            l_match, = ax.plot(t_win, res_t[:, j], 'g-', lw=2,
                               label=r'Match ($\beta=0.3$)' if i == 0 else '')
            l_drift, = ax.plot(t_win, res_d[:, j], 'r--', lw=2,
                               label=r'Drift ($\beta=0.5$)' if i == 0 else '')
            ax.set_title(titles[j], fontsize=16, fontweight='normal')
            for k in range(1, len(window_limits) - 1):
                ax.axvline(x=window_limits[k], color='gray', ls=':', alpha=0.3)

        ax_s = axs[1, 1]
        ax_s.plot(t_win, np.sum(res_t, axis=1), 'g-')
        ax_s.plot(t_win, np.sum(res_d, axis=1), 'r--')
        ax_s.axhline(y=1.0, color='black', ls=':', alpha=0.5)
        ax_s.set_title(titles[3], fontsize=16, fontweight='normal')
        ax_s.set_ylim(0.98, 1.02)
        ax_s.set_xlabel('Time (days)')
        for k in range(1, len(window_limits) - 1):
            ax_s.axvline(x=window_limits[k], color='gray', ls=':', alpha=0.3)

    fig.legend(
        [l_real, l_match, l_drift],
        ['Real', r'Match ($\beta=0.3$)', r'Drift ($\beta=0.5$)'],
        loc='upper center', bbox_to_anchor=(0.5, 1.06),
        ncol=3, frameon=True, facecolor='white', fontsize=16,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig
