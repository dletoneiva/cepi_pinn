"""Sanity check for the physics engine via plotting.

Run this script directly to visually verify that all compartmental models
solve and plot correctly, and to exercise every plotting function in
pinn_epi/analysis/plotting.py.  No test framework required — just run:

    python test_physics.py

Each model will:
  1. Solve its ODE system via scipy (wrapped around the torch model).
  2. Plot the trajectories with a conservation-error annotation.
  3. Save the figure to the figures/ directory.
  4. Return the trajectories dict so values can be inspected in the terminal.

All other plotting helpers (figure 1, Hessian, bifurcation, windowed) are
also called and their outputs saved to figures/.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

from pinn_epi.models.physics import SIRModel, SEIRModel, SIModel
from pinn_epi.analysis.plotting import (
    apply_nature_style,
    plot_compartmental_solution,
    plot_figure_1_corrected,
    plot_hessian_scalarization_r,
    plot_sir_bifurcation,
    plot_windowed_results,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

apply_nature_style()


def fig_path(name: str) -> str:
    """Return the full save path for a figure inside FIGURES_DIR."""
    return os.path.join(FIGURES_DIR, name)


# ---------------------------------------------------------------------------
# Compartmental model sanity checks
# ---------------------------------------------------------------------------

def sanity_check_sir() -> None:
    print("\n--- SIR Model ---")
    model = SIRModel()
    t_span = [0, 60]
    y0 = [0.99, 0.01, 0.00]
    params = {'beta': 0.4, 'gamma': 0.1}

    fig, ax, trajectories = plot_compartmental_solution(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        title='SIR Model — Sanity Check',
        save_path=fig_path('sanity_sir.png'),
    )

    S, I, R = trajectories['S'], trajectories['I'], trajectories['R']
    total = S + I + R
    max_conservation_error = np.max(np.abs(total - total[0]))

    print(f"  Compartments : {model.compartment_names}")
    print(f"  S(0)={S[0]:.4f}  I(0)={I[0]:.4f}  R(0)={R[0]:.4f}")
    print(f"  S(T)={S[-1]:.4f}  I(T)={I[-1]:.4f}  R(T)={R[-1]:.4f}")
    print(f"  Max conservation error : {max_conservation_error:.2e}")
    assert max_conservation_error < 1e-4, "Conservation violated in SIR!"
    assert S[-1] < S[0], "S should decrease over time"
    assert R[-1] > R[0], "R should increase over time"
    print("  [PASS] SIR sanity check passed.")


def sanity_check_seir() -> None:
    print("\n--- SEIR Model ---")
    model = SEIRModel()
    t_span = [0, 60]
    y0 = [0.99, 0.00, 0.01, 0.00]  # S, E, I, R
    params = {'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1}

    fig, ax, trajectories = plot_compartmental_solution(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        title='SEIR Model — Sanity Check',
        save_path=fig_path('sanity_seir.png'),
    )

    S, E, I, R = (trajectories[k] for k in ['S', 'E', 'I', 'R'])
    total = S + E + I + R
    max_conservation_error = np.max(np.abs(total - total[0]))

    print(f"  Compartments : {model.compartment_names}")
    print(f"  S(0)={S[0]:.4f}  E(0)={E[0]:.4f}  I(0)={I[0]:.4f}  R(0)={R[0]:.4f}")
    print(f"  S(T)={S[-1]:.4f}  E(T)={E[-1]:.4f}  I(T)={I[-1]:.4f}  R(T)={R[-1]:.4f}")
    print(f"  Max conservation error : {max_conservation_error:.2e}")
    assert max_conservation_error < 1e-4, "Conservation violated in SEIR!"
    assert S[-1] < S[0], "S should decrease over time"
    assert R[-1] > R[0], "R should increase over time"
    print("  [PASS] SEIR sanity check passed.")


def sanity_check_si() -> None:
    print("\n--- SI Model ---")
    model = SIModel()
    t_span = [0, 60]
    y0 = [0.99, 0.01]  # S, I
    params = {'beta': 0.3}

    fig, ax, trajectories = plot_compartmental_solution(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        title='SI Model — Sanity Check',
        save_path=fig_path('sanity_si.png'),
    )

    S, I = trajectories['S'], trajectories['I']
    total = S + I
    max_conservation_error = np.max(np.abs(total - total[0]))

    print(f"  Compartments : {model.compartment_names}")
    print(f"  S(0)={S[0]:.4f}  I(0)={I[0]:.4f}")
    print(f"  S(T)={S[-1]:.4f}  I(T)={I[-1]:.4f}")
    print(f"  Max conservation error : {max_conservation_error:.2e}")
    assert max_conservation_error < 1e-4, "Conservation violated in SI!"
    assert S[-1] < S[0], "S should decrease over time"
    assert I[-1] > I[0], "I should increase over time (no recovery)"
    print("  [PASS] SI sanity check passed.")


# ---------------------------------------------------------------------------
# Plotting function checks
# ---------------------------------------------------------------------------

def check_plot_figure_1() -> None:
    print("\n--- plot_figure_1_corrected ---")
    t_max = 30
    t_eval = np.linspace(0, t_max, 150)
    y0 = [0.99, 0.01, 0.00]
    beta_true = 0.4
    gamma_true = 0.1

    plot_figure_1_corrected(
        t_max=t_max,
        t_eval=t_eval,
        y0=y0,
        beta_true=beta_true,
        gamma_true=gamma_true,
        save_path=fig_path('figure_1_identifiability.png'),
    )
    print("  [PASS] plot_figure_1_corrected completed.")


def check_plot_hessian() -> None:
    print("\n--- plot_hessian_scalarization_r ---")
    t_max = 30
    t_eval = np.linspace(0, t_max, 150)
    y0 = [0.99, 0.01, 0.00]
    gamma_true = 0.1

    plot_hessian_scalarization_r(
        t_max=t_max,
        t_eval=t_eval,
        y0=y0,
        gamma_true=gamma_true,
        lambda_phys=1.0,
        save_path=fig_path('hessian_scalarization.png'),
    )
    print("  [PASS] plot_hessian_scalarization_r completed.")


def check_plot_bifurcation() -> None:
    print("\n--- plot_sir_bifurcation ---")
    plot_sir_bifurcation(
        gamma=0.4,
        save_path=fig_path('sir_bifurcation.png'),
    )
    print("  [PASS] plot_sir_bifurcation completed.")


def check_plot_windowed_results() -> None:
    """Build minimal synthetic windowed results to exercise plot_windowed_results."""
    print("\n--- plot_windowed_results ---")

    t_max = 30
    t_eval = np.linspace(0, t_max, 300)
    beta_true, gamma_true = 0.3, 0.1
    y0 = [0.99, 0.01, 0.0]

    def _sir(t, y, b, g):
        S, I, R = y
        return [-b * S * I, b * S * I - g * I, g * I]

    sol_ref = solve_ivp(_sir, [0, t_max], y0, args=(beta_true, gamma_true), t_eval=t_eval)

    # Two windows: [0, 15] and [15, 30]
    window_limits = np.array([0, 15, 30], dtype=float)

    results_t = []
    results_d = []
    for i in range(len(window_limits) - 1):
        start, end = window_limits[i], window_limits[i + 1]
        mask = (t_eval >= start) & (t_eval <= end)
        # Use the ground truth ODE solution as a stand-in for both
        # 'match' and 'drift' predictions (no PINN training needed here)
        chunk = sol_ref.y[:, mask].T          # shape (n_points, 3)
        results_t.append(chunk)
        results_d.append(chunk * 0.98 + 0.01) # slight perturbation for 'drift'

    plot_windowed_results(
        window_limits=window_limits,
        results_t=results_t,
        results_d=results_d,
        sol_ref=sol_ref,
        t_eval_ref=t_eval,
        save_path=fig_path('windowed_results.png'),
    )
    print("  [PASS] plot_windowed_results completed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- compartmental model sanity checks ---
    sanity_check_sir()
    sanity_check_seir()
    sanity_check_si()

    # --- plotting function checks ---
    check_plot_figure_1()
    check_plot_hessian()
    check_plot_bifurcation()
    check_plot_windowed_results()

    print(f"\nAll checks passed.  Figures saved to '{FIGURES_DIR}/'.")
