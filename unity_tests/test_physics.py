"""Sanity check for the physics engine via plotting.

Run this script directly to visually verify that all compartmental models
solve and plot correctly.  No test framework required — just run:

    python test_physics.py

Each model will:
  1. Solve its ODE system via scipy (wrapped around the torch model).
  2. Plot the trajectories with model information.
  3. Save the figure to the figures/ directory.
"""

import os
import numpy as np
import datetime

from pinn_epi.models.physics import SIRModel, SEIRModel, SIModel
from pinn_epi.analysis.plotting import (
    apply_nature_style,
    plot_compartmental_solution,
)
from pinn_epi.analysis.evaluator import solve_compartmental_model

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

apply_nature_style()


def fig_path(name: str) -> str:
    """Return the full save path for a figure inside FIGURES_DIR."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(FIGURES_DIR, f"{timestamp}_{name.replace('.', '_')}")


# ---------------------------------------------------------------------------
# Compartmental model sanity checks
# ---------------------------------------------------------------------------

def sanity_check_sir() -> None:
    print("\n--- SIR Model ---")
    model = SIRModel()
    t_span = [0, 60]
    y0 = [0.8, 0.2, 0.00]
    params = {'beta': 0.4, 'gamma': 0.1}
    t_eval = np.linspace(t_span[0], t_span[1], 300)

    trajectories = solve_compartmental_model(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        t_eval=t_eval,
    )

    fig, ax = plot_compartmental_solution(
        t=t_eval,
        trajectories=trajectories,
        title='SIR Model',
        save_path=fig_path('sir_model_beta04_gamma01.png'),
        model_params=params,
        initial_conditions=y0,
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
    y0 = [0.8, 0.19, 0.01, 0.00]  # S, E, I, R
    params = {'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1}
    t_eval = np.linspace(t_span[0], t_span[1], 300)

    trajectories = solve_compartmental_model(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        t_eval=t_eval,
    )

    fig, ax = plot_compartmental_solution(
        t=t_eval,
        trajectories=trajectories,
        title='SEIR Model',
        save_path=fig_path('seir_model_beta04_sigma02_gamma01.png'),
        model_params=params,
        initial_conditions=y0,
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
    y0 = [0.8, 0.2]  # S, I
    params = {'beta': 0.3}
    t_eval = np.linspace(t_span[0], t_span[1], 300)

    trajectories = solve_compartmental_model(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        t_eval=t_eval,
    )

    fig, ax = plot_compartmental_solution(
        t=t_eval,
        trajectories=trajectories,
        title='SI Model',
        save_path=fig_path('si_model_beta03.png'),
        model_params=params,
        initial_conditions=y0,
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- compartmental model sanity checks ---
    sanity_check_sir()
    sanity_check_seir()
    sanity_check_si()

    print(f"\nAll checks passed.  Figures saved to '{FIGURES_DIR}/'.")
