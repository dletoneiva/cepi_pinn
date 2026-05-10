"""Sanity check for the physics engine via plotting.

Run this script directly to visually verify that all compartmental models
solve and plot correctly.  No test framework required — just run:

    python test_physics.py

Each model will:
  1. Solve its ODE system via scipy (wrapped around the torch model).
  2. Plot the trajectories with a conservation-error annotation.
  3. Return the trajectories dict so values can be inspected in the terminal.
"""

import numpy as np

from pinn_epi.models.physics import SIRModel, SEIRModel, SIModel
from pinn_epi.analysis.plotting import plot_compartmental_solution, apply_nature_style

apply_nature_style()


# ---------------------------------------------------------------------------
# SIR
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
    )

    # Programmatic checks
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


# ---------------------------------------------------------------------------
# SEIR
# ---------------------------------------------------------------------------
def sanity_check_seir() -> None:
    print("\n--- SEIR Model ---")
    model = SEIRModel()
    t_span = [0, 60]
    y0 = [0.99, 0.00, 0.01, 0.00]   # S, E, I, R
    params = {'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1}

    fig, ax, trajectories = plot_compartmental_solution(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        title='SEIR Model — Sanity Check',
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


# ---------------------------------------------------------------------------
# SI
# ---------------------------------------------------------------------------
def sanity_check_si() -> None:
    print("\n--- SI Model ---")
    model = SIModel()
    t_span = [0, 60]
    y0 = [0.99, 0.01]   # S, I
    params = {'beta': 0.3}

    fig, ax, trajectories = plot_compartmental_solution(
        model=model,
        t_span=t_span,
        y0=y0,
        params=params,
        title='SI Model — Sanity Check',
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
    sanity_check_sir()
    sanity_check_seir()
    sanity_check_si()
    print("\nAll sanity checks passed.")
