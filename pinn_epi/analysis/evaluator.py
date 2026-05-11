"""Mathematical analysis tools for PINN models."""

import numpy as np
from scipy.integrate import solve_ivp
import torch
from pinn_epi.models.physics import CompartmentalModel


def solve_compartmental_model(
    model: CompartmentalModel,
    t_span: list[float],
    y0: list[float],
    params: dict,
    t_eval: Optional[np.ndarray] = None,
    **solve_ivp_kwargs,
) -> dict[str, np.ndarray]:
    """Solve the ODE system defined by any CompartmentalModel.

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
        **solve_ivp_kwargs: Extra keyword arguments forwarded to
            scipy.integrate.solve_ivp (e.g. method='RK45', rtol=1e-8).

    Returns:
        A dictionary mapping each compartment name to its solved numpy array.
    """
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

    return trajectories
