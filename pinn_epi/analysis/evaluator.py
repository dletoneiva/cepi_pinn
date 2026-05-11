"""Mathematical analysis tools for PINN models."""

from typing import Optional
import numpy as np
import torch
from torchdiffeq import odeint
from pinn_epi.models.physics import CompartmentalModel


def solve_compartmental_model(
    model: CompartmentalModel,
    t_span: list[float],
    y0: list[float],
    params: dict,
    t_eval: Optional[np.ndarray] = None,
    method: str = 'rk4',
    **odeint_kwargs,
) -> dict[str, np.ndarray]:
    """Solve the ODE system defined by any CompartmentalModel using torchdiffeq.

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
        method: Integration method for torchdiffeq.odeint (e.g., 'rk4', 'dopri5').
        **odeint_kwargs: Extra keyword arguments forwarded to torchdiffeq.odeint.

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

    # Convert initial conditions and time points to torch tensors
    y0_tensor = torch.tensor(y0, dtype=torch.float32)
    t_eval_tensor = torch.tensor(t_eval, dtype=torch.float32)

    # --- wrap model.get_derivatives for torchdiffeq ---
    def ode_rhs(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return model.get_derivatives(t, y, params)

    # Solve the ODE using torchdiffeq
    sol = odeint(ode_rhs, y0_tensor, t_eval_tensor, method=method, **odeint_kwargs)

    # Convert solution back to numpy arrays
    sol_np = sol.detach().numpy()

    # --- build trajectories dict ---
    trajectories: dict[str, np.ndarray] = {
        name: sol_np[:, i] for i, name in enumerate(compartment_names)
    }

    return trajectories
