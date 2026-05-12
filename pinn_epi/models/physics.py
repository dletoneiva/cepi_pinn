import abc
import torch


class CompartmentalModel(abc.ABC):
    """Abstract base class for compartmental epidemiological models.
    
    Subclasses must implement:
        - get_derivatives: the ODE right-hand side
        - compartment_names: list of compartment labels (e.g. ['S', 'I', 'R'])
    """

    @property
    @abc.abstractmethod
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names, e.g. ['S', 'I', 'R']."""
        pass

    @property
    def should_conserve(self) -> bool:
        """Return whether this model should conserve total population.
        
        Defaults to True. Override in subclasses to return False for models
        that don't conserve total population (e.g., models with births/deaths).
        """
        return True

    @abc.abstractmethod
    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives of all compartments.

        Args:
            t: scalar time tensor
            u: state tensor of shape (..., n_compartments)
            params: dict of model parameters

        Returns:
            Tensor of same shape as u containing du/dt
        """
        pass


class SIRModel(CompartmentalModel):
    """SIR (Susceptible-Infected-Recovered) epidemiological model."""

    @property
    def compartment_names(self) -> list[str]:
        return ['S', 'I', 'R']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        beta: float = params['beta']
        gamma: float = params['gamma']
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        dR_dt = gamma * I
        return torch.stack([dS_dt, dI_dt, dR_dt], dim=-1)


class SEIRModel(CompartmentalModel):
    """SEIR (Susceptible-Exposed-Infected-Recovered) epidemiological model."""

    @property
    def compartment_names(self) -> list[str]:
        return ['S', 'E', 'I', 'R']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        S, E, I, R = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        beta: float = params['beta']
        sigma: float = params['sigma']  # rate of progression from E to I
        gamma: float = params['gamma']
        dS_dt = -beta * S * I
        dE_dt = beta * S * I - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I
        return torch.stack([dS_dt, dE_dt, dI_dt, dR_dt], dim=-1)


class SIModel(CompartmentalModel):
    """SI (Susceptible-Infected) epidemiological model (no recovery)."""

    @property
    def compartment_names(self) -> list[str]:
        return ['S', 'I']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        S, I = u[..., 0], u[..., 1]
        beta: float = params['beta']
        dS_dt = -beta * S * I
        dI_dt = beta * S * I
        return torch.stack([dS_dt, dI_dt], dim=-1)


class SISModel(CompartmentalModel):
    """SIS (Susceptible-Infected-Susceptible) epidemiological model."""

    @property
    def compartment_names(self) -> list[str]:
        return ['S', 'I']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        S, I = u[..., 0], u[..., 1]
        beta: float = params['beta']
        gamma: float = params['gamma']
        dS_dt = -beta * S * I + gamma * I
        dI_dt = beta * S * I - gamma * I
        return torch.stack([dS_dt, dI_dt], dim=-1)


class SIRVModel(CompartmentalModel):
    """SIRV (Susceptible-Infected-Recovered-Vaccinated) epidemiological model."""

    @property
    def compartment_names(self) -> list[str]:
        return ['S', 'I', 'R', 'V']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        S, I, R, V = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        beta: float = params['beta']
        gamma: float = params['gamma']
        nu: float = params['nu']  # vaccination rate
        dS_dt = -beta * S * I - nu * S
        dI_dt = beta * S * I - gamma * I
        dR_dt = gamma * I
        dV_dt = nu * S
        return torch.stack([dS_dt, dI_dt, dR_dt, dV_dt], dim=-1)


class SIRDModel(CompartmentalModel):
    """SIRD (Susceptible-Infected-Recovered-Dead) epidemiological model."""

    @property
    def compartment_names(self) -> list[str]:
        return ['S', 'I', 'R', 'D']

    @property
    def should_conserve(self) -> bool:
        return False  # Total population is not conserved due to deaths

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        S, I, R, D = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        beta: float = params['beta']
        gamma: float = params['gamma']  # recovery rate
        mu: float = params['mu']        # death rate
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - (gamma + mu) * I
        dR_dt = gamma * I
        dD_dt = mu * I
        return torch.stack([dS_dt, dI_dt, dR_dt, dD_dt], dim=-1)
