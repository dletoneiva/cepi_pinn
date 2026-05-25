"""Compartmental epidemiological models for PINN physics constraints.

This module implements various compartmental models commonly used in epidemiology.
Each model defines the system of ordinary differential equations (ODEs) that
govern the dynamics of disease spread through a population.

The models are designed to work with Physics-Informed Neural Networks (PINNs)
by providing the right-hand side of the ODE system for computing physics-based
loss terms during training.
"""

import abc
import torch


class CompartmentalModel(abc.ABC):
    """Abstract base class for compartmental epidemiological models.
    
    This class defines the interface for all compartmental models. Subclasses
    must implement the get_derivatives method which computes the time derivatives
    of all compartments according to the model's ODE system.
    
    The parameters dictionary is used to pass model-specific parameters like
    transmission rates, recovery rates, etc. This design allows for flexible
    parameter handling during PINN training where some parameters may be
    learnable.
    
    Attributes:
        parameters: Dictionary of model parameters
    """

    def __init__(self, parameters: dict = None):
        """Initialize the compartmental model with parameters.
        
        Args:
            parameters: Dictionary of model parameters
        """
        self.parameters = parameters or {}

    @property
    @abc.abstractmethod
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names, e.g. ['S', 'I', 'R'].
        
        This property defines the order of compartments in the state vector
        and the corresponding derivatives returned by get_derivatives.
        
        Returns:
            List of compartment names in order
        """
        pass

    @abc.abstractmethod
    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives of all compartments.
        
        This method implements the right-hand side of the ODE system:
        du/dt = f(t, u, params)
        
        For use in PINNs, this function is evaluated at collocation points
        to compute the physics loss: ||du/dt - f(t, u, params)||^2
        
        Args:
            t: scalar time tensor
            u: state tensor of shape (..., n_compartments)
            params: dict of model parameters
            
        Returns:
            Tensor of same shape as u containing du/dt
        """
        pass

    def get_observables(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> dict[str, torch.Tensor]:
        """Get observable quantities from the model state.
        
        Default implementation returns the base compartments as observables.
        Subclasses can override this to provide derived observables like
        daily new cases, reproduction numbers, etc.
        
        Args:
            t: scalar time tensor
            u: state tensor of shape (..., n_compartments)
            params: dict of model parameters
            
        Returns:
            Dictionary mapping observable names to their tensor values
        """
        return {name: u[..., i] for i, name in enumerate(self.compartment_names)}


class SIRModel(CompartmentalModel):
    """SIR (Susceptible-Infected-Recovered) epidemiological model.
    
    The SIR model divides the population into three compartments:
    - S: Susceptible individuals who can catch the disease
    - I: Infected individuals who can spread the disease
    - R: Recovered individuals who are immune
    
    The dynamics are governed by:
    dS/dt = -βSI
    dI/dt = βSI - γI
    dR/dt = γI
    
    Where:
    - β (beta) is the transmission rate
    - γ (gamma) is the recovery rate
    
    This model assumes lifelong immunity after recovery and a constant population.
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SIR model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'gamma': recovery rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'I', 'R']
        """
        return ['S', 'I', 'R']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SIR model.
        
        Implements the SIR ODE system:
        dS/dt = -βSI
        dI/dt = βSI - γI
        dR/dt = γI
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I, R values
            params: dict with 'beta' and 'gamma' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        beta: float = params['beta']
        gamma: float = params['gamma']
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        dR_dt = gamma * I
        return torch.stack([dS_dt, dI_dt, dR_dt], dim=-1)

    def get_observables(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> dict[str, torch.Tensor]:
        """Get observable quantities including daily new cases.
        
        Returns base compartments plus derived observables.
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I, R values
            params: dict with model parameters
            
        Returns:
            Dictionary with compartment values and derived observables
        """
        # Get base compartments from parent implementation
        observables = super().get_observables(t, u, params)
        
        # Add derived observables
        S, I = u[..., 0], u[..., 1]
        beta: float = params['beta']
        # Daily new cases = beta * S * I (same as dI_dt + gamma * I)
        observables['daily_new_cases'] = beta * S * I
        
        return observables


class SEIRModel(CompartmentalModel):
    """SEIR (Susceptible-Exposed-Infected-Recovered) epidemiological model.
    
    The SEIR model extends SIR by adding an Exposed compartment for individuals
    who are infected but not yet infectious:
    - S: Susceptible individuals
    - E: Exposed individuals (infected but not infectious)
    - I: Infectious individuals
    - R: Recovered individuals
    
    The dynamics are governed by:
    dS/dt = -βSI
    dE/dt = βSI - σE
    dI/dt = σE - γI
    dR/dt = γI
    
    Where:
    - β (beta) is the transmission rate
    - σ (sigma) is the rate of progression from exposed to infectious
    - γ (gamma) is the recovery rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SEIR model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'sigma': progression rate from exposed to infected
                - 'gamma': recovery rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'E', 'I', 'R']
        """
        return ['S', 'E', 'I', 'R']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SEIR model.
        
        Implements the SEIR ODE system:
        dS/dt = -βSI
        dE/dt = βSI - σE
        dI/dt = σE - γI
        dR/dt = γI
        
        Args:
            t: scalar time tensor
            u: state tensor with S, E, I, R values
            params: dict with 'beta', 'sigma', and 'gamma' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
        """
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
    """SI (Susceptible-Infected) epidemiological model (no recovery).
    
    The simplest epidemiological model with only two compartments:
    - S: Susceptible individuals
    - I: Infected individuals (no recovery)
    
    The dynamics are governed by:
    dS/dt = -βSI
    dI/dt = βSI
    
    This model assumes no recovery, so the entire population eventually
    becomes infected. It's useful for modeling diseases with very long
    infectious periods.
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SI model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'I']
        """
        return ['S', 'I']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SI model.
        
        Implements the SI ODE system:
        dS/dt = -βSI
        dI/dt = βSI
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I values
            params: dict with 'beta' parameter
            
        Returns:
            Tensor with derivatives [dS/dt, dI/dt]
        """
        S, I = u[..., 0], u[..., 1]
        beta: float = params['beta']
        dS_dt = -beta * S * I
        dI_dt = beta * S * I
        return torch.stack([dS_dt, dI_dt], dim=-1)


class SISModel(CompartmentalModel):
    """SIS (Susceptible-Infected-Susceptible) epidemiological model.
    
    This model allows for reinfection by moving recovered individuals
    back to the susceptible compartment:
    - S: Susceptible individuals
    - I: Infected individuals
    
    The dynamics are governed by:
    dS/dt = -βSI + γI
    dI/dt = βSI - γI
    
    Where:
    - β (beta) is the transmission rate
    - γ (gamma) is the recovery rate (becoming susceptible again)
    
    This model approaches an endemic equilibrium where a constant
    proportion of the population is infected.
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SIS model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'gamma': recovery rate (returning to susceptible)
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'I']
        """
        return ['S', 'I']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SIS model.
        
        Implements the SIS ODE system:
        dS/dt = -βSI + γI
        dI/dt = βSI - γI
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I values
            params: dict with 'beta' and 'gamma' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dI/dt]
        """
        S, I = u[..., 0], u[..., 1]
        beta: float = params['beta']
        gamma: float = params['gamma']
        dS_dt = -beta * S * I + gamma * I
        dI_dt = beta * S * I - gamma * I
        return torch.stack([dS_dt, dI_dt], dim=-1)


class SIRVModel(CompartmentalModel):
    """SIRV (Susceptible-Infected-Recovered-Vaccinated) epidemiological model.
    
    This model extends SIR by adding a Vaccinated compartment for individuals
    who have been vaccinated:
    - S: Susceptible individuals
    - I: Infected individuals
    - R: Recovered individuals
    - V: Vaccinated individuals
    
    The dynamics are governed by:
    dS/dt = -βSI - νS
    dI/dt = βSI - γI
    dR/dt = γI
    dV/dt = νS
    
    Where:
    - β (beta) is the transmission rate
    - γ (gamma) is the recovery rate
    - ν (nu) is the vaccination rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SIRV model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'gamma': recovery rate
                - 'nu': vaccination rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'I', 'R', 'V']
        """
        return ['S', 'I', 'R', 'V']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SIRV model.
        
        Implements the SIRV ODE system:
        dS/dt = -βSI - νS
        dI/dt = βSI - γI
        dR/dt = γI
        dV/dt = νS
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I, R, V values
            params: dict with 'beta', 'gamma', and 'nu' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dI/dt, dR/dt, dV/dt]
        """
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
    """SIRD (Susceptible-Infected-Recovered-Dead) epidemiological model.
    
    This model extends SIR by adding a Dead compartment for individuals
    who die from the disease:
    - S: Susceptible individuals
    - I: Infected individuals
    - R: Recovered individuals
    - D: Dead individuals
    
    The dynamics are governed by:
    dS/dt = -βSI
    dI/dt = βSI - (γ + μ)I
    dR/dt = γI
    dD/dt = μI
    
    Where:
    - β (beta) is the transmission rate
    - γ (gamma) is the recovery rate
    - μ (mu) is the death rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SIRD model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'gamma': recovery rate
                - 'mu': death rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'I', 'R', 'D']
        """
        return ['S', 'I', 'R', 'D']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SIRD model.
        
        Implements the SIRD ODE system:
        dS/dt = -βSI
        dI/dt = βSI - (γ + μ)I
        dR/dt = γI
        dD/dt = μI
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I, R, D values
            params: dict with 'beta', 'gamma', and 'mu' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dI/dt, dR/dt, dD/dt]
        """
        S, I, R, D = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        beta: float = params['beta']
        gamma: float = params['gamma']  # recovery rate
        mu: float = params['mu']        # death rate
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - (gamma + mu) * I
        dR_dt = gamma * I
        dD_dt = mu * I
        return torch.stack([dS_dt, dI_dt, dR_dt, dD_dt], dim=-1)


class SIRDVModel(CompartmentalModel):
    """SIRDV (Susceptible-Infected-Recovered-Dead-Vaccinated) epidemiological model.
    
    This model combines SIRD and SIRV by including both death and vaccination:
    - S: Susceptible individuals
    - I: Infected individuals
    - R: Recovered individuals
    - D: Dead individuals
    - V: Vaccinated individuals
    
    The dynamics are governed by:
    dS/dt = -βSI - νS
    dI/dt = βSI - (γ + μ)I
    dR/dt = γI
    dD/dt = μI
    dV/dt = νS
    
    Where:
    - β (beta) is the transmission rate
    - γ (gamma) is the recovery rate
    - μ (mu) is the death rate
    - ν (nu) is the vaccination rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SIRDV model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'gamma': recovery rate
                - 'mu': death rate
                - 'nu': vaccination rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'I', 'R', 'D', 'V']
        """
        return ['S', 'I', 'R', 'D', 'V']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SIRDV model.
        
        Implements the SIRDV ODE system:
        dS/dt = -βSI - νS
        dI/dt = βSI - (γ + μ)I
        dR/dt = γI
        dD/dt = μI
        dV/dt = νS
        
        Args:
            t: scalar time tensor
            u: state tensor with S, I, R, D, V values
            params: dict with 'beta', 'gamma', 'mu', and 'nu' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dI/dt, dR/dt, dD/dt, dV/dt]
        """
        S, I, R, D, V = u[..., 0], u[..., 1], u[..., 2], u[..., 3], u[..., 4]
        beta: float = params['beta']
        gamma: float = params['gamma']  # recovery rate
        mu: float = params['mu']        # death rate
        nu: float = params['nu']        # vaccination rate
        dS_dt = -beta * S * I - nu * S
        dI_dt = beta * S * I - (gamma + mu) * I
        dR_dt = gamma * I
        dD_dt = mu * I
        dV_dt = nu * S
        return torch.stack([dS_dt, dI_dt, dR_dt, dD_dt, dV_dt], dim=-1)


class SEIRDModel(CompartmentalModel):
    """SEIRD (Susceptible-Exposed-Infected-Recovered-Dead) epidemiological model.
    
    This model combines SEIR with disease-induced mortality:
    - S: Susceptible individuals
    - E: Exposed individuals (infected but not infectious)
    - I: Infectious individuals
    - R: Recovered individuals
    - D: Dead individuals
    
    The dynamics are governed by:
    dS/dt = -βSI
    dE/dt = βSI - σE
    dI/dt = σE - (γ + μ)I
    dR/dt = γI
    dD/dt = μI
    
    Where:
    - β (beta) is the transmission rate
    - σ (sigma) is the rate of progression from exposed to infectious
    - γ (gamma) is the recovery rate
    - μ (mu) is the death rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SEIRD model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'sigma': progression rate from exposed to infected
                - 'gamma': recovery rate
                - 'mu': death rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'E', 'I', 'R', 'D']
        """
        return ['S', 'E', 'I', 'R', 'D']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SEIRD model.
        
        Implements the SEIRD ODE system:
        dS/dt = -βSI
        dE/dt = βSI - σE
        dI/dt = σE - (γ + μ)I
        dR/dt = γI
        dD/dt = μI
        
        Args:
            t: scalar time tensor
            u: state tensor with S, E, I, R, D values
            params: dict with 'beta', 'sigma', 'gamma', and 'mu' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dD/dt]
        """
        S, E, I, R, D = u[..., 0], u[..., 1], u[..., 2], u[..., 3], u[..., 4]
        beta: float = params['beta']
        sigma: float = params['sigma']  # rate of progression from E to I
        gamma: float = params['gamma']  # recovery rate
        mu: float = params['mu']        # death rate
        dS_dt = -beta * S * I
        dE_dt = beta * S * I - sigma * E
        dI_dt = sigma * E - (gamma + mu) * I
        dR_dt = gamma * I
        dD_dt = mu * I
        return torch.stack([dS_dt, dE_dt, dI_dt, dR_dt, dD_dt], dim=-1)


class SEIRVModel(CompartmentalModel):
    """SEIRV (Susceptible-Exposed-Infected-Recovered-Vaccinated) epidemiological model.
    
    This model combines SEIR with vaccination:
    - S: Susceptible individuals
    - E: Exposed individuals (infected but not infectious)
    - I: Infectious individuals
    - R: Recovered individuals
    - V: Vaccinated individuals
    
    The dynamics are governed by:
    dS/dt = -βSI - νS
    dE/dt = βSI - σE
    dI/dt = σE - γI
    dR/dt = γI
    dV/dt = νS
    
    Where:
    - β (beta) is the transmission rate
    - σ (sigma) is the rate of progression from exposed to infectious
    - γ (gamma) is the recovery rate
    - ν (nu) is the vaccination rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SEIRV model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'sigma': progression rate from exposed to infected
                - 'gamma': recovery rate
                - 'nu': vaccination rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'E', 'I', 'R', 'V']
        """
        return ['S', 'E', 'I', 'R', 'V']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SEIRV model.
        
        Implements the SEIRV ODE system:
        dS/dt = -βSI - νS
        dE/dt = βSI - σE
        dI/dt = σE - γI
        dR/dt = γI
        dV/dt = νS
        
        Args:
            t: scalar time tensor
            u: state tensor with S, E, I, R, V values
            params: dict with 'beta', 'sigma', 'gamma', and 'nu' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dV/dt]
        """
        S, E, I, R, V = u[..., 0], u[..., 1], u[..., 2], u[..., 3], u[..., 4]
        beta: float = params['beta']
        sigma: float = params['sigma']  # rate of progression from E to I
        gamma: float = params['gamma']  # recovery rate
        nu: float = params['nu']        # vaccination rate
        dS_dt = -beta * S * I - nu * S
        dE_dt = beta * S * I - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I
        dV_dt = nu * S
        return torch.stack([dS_dt, dE_dt, dI_dt, dR_dt, dV_dt], dim=-1)


class SEIRDVModel(CompartmentalModel):
    """SEIRDV (Susceptible-Exposed-Infected-Recovered-Dead-Vaccinated) epidemiological model.
    
    The most comprehensive model combining SEIR with both mortality and vaccination:
    - S: Susceptible individuals
    - E: Exposed individuals (infected but not infectious)
    - I: Infectious individuals
    - R: Recovered individuals
    - D: Dead individuals
    - V: Vaccinated individuals
    
    The dynamics are governed by:
    dS/dt = -βSI - νS
    dE/dt = βSI - σE
    dI/dt = σE - (γ + μ)I
    dR/dt = γI
    dD/dt = μI
    dV/dt = νS
    
    Where:
    - β (beta) is the transmission rate
    - σ (sigma) is the rate of progression from exposed to infectious
    - γ (gamma) is the recovery rate
    - μ (mu) is the death rate
    - ν (nu) is the vaccination rate
    """

    def __init__(self, parameters: dict = None):
        """Initialize the SEIRDV model with parameters.
        
        Args:
            parameters: Dictionary of model parameters with keys:
                - 'beta': transmission rate
                - 'sigma': progression rate from exposed to infected
                - 'gamma': recovery rate
                - 'mu': death rate
                - 'nu': vaccination rate
        """
        super().__init__(parameters=parameters)

    @property
    def compartment_names(self) -> list[str]:
        """Return ordered list of compartment names.
        
        Returns:
            ['S', 'E', 'I', 'R', 'D', 'V']
        """
        return ['S', 'E', 'I', 'R', 'D', 'V']

    def get_derivatives(
        self, t: torch.Tensor, u: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Compute time derivatives for the SEIRDV model.
        
        Implements the SEIRDV ODE system:
        dS/dt = -βSI - νS
        dE/dt = βSI - σE
        dI/dt = σE - (γ + μ)I
        dR/dt = γI
        dD/dt = μI
        dV/dt = νS
        
        Args:
            t: scalar time tensor
            u: state tensor with S, E, I, R, D, V values
            params: dict with 'beta', 'sigma', 'gamma', 'mu', and 'nu' parameters
            
        Returns:
            Tensor with derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dD/dt, dV/dt]
        """
        S, E, I, R, D, V = u[..., 0], u[..., 1], u[..., 2], u[..., 3], u[..., 4], u[..., 5]
        beta: float = params['beta']
        sigma: float = params['sigma']  # rate of progression from E to I
        gamma: float = params['gamma']  # recovery rate
        mu: float = params['mu']        # death rate
        nu: float = params['nu']        # vaccination rate
        dS_dt = -beta * S * I - nu * S
        dE_dt = beta * S * I - sigma * E
        dI_dt = sigma * E - (gamma + mu) * I
        dR_dt = gamma * I
        dD_dt = mu * I
        dV_dt = nu * S
        return torch.stack([dS_dt, dE_dt, dI_dt, dR_dt, dD_dt, dV_dt], dim=-1)
