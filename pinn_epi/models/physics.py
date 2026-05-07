"""Compartmental model base class and SIR implementation."""
import abc
import torch


class CompartmentalModel(abc.ABC):
    """Abstract base class for compartmental epidemiological models."""
    
    @abc.abstractmethod
    def get_derivatives(self, t: torch.Tensor, u: torch.Tensor, params: dict) -> torch.Tensor:
        """Calculate time derivatives of compartment variables.
        
        Args:
            t: Time tensor
            u: State variables tensor with shape (..., n_compartments)
            params: Model parameters dictionary
            
        Returns:
            Time derivatives tensor with same shape as u
        """
        pass


class SIRModel(CompartmentalModel):
    """SIR (Susceptible-Infected-Recovered) epidemiological model."""
    
    def get_derivatives(self, t: torch.Tensor, u: torch.Tensor, params: dict) -> torch.Tensor:
        """Calculate SIR model derivatives.
        
        Args:
            t: Time tensor
            u: State variables tensor with shape (..., 3) where last dimension is S, I, R
            params: Model parameters dictionary containing 'beta' and 'gamma'
            
        Returns:
            Time derivatives tensor with shape (..., 3) for dS/dt, dI/dt, dR/dt
        """
        # Extract compartments
        S = u[..., 0]
        I = u[..., 1]
        R = u[..., 2]
        
        # Extract parameters
        beta = params['beta']
        gamma = params['gamma']
        
        # Calculate derivatives using SIR equations
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        dR_dt = gamma * I
        
        # Stack and return derivatives
        return torch.stack([dS_dt, dI_dt, dR_dt], dim=-1)
