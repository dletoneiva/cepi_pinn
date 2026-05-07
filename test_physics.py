import torch
import sys
import os

# Add the pinn_epi package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pinn_epi.models.physics import SIRModel

def test_sir_model():
    """Test the SIR model implementation."""
    # Create an instance of the SIR model
    sir_model = SIRModel()
    
    # Define test parameters
    beta = 0.4
    gamma = 0.1
    params = {'beta': beta, 'gamma': gamma}
    
    # Define test state: S, I, R
    # Using simple values for easy calculation verification
    S = torch.tensor([0.99])
    I = torch.tensor([0.01])
    R = torch.tensor([0.00])
    u = torch.stack([S, I, R], dim=-1)  # Shape: (1, 3)
    
    # Time tensor (not used in this model but required by interface)
    t = torch.tensor([0.0])
    
    # Calculate derivatives
    derivatives = sir_model.get_derivatives(t, u, params)
    
    # Extract the derivatives
    dS_dt = derivatives[..., 0]
    dI_dt = derivatives[..., 1]
    dR_dt = derivatives[..., 2]
    
    # Expected values based on SIR equations:
    # dS/dt = -beta * S * I
    # dI/dt = beta * S * I - gamma * I
    # dR/dt = gamma * I
    expected_dS_dt = -beta * S * I
    expected_dI_dt = beta * S * I - gamma * I
    expected_dR_dt = gamma * I
    
    # Check if the calculated derivatives match expected values
    assert torch.allclose(dS_dt, expected_dS_dt), f"dS/dt mismatch: got {dS_dt}, expected {expected_dS_dt}"
    assert torch.allclose(dI_dt, expected_dI_dt), f"dI/dt mismatch: got {dI_dt}, expected {expected_dI_dt}"
    assert torch.allclose(dR_dt, expected_dR_dt), f"dR/dt mismatch: got {dR_dt}, expected {expected_dR_dt}"
    
    print("All tests passed!")
    
    # Print the results for visual verification
    print(f"SIR Model Test:")
    print(f"S: {S.item():.4f}, I: {I.item():.4f}, R: {R.item():.4f}")
    print(f"beta: {beta}, gamma: {gamma}")
    print(f"dS/dt: {dS_dt.item():.6f} (expected: {expected_dS_dt.item():.6f})")
    print(f"dI/dt: {dI_dt.item():.6f} (expected: {expected_dI_dt.item():.6f})")
    print(f"dR/dt: {dR_dt.item():.6f} (expected: {expected_dR_dt.item():.6f})")

if __name__ == "__main__":
    test_sir_model()
