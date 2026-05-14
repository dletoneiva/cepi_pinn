import torch
import numpy as np
import tempfile
import os
import sys
from pinn_epi.models.networks import ModularPINN, BaseMLP
from pinn_epi.models.physics import SIModel
from pinn_epi.training.trainer import PINNTrainer

def generate_synthetic_si_data(beta=0.3, gamma=0.0, S0=0.99, I0=0.01, t_span=[0, 10], num_points=50):
    """
    Generate synthetic SI model data for testing.
    
    Args:
        beta: Infection rate
        gamma: Recovery rate (0 for SI model)
        S0: Initial susceptible population
        I0: Initial infected population
        t_span: Time span [t_start, t_end]
        num_points: Number of data points
    
    Returns:
        dict: Dictionary with 't', 'S', 'I' keys containing time series data
    """
    t = np.linspace(t_span[0], t_span[1], num_points)
    
    # Analytical solution for SI model
    # dS/dt = -beta * S * I
    # dI/dt = beta * S * I
    # With S + I = 1 (constant population), we can solve this analytically
    
    # I(t) = I0 * exp(beta * t) / (1 - S0 + I0 * exp(beta * t))
    # S(t) = 1 - I(t)
    
    exp_term = np.exp(beta * t)
    I = I0 * exp_term / (1 - S0 + I0 * exp_term)
    S = 1 - I
    
    return {
        't': t,
        'S': S,
        'I': I
    }

def test_simple_si_training():
    """Test PINN training on a simple SI model with shallow network."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic SI data...")
    data = generate_synthetic_si_data(beta=0.3, gamma=0.0, S0=0.99, I0=0.01, t_span=[0, 10], num_points=30)
    
    # Create SI physics model
    print("Creating SI physics model...")
    physics_model = SIModel()
    
    # Create simple network (shallow and small for fast testing)
    print("Creating simple neural network...")
    backbone = BaseMLP(
        input_dim=1,      # Time is 1-dimensional
        hidden_dims=[8],  # Very small hidden layer
        output_dim=2,     # SI model has 2 compartments
        activation=torch.nn.Tanh
    )
    
    # Create PINN model
    model = ModularPINN(backbone=backbone)
    model.to(device)
    
    # Training configuration
    config = {
        'target_compartments': ['S', 'I'],
        'physics_params': {'beta': 0.3, 'gamma': 0.0},
        'data_weight': 1.0,
        'physics_weight': 1.0,
        'adam_lr': 0.01,
        'lbfgs_max_iter': 5,  # Very few iterations for quick test
        'adam_epochs': 20,    # Very few epochs for quick test
        'n_collocation_points': 20,
        'log_interval': 10
    }
    
    # Create trainer
    print("Creating trainer...")
    trainer = PINNTrainer(
        model=model,
        physics_model=physics_model,
        data=data,
        config=config
    )
    
    # Test collocation point sampling
    print("Testing collocation point sampling...")
    collocation_points = trainer.sample_collocation_points((0, 10), 10)
    assert collocation_points.shape == (10,), "Collocation points should have correct shape"
    assert torch.all(collocation_points >= 0) and torch.all(collocation_points <= 10), "Collocation points should be in correct range"
    print("✓ Collocation point sampling works correctly")
    
    # Test loss computation
    print("Testing loss computation...")
    t_tensor = torch.tensor(data['t'], dtype=torch.float32).view(-1, 1).to(device)
    y_true_array = np.column_stack([data['S'], data['I']])
    y_true_tensor = torch.tensor(y_true_array, dtype=torch.float32).to(device)
    
    try:
        total_loss, data_loss, physics_loss = trainer.compute_loss(
            t_tensor, y_true_tensor, collocation_points, ['S', 'I']
        )
        print(f"✓ Loss computation successful - Total: {total_loss.item():.6f}, Data: {data_loss.item():.6f}, Physics: {physics_loss.item():.6f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        raise
    
    # Test full training
    print("Testing full training...")
    try:
        trainer.train()
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise
    
    # Test model prediction
    print("Testing model prediction...")
    t_test = torch.linspace(0, 10, 50).view(-1, 1).to(device)
    with torch.no_grad():
        predictions = model(t_test)
        assert predictions.shape == (50, 2), f"Predictions should have shape (50, 2), got {predictions.shape}"
        print("✓ Model prediction works correctly")
    
    print("\nAll tests passed! PINNTrainer is working correctly.")

def test_physics_consistency():
    """Test that the trained model satisfies the physics equations."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic data
    data = generate_synthetic_si_data(beta=0.3, gamma=0.0, S0=0.99, I0=0.01, t_span=[0, 10], num_points=20)
    
    # Create SI physics model
    physics_model = SIModel()
    
    # Create simple network
    backbone = BaseMLP(
        input_dim=1,
        hidden_dims=[8],
        output_dim=2,
        activation=torch.nn.Tanh
    )
    
    # Create PINN model
    model = ModularPINN(backbone=backbone)
    model.to(device)
    
    # Training configuration with more epochs for better physics consistency
    config = {
        'target_compartments': ['S', 'I'],
        'physics_params': {'beta': 0.3, 'gamma': 0.0},
        'data_weight': 1.0,
        'physics_weight': 1.0,
        'adam_lr': 0.01,
        'lbfgs_max_iter': 3,
        'adam_epochs': 10,
        'n_collocation_points': 15,
        'log_interval': 5
    }
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        physics_model=physics_model,
        data=data,
        config=config
    )
    
    # Train the model
    trainer.train()
    
    # Test physics consistency at collocation points
    t_min, t_max = data['t'].min(), data['t'].max()
    collocation_points = trainer.sample_collocation_points((t_min, t_max), 10)
    
    # Compute physics residuals
    t_phys = collocation_points.clone().requires_grad_(True)
    y_pred_phys = model(t_phys)
    
    # Compute derivatives using autograd
    du_dt = []
    for i in range(y_pred_phys.shape[1]):
        grad = torch.autograd.grad(
            y_pred_phys[:, i], 
            t_phys, 
            grad_outputs=torch.ones_like(y_pred_phys[:, i]),
            create_graph=True
        )[0]
        du_dt.append(grad)
    du_dt = torch.cat(du_dt, dim=1)
    
    # Get expected physics residuals
    physics_residuals = physics_model.get_derivatives(t_phys, y_pred_phys, config.get('physics_params', {}))
    
    # Check that residuals are small (physics is satisfied)
    residual_error = torch.mean((du_dt - physics_residuals) ** 2)
    print(f"Physics residual error: {residual_error.item():.6f}")
    
    # The physics should be reasonably satisfied
    assert residual_error.item() < 1.0, f"Physics residual error {residual_error.item():.6f} is too large"
    print("✓ Physics consistency test passed")

def test_network_physics_integration():
    """Test integration between network and physics model."""
    
    # Create a simple network
    backbone = BaseMLP(
        input_dim=1,
        hidden_dims=[4],
        output_dim=2,
        activation=torch.nn.Tanh
    )
    
    model = ModularPINN(backbone=backbone)
    
    # Create SI physics model
    physics_model = SIModel()
    
    # Test that the network output dimensions match physics model expectations
    test_input = torch.tensor([[0.0], [5.0], [10.0]])  # 3 time points
    output = model(test_input)
    
    # Check output shape
    assert output.shape == (3, 2), f"Network output shape {output.shape} doesn't match expected (3, 2)"
    
    # Check that physics model can process the output
    params = {'beta': 0.3, 'gamma': 0.0}
    derivatives = physics_model.get_derivatives(test_input, output, params)
    
    # Check derivatives shape
    assert derivatives.shape == output.shape, f"Derivatives shape {derivatives.shape} doesn't match output shape {output.shape}"
    
    print("✓ Network and physics integration test passed")

if __name__ == "__main__":
    # Set up temporary directory for MLflow (if needed)
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['MLFLOW_TRACKING_URI'] = f"sqlite:///{os.path.join(temp_dir, 'mlflow.db')}"
        
        try:
            test_simple_si_training()
            test_physics_consistency()
            test_network_physics_integration()
            print("\n🎉 All tests completed successfully!")
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            raise
