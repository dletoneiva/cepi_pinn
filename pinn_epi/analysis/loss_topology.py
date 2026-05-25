import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from pinn_epi.models.physics import CompartmentalModel
from pinn_epi.analysis.evaluator import solve_compartmental_model
from pinn_epi.analysis.plotting import apply_plot_style

def compute_loss_topology(
    model: CompartmentalModel,
    reference_trajectories: Dict[str, np.ndarray],
    t_eval: np.ndarray,
    param1_name: str,
    param1_range: np.ndarray,
    param2_name: str,
    param2_range: np.ndarray,
    initial_conditions: list,
    loss_type: str = 'data',  # 'data', 'physics', or 'total'
    observables_config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute loss landscape for a compartmental model over a 2D parameter grid.
    
    Args:
        model: CompartmentalModel instance
        reference_trajectories: Ground truth trajectories
        t_eval: Time points for evaluation
        param1_name: Name of first parameter to sweep
        param1_range: Range of values for first parameter
        param2_name: Name of second parameter to sweep
        param2_range: Range of values for second parameter
        initial_conditions: Initial conditions for ODE solver
        loss_type: Type of loss to compute
        observables_config: Configuration for observables
        
    Returns:
        Tuple of (param1_grid, param2_grid, loss_matrix)
    """
    
    # Get reference infected compartment for physics loss calculation
    if 'I' in reference_trajectories:
        I_ref = reference_trajectories['I']
        S_ref = reference_trajectories.get('S', np.zeros_like(I_ref))
    else:
        # For models without explicit I compartment, use first compartment
        I_ref = reference_trajectories[list(reference_trajectories.keys())[0]]
        S_ref = I_ref  # fallback
    
    # Precompute derivatives for physics loss if needed
    if loss_type in ['physics', 'total']:
        # This would need to be adapted based on your specific model
        dI_dt_ref = np.gradient(I_ref, t_eval)
    
    # Create parameter grids
    P1, P2 = np.meshgrid(param1_range, param2_range)
    loss_matrix = np.zeros_like(P1)
    
    print(f"Computing {loss_type} loss topology...")
    
    # Compute loss for each parameter combination
    for i in range(len(param2_range)):
        for j in range(len(param1_range)):
            param1_val = P1[i, j]
            param2_val = P2[i, j]
            
            # Set parameters
            params = model.parameters.copy() if model.parameters else {}
            params[param1_name] = param1_val
            params[param2_name] = param2_val
            
            # Solve model with current parameters
            try:
                solution = solve_compartmental_model(
                    model, [t_eval[0], t_eval[-1]], initial_conditions, 
                    params, t_eval=t_eval
                )
                
                # Extract predicted trajectory
                if 'I' in solution:
                    I_pred = solution['I']
                else:
                    I_pred = solution[list(solution.keys())[0]]
                
                # Compute data loss
                if loss_type in ['data', 'total']:
                    data_loss = np.mean((I_pred - I_ref)**2)
                else:
                    data_loss = 0.0
                
                # Compute physics loss
                if loss_type in ['physics', 'total']:
                    # This needs to be model-specific
                    # For SIR-like models: dI/dt = βSI - γI
                    dI_dt_pred = np.gradient(I_pred, t_eval)
                    physics_residual = dI_dt_ref - dI_dt_pred
                    physics_loss = np.mean(physics_residual**2)
                else:
                    physics_loss = 0.0
                
                # Combine losses
                if loss_type == 'data':
                    loss_matrix[i, j] = data_loss
                elif loss_type == 'physics':
                    loss_matrix[i, j] = physics_loss
                else:  # total
                    loss_matrix[i, j] = data_loss + physics_loss
                    
            except Exception as e:
                print(f"Error at {param1_name}={param1_val}, {param2_name}={param2_val}: {e}")
                loss_matrix[i, j] = np.nan
    
    return P1, P2, loss_matrix

def plot_loss_topology(
    P1: np.ndarray,
    P2: np.ndarray,
    loss_matrix: np.ndarray,
    param1_name: str,
    param2_name: str,
    title: str = "Loss Topology",
    save_path: Optional[str] = None,
    true_params: Optional[Dict[str, float]] = None,
    valley_line: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> None:
    """
    Plot loss topology as a contour plot.
    
    Args:
        P1: Parameter 1 grid
        P2: Parameter 2 grid
        loss_matrix: Computed loss values
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        title: Plot title
        save_path: Path to save figure (PDF)
        true_params: True parameter values to mark on plot
        valley_line: Line showing unidentifiable valley (x, y coordinates)
    """
    
    apply_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to log scale for better visualization
    log_loss = np.log10(loss_matrix + 1e-12)  # Add small value to avoid log(0)
    
    # Plot contour
    contour = ax.contourf(P1, P2, log_loss, levels=25, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(r'$\log_{10}$(Loss)', fontsize=14)
    
    # Plot true parameters if provided
    if true_params and param1_name in true_params and param2_name in true_params:
        ax.plot(true_params[param1_name], true_params[param2_name], 
                'r*', markersize=15, markeredgecolor='white', 
                label='True Parameters')
    
    # Plot valley line if provided
    if valley_line:
        ax.plot(valley_line[0], valley_line[1], 'w--', lw=2, 
                label='Unidentifiable Valley')
    
    ax.set_xlabel(f'{param1_name}', fontsize=14)
    ax.set_ylabel(f'{param2_name}', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    if true_params or valley_line:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Loss topology saved to {save_path}")
    
    plt.show()

def analyze_model_loss_topology(
    model: CompartmentalModel,
    reference_trajectories: Dict[str, np.ndarray],
    t_eval: np.ndarray,
    initial_conditions: list,
    param_sweep_config: Dict[str, Any],
    save_dir: str,
    model_name: str = "CompartmentalModel"
) -> None:
    """
    Perform complete loss topology analysis for a model.
    
    Args:
        model: CompartmentalModel instance
        reference_trajectories: Ground truth data
        t_eval: Time evaluation points
        initial_conditions: Initial conditions
        param_sweep_config: Configuration for parameter sweeps
        save_dir: Directory to save results
        model_name: Name of the model for labeling
    """
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get parameter names to sweep (first two parameters)
    param_names = list(model.parameters.keys()) if model.parameters else []
    if len(param_names) < 2:
        print("Need at least 2 parameters for 2D sweep")
        return
    
    param1_name = param_names[0]
    param2_name = param_names[1]
    
    # Get parameter ranges from config or use defaults
    param1_range = param_sweep_config.get(param1_name, np.linspace(0.1, 1.0, 30))
    param2_range = param_sweep_config.get(param2_name, np.linspace(0.01, 0.5, 30))
    
    # True parameters for reference
    true_params = model.parameters if model.parameters else {}
    
    # Compute valley line (example: r = β - γ = constant)
    if param1_name == 'beta' and param2_name == 'gamma':
        valley_r = 0.3  # Example R0 value
        valley_x = np.linspace(param1_range[0], param1_range[-1], 50)
        valley_y = valley_x - valley_r
        valley_line = (valley_x, valley_y)
    else:
        valley_line = None
    
    # Compute and plot data loss topology
    P1, P2, data_loss = compute_loss_topology(
        model, reference_trajectories, t_eval,
        param1_name, param1_range, param2_name, param2_range,
        initial_conditions, loss_type='data'
    )
    
    plot_loss_topology(
        P1, P2, data_loss,
        param1_name, param2_name,
        title=f"{model_name} - Data Loss Topology",
        save_path=f"{save_dir}/{model_name}_data_loss_topology.pdf",
        true_params=true_params,
        valley_line=valley_line
    )
    
    # Compute and plot physics loss topology
    P1, P2, physics_loss = compute_loss_topology(
        model, reference_trajectories, t_eval,
        param1_name, param1_range, param2_name, param2_range,
        initial_conditions, loss_type='physics'
    )
    
    plot_loss_topology(
        P1, P2, physics_loss,
        param1_name, param2_name,
        title=f"{model_name} - Physics Loss Topology",
        save_path=f"{save_dir}/{model_name}_physics_loss_topology.pdf",
        true_params=true_params,
        valley_line=valley_line
    )
    
    # Compute and plot total loss topology
    P1, P2, total_loss = compute_loss_topology(
        model, reference_trajectories, t_eval,
        param1_name, param1_range, param2_name, param2_range,
        initial_conditions, loss_type='total'
    )
    
    plot_loss_topology(
        P1, P2, total_loss,
        param1_name, param2_name,
        title=f"{model_name} - Total Loss Topology",
        save_path=f"{save_dir}/{model_name}_total_loss_topology.pdf",
        true_params=true_params,
        valley_line=valley_line
    )
