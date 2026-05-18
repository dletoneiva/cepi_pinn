"""Data handling tools for saving and loading simulation data."""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import torch
from pinn_epi.models.physics import CompartmentalModel


class DataWrangler:
    """Handles data processing, validation, and tensor extraction for PINN training."""
    
    def __init__(
        self, 
        physics_model: CompartmentalModel, 
        observables_config: Dict[str, Any]
    ):
        """Initialize the DataWrangler with physics model and observables configuration.
        
        Args:
            physics_model: The compartmental model used for validation
            observables_config: Configuration for observables processing
        """
        self.physics_model = physics_model
        self.observables_config = observables_config
        self.df: Optional[pd.DataFrame] = None
        self.model_params: Optional[Dict[str, float]] = None
        self.initial_conditions: Optional[List[float]] = None
        self.compartment_names: Optional[List[str]] = None
        
    def load_full_dataset(
        self,
        trajectories: Dict[str, np.ndarray],
        model_params: Dict[str, float],
        initial_conditions: List[float],
        compartment_names: List[str],
        t_eval: Optional[np.ndarray] = None
    ) -> None:
        """Load the full dataset from evaluator output.
        
        Args:
            trajectories: Dictionary mapping compartment names to their time series data
            model_params: Dictionary of model parameters
            initial_conditions: List of initial condition values
            compartment_names: List of compartment names in order
            t_eval: Time points array (optional)
        """
        # Store metadata
        self.model_params = model_params
        self.initial_conditions = initial_conditions
        self.compartment_names = compartment_names
        
        # Create dataframe with all compartments
        if t_eval is not None:
            data_dict = {'time': t_eval}
            data_dict.update(trajectories)
        else:
            data_dict = trajectories.copy()
            
        self.df = pd.DataFrame(data_dict)
        
    def validate_observables(self) -> None:
        """Validate that observed variables exist in the model compartments.
        
        Raises:
            ValueError: If any observed variable is not in the model compartments
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_full_dataset first.")
            
        observed_variables = self.observables_config.get('observed_variables', [])
        model_compartments = self.physics_model.compartment_names
        
        invalid_vars = set(observed_variables) - set(model_compartments)
        if invalid_vars:
            raise ValueError(
                f"Observed variables {list(invalid_vars)} not found in model compartments {model_compartments}"
            )
            
    def get_training_tensors(self) -> Dict[str, torch.Tensor]:
        """Extract PyTorch tensors for observed variables only.
        
        Returns:
            Dictionary mapping observed variable names to their PyTorch tensors
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_full_dataset first.")
            
        self.validate_observables()
        observed_variables = self.observables_config.get('observed_variables', [])
        
        # Extract tensors for observed variables only
        training_tensors = {}
        for var in observed_variables:
            training_tensors[var] = torch.tensor(
                self.df[var].values, 
                dtype=torch.float32
            )
            
        # Always include time
        if 'time' in self.df.columns:
            training_tensors['t'] = torch.tensor(
                self.df['time'].values, 
                dtype=torch.float32
            )
        else:
            raise ValueError("Time column not found in dataset")
            
        return training_tensors


def save_simulation_data(
    trajectories: Dict[str, np.ndarray],
    model_params: Dict[str, float],
    initial_conditions: list,
    compartment_names: list,
    save_dir: str,
    t_eval: Optional[np.ndarray] = None,
    file_prefix: str = "simulation",
) -> None:
    """Save simulation data to files in the specified directory.
    
    Args:
        trajectories: Dictionary mapping compartment names to their time series data
        model_params: Dictionary of model parameters
        initial_conditions: List of initial condition values
        compartment_names: List of compartment names in order
        save_dir: Directory to save the data files
        t_eval: Time points array (optional)
        file_prefix: Prefix for saved files
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories as CSV
    if t_eval is not None:
        data_dict = {'time': t_eval}
        data_dict.update(trajectories)
    else:
        data_dict = trajectories.copy()
    
    df = pd.DataFrame(data_dict)
    csv_path = save_dir / f"{file_prefix}_trajectories.csv"
    df.to_csv(csv_path, index=False)
    
    # Save metadata as JSON
    metadata = {
        'model_params': model_params,
        'initial_conditions': initial_conditions,
        'compartment_names': compartment_names
    }
    json_path = save_dir / f"{file_prefix}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved simulation data to {save_dir}")


def load_simulation_data(
    save_dir: str,
    file_prefix: str = "simulation"
) -> Dict[str, Any]:
    """Load simulation data from files.
    
    Args:
        save_dir: Directory containing the data files
        file_prefix: Prefix used when saving files
        
    Returns:
        Dictionary containing trajectories, model_params, initial_conditions, and compartment_names
    """
    save_dir = Path(save_dir)
    
    # Load trajectories from CSV
    csv_path = save_dir / f"{file_prefix}_trajectories.csv"
    df = pd.read_csv(csv_path)
    trajectories = {col: df[col].values for col in df.columns if col != 'time'}
    t_eval = df['time'].values if 'time' in df.columns else None
    
    # Load metadata from JSON
    json_path = save_dir / f"{file_prefix}_metadata.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    result = {
        'trajectories': trajectories,
        'model_params': metadata['model_params'],
        'initial_conditions': metadata['initial_conditions'],
        'compartment_names': metadata['compartment_names']
    }
    
    if t_eval is not None:
        result['t_eval'] = t_eval
    
    return result


def save_trajectories(
    trajectories: Dict[str, np.ndarray],
    save_path: str,
    t_eval: Optional[np.ndarray] = None
) -> None:
    """Save just the trajectory data to a CSV file.
    
    Args:
        trajectories: Dictionary mapping compartment names to their time series data
        save_path: Path to save the CSV file
        t_eval: Time points array (optional)
    """
    if t_eval is not None:
        data_dict = {'time': t_eval}
        data_dict.update(trajectories)
    else:
        data_dict = trajectories.copy()
    
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, index=False)
    print(f"Saved trajectories to {save_path}")


def load_trajectories(save_path: str) -> Dict[str, np.ndarray]:
    """Load trajectory data from a CSV file.
    
    Args:
        save_path: Path to the CSV file
        
    Returns:
        Dictionary mapping column names to their data arrays
    """
    df = pd.read_csv(save_path)
    return {col: df[col].values for col in df.columns}
