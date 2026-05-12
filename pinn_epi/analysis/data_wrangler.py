"""Data handling tools for saving and loading simulation data."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path


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
