"""Unit tests for the data_wrangler module."""

import unittest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from pinn_epi.analysis.data_wrangler import DataWrangler, save_simulation_data, load_simulation_data, save_trajectories, load_trajectories
from pinn_epi.models.physics import SIRModel


class TestSIRModelForTesting(SIRModel):
    """Test implementation of SIRModel for testing purposes."""
    
    @property
    def compartment_names(self):
        return ['S', 'I', 'R']
    
    def get_derivatives(self, t, u, params):
        # Simplified implementation for testing
        return torch.zeros_like(u)


class TestDataWrangler(unittest.TestCase):
    """Test cases for the DataWrangler class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.physics_model = TestSIRModelForTesting()
        self.observables_config = {
            'observed_variables': ['S', 'I']
        }
        self.wrangler = DataWrangler(self.physics_model, self.observables_config)
        
        # Sample test data
        self.t_eval = np.linspace(0, 10, 11)
        self.trajectories = {
            'S': np.linspace(0.9, 0.1, 11),
            'I': np.linspace(0.1, 0.8, 11),
            'R': np.linspace(0.0, 0.1, 11)
        }
        self.model_params = {'beta': 0.3, 'gamma': 0.1}
        self.initial_conditions = [0.9, 0.1, 0.0]
        self.compartment_names = ['S', 'I', 'R']
    
    def test_init(self):
        """Test DataWrangler initialization."""
        self.assertIsInstance(self.wrangler, DataWrangler)
        self.assertEqual(self.wrangler.physics_model, self.physics_model)
        self.assertEqual(self.wrangler.observables_config, self.observables_config)
        self.assertIsNone(self.wrangler.df)
        self.assertIsNone(self.wrangler.model_params)
        self.assertIsNone(self.wrangler.initial_conditions)
        self.assertIsNone(self.wrangler.compartment_names)
    
    def test_load_full_dataset_with_time(self):
        """Test loading full dataset with time evaluation points."""
        self.wrangler.load_full_dataset(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names,
            t_eval=self.t_eval
        )
        
        self.assertIsNotNone(self.wrangler.df)
        self.assertEqual(self.wrangler.model_params, self.model_params)
        self.assertEqual(self.wrangler.initial_conditions, self.initial_conditions)
        self.assertEqual(self.wrangler.compartment_names, self.compartment_names)
        self.assertIn('time', self.wrangler.df.columns)
        self.assertEqual(len(self.wrangler.df), 11)
    
    def test_load_full_dataset_without_time(self):
        """Test loading full dataset without time evaluation points."""
        self.wrangler.load_full_dataset(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names
        )
        
        self.assertIsNotNone(self.wrangler.df)
        self.assertEqual(self.wrangler.model_params, self.model_params)
        self.assertEqual(self.wrangler.initial_conditions, self.initial_conditions)
        self.assertEqual(self.wrangler.compartment_names, self.compartment_names)
        self.assertNotIn('time', self.wrangler.df.columns)
    
    def test_validate_observables_success(self):
        """Test successful validation of observables."""
        self.wrangler.load_full_dataset(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names,
            t_eval=self.t_eval
        )
        
        # Should not raise an exception
        self.wrangler.validate_observables()
    
    def test_validate_observables_invalid_variable(self):
        """Test validation failure with invalid observable variable."""
        invalid_config = {'observed_variables': ['S', 'X']}  # X is not a valid compartment
        wrangler = DataWrangler(self.physics_model, invalid_config)
        
        wrangler.load_full_dataset(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names,
            t_eval=self.t_eval
        )
        
        with self.assertRaises(ValueError) as context:
            wrangler.validate_observables()
        
        self.assertIn("Observed variables ['X'] not found", str(context.exception))
    
    def test_validate_observables_no_dataset(self):
        """Test validation failure when no dataset is loaded."""
        with self.assertRaises(ValueError) as context:
            self.wrangler.validate_observables()
        
        self.assertIn("Dataset not loaded", str(context.exception))
    
    def test_get_training_tensors_success(self):
        """Test successful extraction of training tensors."""
        self.wrangler.load_full_dataset(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names,
            t_eval=self.t_eval
        )
        
        tensors = self.wrangler.get_training_tensors()
        
        self.assertIn('S', tensors)
        self.assertIn('I', tensors)
        self.assertIn('t', tensors)
        self.assertIsInstance(tensors['S'], torch.Tensor)
        self.assertIsInstance(tensors['I'], torch.Tensor)
        self.assertIsInstance(tensors['t'], torch.Tensor)
        self.assertEqual(tensors['S'].shape, torch.Size([11]))
        self.assertEqual(tensors['I'].shape, torch.Size([11]))
        self.assertEqual(tensors['t'].shape, torch.Size([11]))
    
    def test_get_training_tensors_no_time_column(self):
        """Test failure when time column is missing."""
        self.wrangler.load_full_dataset(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names
            # No t_eval provided
        )
        
        with self.assertRaises(ValueError) as context:
            self.wrangler.get_training_tensors()
        
        self.assertIn("Time column not found", str(context.exception))
    
    def test_get_training_tensors_no_dataset(self):
        """Test failure when no dataset is loaded."""
        with self.assertRaises(ValueError) as context:
            self.wrangler.get_training_tensors()
        
        self.assertIn("Dataset not loaded", str(context.exception))


class TestSaveLoadFunctions(unittest.TestCase):
    """Test cases for the save/load functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.t_eval = np.linspace(0, 10, 11)
        self.trajectories = {
            'S': np.linspace(0.9, 0.1, 11),
            'I': np.linspace(0.1, 0.8, 11),
            'R': np.linspace(0.0, 0.1, 11)
        }
        self.model_params = {'beta': 0.3, 'gamma': 0.1}
        self.initial_conditions = [0.9, 0.1, 0.0]
        self.compartment_names = ['S', 'I', 'R']
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_simulation_data(self):
        """Test saving and loading simulation data."""
        # Save data
        save_simulation_data(
            trajectories=self.trajectories,
            model_params=self.model_params,
            initial_conditions=self.initial_conditions,
            compartment_names=self.compartment_names,
            save_dir=self.temp_dir,
            t_eval=self.t_eval,
            file_prefix="test_sim"
        )
        
        # Check that files were created
        csv_path = Path(self.temp_dir) / "test_sim_trajectories.csv"
        json_path = Path(self.temp_dir) / "test_sim_metadata.json"
        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())
        
        # Load data
        loaded_data = load_simulation_data(self.temp_dir, "test_sim")
        
        # Check loaded data
        self.assertIn('trajectories', loaded_data)
        self.assertIn('model_params', loaded_data)
        self.assertIn('initial_conditions', loaded_data)
        self.assertIn('compartment_names', loaded_data)
        self.assertIn('t_eval', loaded_data)
        
        np.testing.assert_array_equal(loaded_data['t_eval'], self.t_eval)
        self.assertEqual(loaded_data['model_params'], self.model_params)
        self.assertEqual(loaded_data['initial_conditions'], self.initial_conditions)
        self.assertEqual(loaded_data['compartment_names'], self.compartment_names)
        
        # Check trajectories
        for key in self.trajectories:
            np.testing.assert_array_almost_equal(
                loaded_data['trajectories'][key], 
                self.trajectories[key]
            )
    
    def test_save_and_load_trajectories(self):
        """Test saving and loading just trajectory data."""
        save_path = os.path.join(self.temp_dir, "trajectories.csv")
        
        # Save trajectories
        save_trajectories(self.trajectories, save_path, self.t_eval)
        
        # Check file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Load trajectories
        loaded_trajectories = load_trajectories(save_path)
        
        # Check loaded data
        self.assertIn('time', loaded_trajectories)
        np.testing.assert_array_equal(loaded_trajectories['time'], self.t_eval)
        
        for key in self.trajectories:
            np.testing.assert_array_almost_equal(
                loaded_trajectories[key], 
                self.trajectories[key]
            )
    
    def test_save_and_load_trajectories_without_time(self):
        """Test saving and loading trajectory data without time."""
        save_path = os.path.join(self.temp_dir, "trajectories_no_time.csv")
        
        # Save trajectories without time
        save_trajectories(self.trajectories, save_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Load trajectories
        loaded_trajectories = load_trajectories(save_path)
        
        # Check loaded data (no time column)
        self.assertNotIn('time', loaded_trajectories)
        
        for key in self.trajectories:
            np.testing.assert_array_almost_equal(
                loaded_trajectories[key], 
                self.trajectories[key]
            )


if __name__ == '__main__':
    unittest.main()
