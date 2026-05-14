import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from omegaconf import DictConfig, OmegaConf
from pinn_epi.experiments.train_pinn import main

def create_test_config() -> DictConfig:
    """Create a minimal test configuration for PINN training."""
    config_dict = {
        "compartmental": {
            "model": {
                "type": "SIR",
                "parameters": {
                    "beta": 0.3,
                    "gamma": 0.1
                },
                "initial_conditions": [0.99, 0.01, 0.0]
            }
        },
        "training": {
            "data": {
                "source": "generated",
                "t_span": [0, 20],
                "num_points": 50
            },
            "optimizer": {
                "adam": {
                    "lr": 0.001,
                    "epochs": 2
                },
                "lbfgs": {
                    "lr": 0.01,
                    "max_iter": 2
                }
            },
            "physics": {
                "residual_weight": 1.0
            },
            "data_loss_weight": 1.0
        },
        "network": {
            "input_dim": 1,
            "network": {
                "layer_size": 16,
                "num_layers": 2,
                "activation": "Tanh"
            },
            "use_time_normalization": False,
            "use_hard_ic": False
        },
        "seed": 42
    }
    return OmegaConf.create(config_dict)

def test_pinn_training_execution():
    """Test that PINN training executes without errors."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to the temporary directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create test configuration
            cfg = create_test_config()
            
            # Run the training
            main(cfg)
            
            # If we reach this point, the training executed successfully
            assert True
            
        except Exception as e:
            # If there's an exception, fail the test with the error message
            pytest.fail(f"PINN training failed with error: {str(e)}")
            
        finally:
            # Change back to the original directory
            os.chdir(original_cwd)
