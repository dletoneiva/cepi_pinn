# pinn_epi: Physics-Informed Neural Networks for Epidemiological Modeling

This library provides a modular Physics-Informed Neural Network (PINN) framework for epidemiological models, specifically designed for the SIR (Susceptible-Infected-Recovered) model. The framework follows a clean architectural pattern with separated concerns for physics modeling, neural networks, data generation, training, and analysis.

## Project Structure

- `configs/` - Hydra YAML configurations for experiments
- `pinn_epi/` - Core library package
  - `data/` - Data generation utilities
  - `models/` - Physics models and neural network architectures
  - `training/` - Training orchestration
  - `analysis/` - Evaluation and plotting tools
- `requirements.txt` - Development dependencies
- `pyproject.toml` - Build system configuration

## Installation

```bash
pip install -e .
```

## Key Features

- **Modular Design**: Clean separation between physics models, neural networks, and training logic
- **Composable Architecture**: Neural networks built with wrapper modules rather than conditional logic
- **Single Source of Truth**: Physics equations defined once and reused across components
- **Decoupled Analysis**: Evaluation and plotting separated from core computational logic

## Usage

The library is organized into modules that can be imported and used independently:

```python
from pinn_epi.models.physics import SIRModel
from pinn_epi.models.networks import ModularPINN, BaseMLP
from pinn_epi.data.generator import ODESimulator
```

For detailed examples, please see the example scripts and documentation.
