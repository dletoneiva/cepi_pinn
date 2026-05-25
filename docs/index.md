# Welcome to cepi_pinn Documentation

cepi_pinn is a Python library for Physics-Informed Neural Networks (PINNs) applied to epidemiological modeling. This library provides tools to solve and learn from compartmental models like SIR, SEIR, and their variants using deep learning techniques.

## Key Features

- **Multiple Epidemiological Models**: SIR, SEIR, SI, SIS, and extended models with vaccination and mortality
- **Physics-Informed Training**: Neural networks trained to respect the underlying differential equations
- **Modular Architecture**: Flexible network components that can be combined
- **Configuration-Driven**: YAML-based configuration for experiments
- **MLflow Integration**: Experiment tracking and model logging
- **Comprehensive Documentation**: Detailed API reference and usage examples

## Getting Started

To get started with cepi_pinn, check out our [Getting Started guide](getting_started.md) which walks through installation and basic usage.

## API Reference

For detailed information about the library's components, see our [API Reference](api/physics.md):

- [Physics Models](api/physics.md): Compartmental models defining the epidemiological dynamics
- [Trainer](api/trainer.md): Training orchestration for PINNs
- [Networks](api/networks.md): Neural network architectures for PINNs
- [Model Loader](api/model_loader.md): Configuration-based model creation

## Use Cases

cepi_pinn is designed for researchers and practitioners who want to:

- Learn epidemiological parameters from data
- Solve complex compartmental models without traditional numerical methods
- Incorporate physical constraints into machine learning models
- Explore "what-if" scenarios in disease modeling
