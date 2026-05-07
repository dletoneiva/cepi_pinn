# Aider Execution Plan: `pinn_epi` Research Framework

## Core Architectural Principles
Before writing any code, adhere to these strict architectural constraints:
1. **Composition over Inheritance:** Do NOT use boolean flags (`if hard_ics:`) to alter network forward passes. Neural networks must be built using composable `nn.Module` wrappers (e.g., wrapping a base `MLP` inside a `HardICWrapper`).
2. **Single Source of Truth for Physics:** The mathematical equations (e.g., SIR dynamics) must be defined exactly once in `models/physics.py`. They will be used both for generating ground truth data via ODE solvers and for calculating physics residuals in the PINN loss function.
3. **Decoupled Orchestration:** The training loop, data generation, and plotting/analysis must exist in completely separate modules. Plotting functions must accept raw arrays/dicts and never compute gradients or losses themselves.

## Plain Text Architecture
The final repository must strictly mirror this file structure:
```text
pinn_epi_project/
├── configs/                  # Hydra YAML configurations (local experiments)
├── pinn_epi/                 # Core Library
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── generator.py      # ODE solving (solve_ivp) using physics models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── physics.py        # CompartmentalModel base class & SIR implementation
│   │   └── networks.py       # Base MLP, HardICWrapper, ModularPINN
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py        # Orchestrator handling Adam/L-BFGS, collocation, MLflow
│   └── analysis/
│       ├── __init__.py
│       ├── evaluator.py      # Hessian, loss landscape, parameter drift calculations
│       └── plotting.py       # Decoupled matplotlib visualizations
├── requirements.txt          # Local dev requirements
└── pyproject.toml            # Minimal build system for editable install
```

---

## Phase 0: Local Project Scaffolding
**Prompt for Aider:**
> "Aider, read the 'Plain Text Architecture' and 'Core Architectural Principles' from this plan. Please generate the exact directory structure outlined above. Then, create a minimal `pyproject.toml` strictly for local editable installation. Create a `requirements.txt` including: `torch`, `numpy`, `scipy`, `matplotlib`, `hydra-core`, `mlflow`, `torchdiffeq`. Finally, create empty `__init__.py` files where necessary."

## Phase 1: The Physics Engine (`models/physics.py`)
**Prompt for Aider:**
> "Aider, let's implement the physics engine in `pinn_epi/models/physics.py`. 
> 1. Create an abstract base class `CompartmentalModel` using `abc.ABC`. It must mandate a method `get_derivatives(self, t: torch.Tensor, u: torch.Tensor, params: dict) -> torch.Tensor`.
> 2. Look at the `sir_model` function in my original file. Port this logic into a new class `SIRModel(CompartmentalModel)`. 
> 3. The `get_derivatives` method in `SIRModel` should extract $S, I, R$ from the tensor `u`, extract `beta` and `gamma` from `params`, and return the stacked time derivatives. Ensure it uses PyTorch operations so gradients can flow through it.
> Do not write any neural network code yet."

## Phase 2: The Neural Network Pipeline (`models/networks.py`)
**Prompt for Aider:**
> "Aider, now we implement the composable PINN architecture in `pinn_epi/models/networks.py`, adhering to the Composition principle.
> 1. Implement a standard `BaseMLP(nn.Module)` that accepts input/output dimensions and hidden layer configurations.
> 2. Look at the `ResearchPINN_HardIC` class in my original code. Implement a `HardICAnsatz(nn.Module)` that takes initial condition values and a starting time `t0`. Its forward pass should take `(t, raw_nn_output)` and return `IC + (t - t0) * raw_nn_output`.
> 3. Implement the `ModularPINN(nn.Module)` base class. Its `__init__` should accept an `encoder` (optional), a `backbone` (like `BaseMLP`), and an `ansatz` (like `HardICAnsatz`). Its forward pass should chain these together strictly: `features = encoder(t)` -> `raw = backbone(features)` -> `ansatz(t, raw)`. 
> Use strict type hinting."

## Phase 3: Data Generation (`data/generator.py`)
**Prompt for Aider:**
> "Aider, let's adhere to the Single Source of Truth principle. Create `pinn_epi/data/generator.py`.
> 1. Create an `ODESimulator` class. Its constructor should accept an instance of `CompartmentalModel`.
> 2. Add a `generate(self, t_span, y0, params, t_eval)` method. 
> 3. Inside `generate`, wrap the `CompartmentalModel.get_derivatives` method so it can be consumed by `scipy.integrate.solve_ivp`. 
> 4. Return the simulated trajectories as standard NumPy arrays. Keep this file strictly NumPy/SciPy for numerical ground truth generation."

## Phase 4: Training Orchestration (`training/trainer.py`)
**Prompt for Aider:**
> "Aider, let's build the decoupled `PINNTrainer` in `pinn_epi/training/trainer.py`. Look at the `train_window` and `train_hybrid` functions in the original code.
> 1. The `PINNTrainer` should take a `ModularPINN` model, a `CompartmentalModel` instance, and an empirical data dictionary/dataset.
> 2. Implement the loss calculation: It MUST compute `autograd` derivatives of the network's output with respect to time, and then call `CompartmentalModel.get_derivatives` to calculate the physics residuals.
> 3. Implement the two-phase optimization loop (Adam followed by L-BFGS). 
> 4. Include a method `sample_collocation_points(self, t_range, n_points)` to dynamically generate the time points where the physics loss is evaluated.
> 5. Add placeholder context managers for `mlflow` to log `total_loss`, `data_loss`, and `phys_loss` at defined intervals."

## Phase 5: Mathematical Analysis (`analysis/evaluator.py`)
**Prompt for Aider:**
> "Aider, let's extract the mathematical analysis tools. Create `pinn_epi/analysis/evaluator.py`.
> Look at the original code block that calculates the 'Data Loss Topology', 'Physics Loss Topology', and 'Normalized Hessian'. 
> 1. Create a function `compute_loss_landscape` that takes a `CompartmentalModel`, ground truth data, and a parameter grid, returning the `data_loss` and `phys_loss` arrays.
> 2. Create a function `compute_1d_hessian` that replicates the finite-difference Hessian calculation (data, physics, and total).
> Ensure these functions are stateless, accept inputs cleanly, and return standard dictionaries or NumPy arrays. Absolutely NO matplotlib code in this file."

## Phase 6: Plotting (`analysis/plotting.py`)
**Prompt for Aider:**
> "Aider, finally, handle visualizations in `pinn_epi/analysis/plotting.py`.
> 1. Look at `plot_figure_1_corrected`, `plot_hessian_scalarization_r`, `plot_sir_bifurcation`, and `plot_windowed_results` in the original code.
> 2. Port these plotting functions over. Modify their signatures so they take the computed arrays/data directly as arguments rather than simulating the data themselves.
> 3. Apply the `plt.rcParams.update` Nature-style configuration dynamically inside a context manager or local initialization function."
