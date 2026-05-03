# SIR PINN Library

This library provides a Physics-Informed Neural Network (PINN) implementation for the SIR (Susceptible-Infected-Recovered) epidemiological model. The library allows for customizable layers in the PINN and optional hard initial conditions.

## Files

- `pinn.py`: Defines the PINN class with customizable layers and optional hard initial conditions.
- `train.py`: Contains functions for training the PINN model.
- `plotting.py`: Contains functions for plotting the results of the PINN model.
- `sir_data.py`: Contains functions for generating SIR model data, including the ground truth and residuals.
- `main.py`: Main script that demonstrates how to use the library to train a PINN model on SIR data and plot the results.

## Usage

1. Import the necessary modules:
   ```python
   import torch
   from pinn import PINN
   from train import train_loop
   from plotting import plot_results
   from sir_data import generate_sir_data
   ```

2. Generate SIR model data:
   ```python
   t_max = 30
   t_eval = np.linspace(0, t_max, 100)
   beta_t, gamma_t = 0.3, 0.1
   y0 = [0.99, 0.01, 0.0]
   sol, ds_dt, di_dt, dr_dt = generate_sir_data(t_max, t_eval, beta_t, gamma_t, y0)
   ```

3. Create PINN models with optional hard initial conditions:
   ```python
   model_d = PINN(initial_condition=(0, torch.tensor(y0)))
   model_t = PINN(initial_condition=(0, torch.tensor(y0)))
   ```

4. Train the PINN models:
   ```python
   num_epochs = 5000
   train_loop(model_d, t_tensor, I_obs, 0.5, gamma_t, num_epochs, device)
   train_loop(model_t, t_tensor, I_obs, 0.3, gamma_t, num_epochs, device)
   ```

5. Plot the results:
   ```python
   with torch.no_grad():
       res_t = model_t(t_tensor).cpu()
       res_d = model_d(t_tensor).cpu()
   plot_results(t_eval, sol, res_t, res_d, "plots")
   ```

For more detailed usage, refer to the `main.py` script.
