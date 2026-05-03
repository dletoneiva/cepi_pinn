import numpy as np
from scipy.integrate import solve_ivp
import torch
from pinn import PINN
from train import train_loop
from plotting import plot_results
from sir_data import generate_sir_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ground Truth
t_max = 30
t_eval = np.linspace(0, t_max, 100)
beta_t, gamma_t = 0.3, 0.1
y0 = [0.99, 0.01, 0.0]

sol, ds_dt, di_dt, dr_dt = generate_sir_data(t_max, t_eval, beta_t, gamma_t, y0)

I_obs = torch.tensor(sol.y[1], dtype=torch.float32).view(-1, 1).to(device)
t_tensor = torch.tensor(t_eval, dtype=torch.float32).view(-1, 1).to(device)

# PINN
model_d = PINN(initial_condition=(0, torch.tensor(y0))).to(device)
model_t = PINN(initial_condition=(0, torch.tensor(y0))).to(device)

# Training
num_epochs = 5000
train_loop(model_d, t_tensor, I_obs, 0.5, gamma_t, num_epochs, device)
train_loop(model_t, t_tensor, I_obs, 0.3, gamma_t, num_epochs, device)

# Plotting
with torch.no_grad():
    res_t = model_t(t_tensor).cpu()
    res_d = model_d(t_tensor).cpu()

plot_results(t_eval, sol, res_t, res_d, "plots")
