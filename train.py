import torch
import numpy as np
from pinn import PINN

def train(model, t_tensor, I_obs, beta, gamma, optimizer, device):
    t_p = t_tensor.clone().requires_grad_(True)
    u = model(t_p)
    S, I, R = u[:,0:1], u[:,1:2], u[:,2:3]
    ds = torch.autograd.grad(S, t_p, torch.ones_like(S), create_graph=True)[0]
    di = torch.autograd.grad(I, t_p, torch.ones_like(I), create_graph=True)[0]
    dr = torch.autograd.grad(R, t_p, torch.ones_like(R), create_graph=True)[0]
    res_s = ds + beta * S * I
    res_i = di - (beta * S * I - gamma * I)
    res_r = dr - gamma * I
    l_phys = torch.mean(res_s**2 + res_i**2 + res_r**2)
    l_data = torch.mean((I - I_obs)**2)
    l_cons = torch.mean((S + I + R - 1.0)**2)
    loss = l_data + l_phys + l_cons
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_loop(model, t_tensor, I_obs, beta, gamma, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(num_epochs):
        loss = train(model, t_tensor, I_obs, beta, gamma, optimizer, device)
        losses.append(loss)
    return losses
