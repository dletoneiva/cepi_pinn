import numpy as np
from scipy.integrate import solve_ivp

def sir_model(t, y, b, g):
    S, I, R = y
    return [-b*S*I, b*S*I - g*I, g*I]

def generate_sir_data(t_max, t_eval, beta, gamma, y0):
    sol = solve_ivp(sir_model, [0, t_max], y0, args=(beta, gamma), t_eval=t_eval)
    
    S, I, R = sol.y
    ds_dt = -beta * S * I
    di_dt = beta * S * I - gamma * I
    dr_dt = gamma * I
    
    return sol, ds_dt, di_dt, dr_dt
