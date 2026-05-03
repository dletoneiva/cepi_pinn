import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=[1, 128, 128, 128, 3], initial_condition=None):
        super().__init__()
        self.net = nn.Sequential(*self.build_layers(layers))
        self.initial_condition = initial_condition
    
    def build_layers(self, layers):
        nn_layers = []
        for i in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                nn_layers.append(nn.Tanh())
        return nn_layers
    
    def forward(self, t):
        if self.initial_condition is not None:
            t0, u0 = self.initial_condition
            return u0 + (t - t0) * self.net(t)
        else:
            return self.net(t)
