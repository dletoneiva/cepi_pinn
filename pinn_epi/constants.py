"""Centralized constants for the pinn_epi library."""

# Model registry mapping model names to classes
from pinn_epi.models.physics import (
    SIRModel, SEIRModel, SIModel, SISModel, 
    SIRVModel, SIRDModel, SIRDVModel, SEIRDModel, 
    SEIRVModel, SEIRDVModel
)

COMPARTMENTAL_MODEL_REGISTRY = {
    "SIRModel": "pinn_epi.models.physics.SIRModel",
    "SEIRModel": "pinn_epi.models.physics.SEIRModel",
    "SIModel": "pinn_epi.models.physics.SIModel",
    "SISModel": "pinn_epi.models.physics.SISModel",
    "SIRVModel": "pinn_epi.models.physics.SIRVModel",
    "SIRDModel": "pinn_epi.models.physics.SIRDModel",
    "SIRDVModel": "pinn_epi.models.physics.SIRDVModel",
    "SEIRDModel": "pinn_epi.models.physics.SEIRDModel",
    "SEIRVModel": "pinn_epi.models.physics.SEIRVModel",
    "SEIRDVModel": "pinn_epi.models.physics.SEIRDVModel",
}

# Map model types to their classes for easy access
COMPARTMENTAL_MODEL_MAP = {
    'sir': SIRModel,
    'seir': SEIRModel,
    'si': SIModel,
    'sis': SISModel,
    'sird': SIRDModel,
    'sirv': SIRVModel,
    'seird': SEIRDModel,
    'seirv': SEIRVModel,
    'sirdv': SIRDVModel,
    'seirdv': SEIRDVModel
}

# Standard compartment colors for plotting
COMPARTMENT_COLORS = {
    'S': '#377EB8',  # Blue
    'I': '#E41A1C',  # Red
    'R': '#4DAF4A',  # Green
    'E': '#984EA3',  # Purple
    'D': '#A65628',  # Brown
    'V': '#FF7F00',  # Orange
    'H': '#FFFF33',  # Yellow
    'C': '#F781BF',  # Pink
    'Q': '#999999',  # Gray
    'A': '#8DD3C7',  # Light blue
    'P': '#BEBADA',  # Light purple
    'W': '#FDB462',  # Light orange
    'B': '#B3DE69',  # Light green
}

# Universal color for sum of compartments plot
SUM_COLOR = 'black'

# Greek letters for parameter display
GREEK_LETTERS = {
    'alpha': r'$\alpha$',
    'beta': r'$\beta$',
    'gamma': r'$\gamma$',
    'delta': r'$\delta$',
    'epsilon': r'$\epsilon$',
    'zeta': r'$\zeta$',
    'eta': r'$\eta$',
    'theta': r'$\theta$',
    'iota': r'$\iota$',
    'kappa': r'$\kappa$',
    'lambda': r'$\lambda$',
    'mu': r'$\mu$',
    'nu': r'$\nu$',
    'xi': r'$\xi$',
    'omicron': r'$\omicron$',
    'pi': r'$\pi$',
    'rho': r'$\rho$',
    'sigma': r'$\sigma$',
    'tau': r'$\tau$',
    'upsilon': r'$\upsilon$',
    'phi': r'$\phi$',
    'chi': r'$\chi$',
    'psi': r'$\psi$',
    'omega': r'$\omega$',
}

# Plot style configuration
PLOT_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'figure.dpi': 300,
}
