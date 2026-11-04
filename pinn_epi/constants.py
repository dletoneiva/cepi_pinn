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

# Color order for parameter grid sweeps (uses compartment palette order)
PARAMETER_GRID_COLORS = list(COMPARTMENT_COLORS.values())

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

# --- Loss-landscape plotting (used by analysis/loss_landscape_plotting.py) ---

# Display symbols for parameter names (values are bare mathtext, no $)
PARAMETER_GREEK_MAP = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
    "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
    "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho",
    "sigma": r"\sigma", "tau": r"\tau", "upsilon": r"\upsilon",
    "phi": r"\phi", "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
    "v": r"\upsilon", "vaccination": r"\upsilon",
    "vaccination_rate": r"\upsilon", "vaccine": r"\upsilon",
    "vaccine_rate": r"\upsilon", "vax": r"\upsilon", "vax_rate": r"\upsilon",
    "waning": r"\nu", "waning_rate": r"\nu",
    "immunity_loss": r"\nu", "immunity_loss_rate": r"\nu",
    "exposure": r"\sigma", "exposure_rate": r"\sigma",
    "incubation": r"\sigma", "incubation_rate": r"\sigma",
    "latent": r"\sigma", "latent_rate": r"\sigma",
    "recovery": r"\gamma", "recovery_rate": r"\gamma",
    "death": r"\delta", "death_rate": r"\delta",
    "mortality": r"\delta", "mortality_rate": r"\delta",
    "fatality": r"\delta", "fatality_rate": r"\delta",
    "birth": r"\mu", "birth_rate": r"\mu",
    "natural_death": r"\mu", "natural_death_rate": r"\mu",
    "population_death": r"\mu", "population_death_rate": r"\mu",
    "transmission": r"\beta", "transmission_rate": r"\beta",
    "contact": r"\beta", "contact_rate": r"\beta",
    "infection": r"\beta", "infection_rate": r"\beta",
    "vaccine_efficacy": r"\epsilon", "efficacy": r"\epsilon",
}

# ODE equations per model, rendered on loss-landscape PDF pages. Each model
# maps to a list of single-line mathtext expressions (no surrounding $
# delimiters) so matplotlib's mathtext parser can render them as a stacked
# block. NOTE: the SIRD entry uses \delta for the death-rate term even though
# SIRDModel.get_derivatives uses the key 'mu'; these strings are display-only
# and are kept as-is for now.
PHYSICS_EQUATION_LATEX = {
    "SI": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dI}{dt} = \beta S I",
    ],
    "SIR": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dI}{dt} = \beta S I - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
    ],
    "SIS": [
        r"\frac{dS}{dt} = -\beta S I + \gamma I",
        r"\frac{dI}{dt} = \beta S I - \gamma I",
    ],
    "SEIR": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
    ],
    "SIRD": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dI}{dt} = \beta S I - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
    ],
    "SIRV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dI}{dt} = \beta S I - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
    "SIRDV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dI}{dt} = \beta S I - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
    "SEIRD": [
        r"\frac{dS}{dt} = -\beta S I",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
    ],
    "SEIRV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
    "SEIRDV": [
        r"\frac{dS}{dt} = -\beta S I - \upsilon S",
        r"\frac{dE}{dt} = \beta S I - \sigma E",
        r"\frac{dI}{dt} = \sigma E - \gamma I - \delta I",
        r"\frac{dR}{dt} = \gamma I",
        r"\frac{dD}{dt} = \delta I",
        r"\frac{dV}{dt} = \upsilon S",
    ],
}

# Color scheme for loss-landscape plots
TRAINER_PLOT_COLORS = {
    'best_curve': '#007AFF',
    'current': 'black',
    'sweep': 'gray',
    'adam': '#377EB8',
    'lbfgs': '#4DAF4A',
    'transition': 'gray',
}

HIGH_CONSPICUITY_PALETTE = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE", "#007AFF",
    "#5856D6", "#AF52DE", "#FF2D55", "#A2845E", "#7FFF00", "#00FFFF",
    "#FF00FF", "#FF8C00", "#32CD32", "#1E90FF", "#9400D3", "#00CED1",
    "#FF1493", "#8A2BE2", "#FF4500", "#00BFFF", "#BA55D3", "#20B2AA",
]

# CSV columns that may never be treated as compartmental parameters
RESERVED_COLUMNS = frozenset({
    'total_loss', 'data_loss', 'physics_loss', 'parameter', 'value',
    'is_current', 'phase', 'epoch',
})

LOSS_LABELS = ("Physics Loss", "Data Loss", "Total Loss")
