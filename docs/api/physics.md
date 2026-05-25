# Physics Models API

This module contains implementations of various compartmental epidemiological models. Each model inherits from the `CompartmentalModel` abstract base class.

## CompartmentalModel

::: pinn_epi.models.physics.CompartmentalModel
    rendering:
      show_root_heading: false
      show_source: true

### Abstract Methods

#### compartment_names
Property that returns the ordered list of compartment names.

#### get_derivatives
Computes the time derivatives of all compartments.

#### get_observables
Computes observable quantities from the model state. Default implementation returns base compartments.

## SIRModel

::: pinn_epi.models.physics.SIRModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'I', 'R']`

### Methods

#### get_derivatives
Implements the SIR model equations:
- dS/dt = -βSI
- dI/dt = βSI - γI
- dR/dt = γI

Where:
- S: Susceptible population
- I: Infected population
- R: Recovered population
- β: Infection rate
- γ: Recovery rate

#### get_observables
Returns base compartments plus daily new cases: βSI

## SEIRModel

::: pinn_epi.models.physics.SEIRModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'E', 'I', 'R']`

### Methods

#### get_derivatives
Implements the SEIR model equations:
- dS/dt = -βSI
- dE/dt = βSI - σE
- dI/dt = σE - γI
- dR/dt = γI

Where:
- S: Susceptible population
- E: Exposed population (infected but not yet infectious)
- I: Infected population
- R: Recovered population
- β: Infection rate
- σ: Rate of progression from exposed to infected
- γ: Recovery rate

## SIModel

::: pinn_epi.models.physics.SIModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'I']`

### Methods

#### get_derivatives
Implements the SI model equations (no recovery):
- dS/dt = -βSI
- dI/dt = βSI

Where:
- S: Susceptible population
- I: Infected population
- β: Infection rate

## SISModel

::: pinn_epi.models.physics.SISModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'I']`

### Methods

#### get_derivatives
Implements the SIS model equations:
- dS/dt = -βSI + γI
- dI/dt = βSI - γI

Where:
- S: Susceptible population
- I: Infected population
- β: Infection rate
- γ: Recovery rate (individuals become susceptible again)

## SIRVModel

::: pinn_epi.models.physics.SIRVModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'I', 'R', 'V']`

### Methods

#### get_derivatives
Implements the SIRV model equations:
- dS/dt = -βSI - νS
- dI/dt = βSI - γI
- dR/dt = γI
- dV/dt = νS

Where:
- S: Susceptible population
- I: Infected population
- R: Recovered population
- V: Vaccinated population
- β: Infection rate
- γ: Recovery rate
- ν: Vaccination rate

## SIRDModel

::: pinn_epi.models.physics.SIRDModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'I', 'R', 'D']`

### Methods

#### get_derivatives
Implements the SIRD model equations:
- dS/dt = -βSI
- dI/dt = βSI - (γ + μ)I
- dR/dt = γI
- dD/dt = μI

Where:
- S: Susceptible population
- I: Infected population
- R: Recovered population
- D: Dead population
- β: Infection rate
- γ: Recovery rate
- μ: Death rate

## SIRDVModel

::: pinn_epi.models.physics.SIRDVModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'I', 'R', 'D', 'V']`

### Methods

#### get_derivatives
Implements the SIRDV model equations:
- dS/dt = -βSI - νS
- dI/dt = βSI - (γ + μ)I
- dR/dt = γI
- dD/dt = μI
- dV/dt = νS

Where:
- S: Susceptible population
- I: Infected population
- R: Recovered population
- D: Dead population
- V: Vaccinated population
- β: Infection rate
- γ: Recovery rate
- μ: Death rate
- ν: Vaccination rate

## SEIRDModel

::: pinn_epi.models.physics.SEIRDModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'E', 'I', 'R', 'D']`

### Methods

#### get_derivatives
Implements the SEIRD model equations:
- dS/dt = -βSI
- dE/dt = βSI - σE
- dI/dt = σE - (γ + μ)I
- dR/dt = γI
- dD/dt = μI

Where:
- S: Susceptible population
- E: Exposed population
- I: Infected population
- R: Recovered population
- D: Dead population
- β: Infection rate
- σ: Rate of progression from exposed to infected
- γ: Recovery rate
- μ: Death rate

## SEIRVModel

::: pinn_epi.models.physics.SEIRVModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'E', 'I', 'R', 'V']`

### Methods

#### get_derivatives
Implements the SEIRV model equations:
- dS/dt = -βSI - νS
- dE/dt = βSI - σE
- dI/dt = σE - γI
- dR/dt = γI
- dV/dt = νS

Where:
- S: Susceptible population
- E: Exposed population
- I: Infected population
- R: Recovered population
- V: Vaccinated population
- β: Infection rate
- σ: Rate of progression from exposed to infected
- γ: Recovery rate
- ν: Vaccination rate

## SEIRDVModel

::: pinn_epi.models.physics.SEIRDVModel
    rendering:
      show_root_heading: false
      show_source: true

### Properties

#### compartment_names
Returns `['S', 'E', 'I', 'R', 'D', 'V']`

### Methods

#### get_derivatives
Implements the SEIRDV model equations:
- dS/dt = -βSI - νS
- dE/dt = βSI - σE
- dI/dt = σE - (γ + μ)I
- dR/dt = γI
- dD/dt = μI
- dV/dt = νS

Where:
- S: Susceptible population
- E: Exposed population
- I: Infected population
- R: Recovered population
- D: Dead population
- V: Vaccinated population
- β: Infection rate
- σ: Rate of progression from exposed to infected
- γ: Recovery rate
- μ: Death rate
- ν: Vaccination rate
