from __future__ import annotations

from typing import Optional
import datetime
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pinn_epi.constants import PLOT_STYLE, GREEK_LETTERS, COMPARTMENT_COLORS


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def plot_compartmental_solution(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    compartment_colors: Optional[dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    model_params: Optional[dict] = None,
    initial_conditions: Optional[list] = None,
    model_type: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:

    apply_plot_style()
    compartment_names = list(trajectories.keys())

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
    else:
        fig = ax.get_figure()

    if compartment_colors is None:
        compartment_colors = COMPARTMENT_COLORS

    for name in compartment_names:
        color = compartment_colors.get(name, f'C{compartment_names.index(name)}')
        ax.plot(t, trajectories[name], label=f'{name}(t)', color=color, lw=2.5)

    ax.set_xlabel('Time (days)', fontsize=PLOT_STYLE['axes.labelsize'], labelpad=15)
    ax.set_ylabel('Fraction of Population', fontsize=PLOT_STYLE['axes.labelsize'], labelpad=15)
    
    ax.tick_params(axis='x', labelsize=PLOT_STYLE['xtick.labelsize'])
    ax.tick_params(axis='y', labelsize=PLOT_STYLE['ytick.labelsize'])

    title = f"{model_type} Differential Equation Solutions"
    ax.set_title(title, pad=60, fontsize=PLOT_STYLE['figure.titlesize'])

    if created_fig:
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=len(compartment_names), 
                  columnspacing=1.5, handletextpad=0.5, fontsize=PLOT_STYLE['legend.fontsize'])
        plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.25)

    if model_params is not None and initial_conditions is not None:
        param_items = []
        for k, v in model_params.items():
            latex_k = GREEK_LETTERS.get(k, k)
            param_items.append(f"{latex_k}={v}")
        
        param_str = ", ".join(param_items)
        y0_items = []
        for i, name in enumerate(compartment_names):
            y0_items.append(f"{name}(0)={initial_conditions[i]:.2f}")
        y0_str = ", ".join(y0_items)
        info_text = f"Parameters: {param_str} | Initial: {y0_str}"
        
        wrapped_text = textwrap.fill(info_text, width=80)
        lines = wrapped_text.count('\n') + 1
        
        font_size = 14
        if lines > 2:
            font_size = max(10, 14 - (lines - 2) * 2)
        
        ax.text(0.5, -0.25, wrapped_text, ha='center', va='top', fontsize=font_size,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                transform=ax.transAxes)

    if save_path:
        fig.savefig(save_path, dpi=600)
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_actual_vs_predicted(
    actual_data: dict[str, np.ndarray], 
    predicted_data: dict[str, np.ndarray],
    t_data: np.ndarray,
    compartment_colors: Optional[dict[str, str]] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    plot_config: Optional[dict] = None,
    model_compartments: Optional[list] = None,
    training_config: Optional[dict] = None,
    observed_variables: Optional[list] = None,
    network_config: Optional[dict] = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a stacked plot comparing actual vs predicted compartments/parameters.

    Args:
        actual_data: Dictionary containing actual compartment/parameter values over time
        predicted_data: Dictionary containing predicted compartment/parameter values over time
        t_data: Time array
        compartment_colors: Color mapping for compartments
        show: Whether to display the plot
        save_path: Path to save the plot
        plot_config: Configuration specifying what to plot
        model_compartments: Ordered list of compartment names from the model
        training_config: Training configuration for summary text
        observed_variables: List of observed variables from the observables config
        network_config: Network configuration for architecture information

    Returns:
        Figure and axes objects
    """
    apply_plot_style()
    
    if compartment_colors is None:
        compartment_colors = COMPARTMENT_COLORS

    # Determine which items to plot based on plot_config
    compartments_to_plot = []
    if plot_config and "compartments_to_plot" in plot_config:
        compartments_to_plot = plot_config["compartments_to_plot"]
    else:
        # Get all keys from both datasets
        all_actual_keys = set(actual_data.keys())
        all_predicted_keys = set(predicted_data.keys())
        # Find intersection of keys that exist in both datasets
        common_keys = all_actual_keys.intersection(all_predicted_keys)
        
        if model_compartments:
            # Prioritize the order from model compartments if available
            compartments_to_plot = [comp for comp in model_compartments if comp in common_keys]
            # Add any remaining common keys not in model_compartments
            remaining_keys = [comp for comp in sorted(common_keys) if comp not in compartments_to_plot]
            compartments_to_plot.extend(remaining_keys)
        else:
            # Sort common keys alphabetically if no specific ordering is provided
            compartments_to_plot = sorted(list(common_keys))

    # Validation: check if specified compartments exist in both datasets
    filtered_compartments = []
    for comp in compartments_to_plot:
        if comp in actual_data and comp in predicted_data:
            filtered_compartments.append(comp)
        elif comp in actual_data or comp in predicted_data:
            # Include compartment if it exists in either dataset and warn about missing parts
            if comp not in actual_data:
                print(f"Warning: '{comp}' found in predicted data but not in actual data")
            elif comp not in predicted_data:
                print(f"Warning: '{comp}' found in actual data but not in predicted data")

    compartments_to_plot = filtered_compartments

    if not compartments_to_plot:
        # If no common keys, try to plot what exists in each
        all_actual_keys = set(actual_data.keys())
        all_predicted_keys = set(predicted_data.keys())
        
        # Get compartment names in model order if available
        if model_compartments:
            available_compartments = [comp for comp in model_compartments if comp in all_actual_keys or comp in all_predicted_keys]
        else:
            # Combine both sets of keys preserving order
            available_compartments = sorted(list(all_actual_keys.union(all_predicted_keys)))
        
        compartments_to_plot = []
        for comp in available_compartments:
            if comp in all_actual_keys and comp in all_predicted_keys:
                compartments_to_plot.append(comp)
    
    if not compartments_to_plot:
        raise ValueError("No compartments found to plot - check that actual and predicted data have common keys")

    # Create subplots - one for each compartment to plot + one for sum of compartments
    n_comps = len(compartments_to_plot)
    fig, axes = plt.subplots(n_comps + 1, 1, figsize=(12, 4 * (n_comps + 1)), dpi=300)

    # Handle case where there's only one subplot
    if n_comps + 1 == 1:
        axes = [axes]

    # Plot each compartment
    for idx, comp in enumerate(compartments_to_plot):
        # Only plot if the compartment exists in both datasets
        if comp in actual_data and comp in predicted_data:
            actual_values = actual_data[comp]
            predicted_values = predicted_data[comp]
            color = compartment_colors.get(comp, f'C{idx}')

            # Plot actual data as solid line (Ground Truth)
            axes[idx].plot(t_data, actual_values, label=f'Ground Truth {comp}', color=color, lw=2, linestyle='-', zorder=2)

            # Plot predicted data as dashed line with different style
            axes[idx].plot(t_data, predicted_values, label=f'Predicted {comp}', color=color, lw=3, linestyle='--', zorder=3)

            axes[idx].set_ylabel(f'{comp}', fontsize=PLOT_STYLE['axes.labelsize'])
            axes[idx].tick_params(axis='x', labelsize=PLOT_STYLE['xtick.labelsize'])
            axes[idx].tick_params(axis='y', labelsize=PLOT_STYLE['ytick.labelsize'])

            # Add legend to each subplot, positioned above the chart
            axes[idx].legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, 
                             columnspacing=1.5, handletextpad=0.5, fontsize=PLOT_STYLE['legend.fontsize'])
        else:
            # If compartment only exists in one dataset, just note it in the axis
            axes[idx].text(0.5, 0.5, f'Compartment {comp} data incomplete', 
                          horizontalalignment='center', verticalalignment='center', 
                          transform=axes[idx].transAxes, fontsize=14)
            axes[idx].set_ylabel(f'{comp}', fontsize=PLOT_STYLE['axes.labelsize'])

    # Plot sum of all compartments
    sum_actual = np.zeros_like(t_data)
    sum_predicted = np.zeros_like(t_data)
    
    for comp in compartments_to_plot:
        if comp in actual_data:
            sum_actual += actual_data[comp]
        if comp in predicted_data:
            sum_predicted += predicted_data[comp]
    
    # Universal color for sum plot
    sum_color = 'black'
    axes[n_comps].plot(t_data, sum_actual, label='Ground Truth Sum', color=sum_color, lw=2, linestyle='-', zorder=2)
    axes[n_comps].plot(t_data, sum_predicted, label='Predicted Sum', color=sum_color, lw=3, linestyle='--', zorder=3)
    axes[n_comps].set_ylabel('Sum of Compartments', fontsize=PLOT_STYLE['axes.labelsize'])
    axes[n_comps].tick_params(axis='x', labelsize=PLOT_STYLE['xtick.labelsize'])
    axes[n_comps].tick_params(axis='y', labelsize=PLOT_STYLE['ytick.labelsize'])
    axes[n_comps].legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, 
                         columnspacing=1.5, handletextpad=0.5, fontsize=PLOT_STYLE['legend.fontsize'])

    # Set x-label for the bottom subplot
    axes[-1].set_xlabel('Time', fontsize=PLOT_STYLE['axes.labelsize'])

    fig.suptitle('Actual vs Predicted Network Prediction Comparison', fontsize=PLOT_STYLE['figure.titlesize'])

    # Add training configuration summary and network architecture at the bottom
    summary_items = []
    
    if training_config:
        # Extract relevant parameters for summary
        data_weight = training_config.get('data_weight', 1.0)
        physics_weight = training_config.get('physics_weight', 1.0)
        adam_epochs = training_config.get('adam_epochs', 1000)
        adam_lr = training_config.get('adam_lr', 0.001)
        lbfgs_max_iter = training_config.get('lbfgs_max_iter', 50)
        n_collocation_points = training_config.get('n_collocation_points', 100)
        
        # Format learnable parameters
        learnable_params = training_config.get('learnable_parameters', [])
        params_str = ', '.join(learnable_params) if learnable_params else 'None'
        
        training_info = f"Loss: Data={data_weight}, Physics={physics_weight} | " \
                       f"Adam: {adam_epochs}ep, lr={adam_lr} | " \
                       f"LBFGS: {lbfgs_max_iter}iter | " \
                       f"Collocation: {n_collocation_points}pts | " \
                       f"Learned: {params_str}"
        summary_items.append(training_info)
    
    if observed_variables:
        obs_str = ','.join(observed_variables)
        observed_info = f"Observed: [{obs_str}]"
        summary_items.append(observed_info)
        
    if network_config:
        # Add network architecture information
        backbone_info = f"Backbone: {network_config.get('backbone', {}).get('type', 'BaseMLP')}"
        encoder_info = f"Encoder: {network_config.get('encoder', 'None')}"
        head_info = f"Head: {network_config.get('head', 'None')}"
        architecture_info = f"Architecture: {backbone_info}, {encoder_info}, {head_info}"
        summary_items.append(architecture_info)
    
    if summary_items:
        summary_text = " | ".join(summary_items)
        # Wrap text to fit the figure
        wrapped_summary = textwrap.fill(summary_text, width=80)
        # Add the summary text closer to the figure
        fig.text(0.5, 0.01, wrapped_summary, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 transform=fig.transFigure)

    # Adjust subplot spacing to accommodate legends and summary text
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1, hspace=0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved network prediction plot → {save_path}")

    if show:
        plt.show()

    return fig, axes
