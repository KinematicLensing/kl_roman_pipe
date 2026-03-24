"""Diagnostic plotting utilities.

Re-exports from submodules for backward compatibility.
Prefer explicit submodule imports (e.g., ``from kl_pipe.diagnostics.imaging import ...``).
"""

from kl_pipe.diagnostics.imaging import (
    compute_joint_nsigma,
    nsigma_to_color,
    plot_data_comparison_panels,
    plot_combined_data_comparison,
    plot_parameter_recovery,
)
from kl_pipe.diagnostics.datacube import plot_datacube_overview
from kl_pipe.diagnostics.grism import (
    plot_grism_overview,
    plot_dispersion_angles,
    plot_dispersion_angle_study,
)

__all__ = [
    # imaging
    'compute_joint_nsigma',
    'nsigma_to_color',
    'plot_data_comparison_panels',
    'plot_combined_data_comparison',
    'plot_parameter_recovery',
    # datacube
    'plot_datacube_overview',
    # grism
    'plot_grism_overview',
    'plot_dispersion_angles',
    'plot_dispersion_angle_study',
]
