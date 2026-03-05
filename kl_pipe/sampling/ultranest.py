"""
UltraNest nested sampler backend.

NOT YET IMPLEMENTED - placeholder for future development.

UltraNest is a robust nested sampling implementation with MLFriends
bounds and excellent performance for high-dimensional problems.

References
----------
Buchner (2021): https://arxiv.org/abs/2101.09604
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import NestedSamplerConfig

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask


class UltraNestSampler(Sampler):
    """
    UltraNest nested sampler backend.

    NOT YET IMPLEMENTED - placeholder for future development.

    UltraNest provides:
    - Robust nested sampling with MLFriends bounds
    - Excellent performance for high-dimensional problems (50+ dimensions)
    - Built-in checkpointing and resume capability
    - Accurate evidence estimates

    When implemented, will use similar interface to NautilusSampler
    with prior_transform pattern.

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : NestedSamplerConfig
        Sampler configuration options.

    Raises
    ------
    NotImplementedError
        Always raised - this sampler is not yet implemented.

    See Also
    --------
    NautilusSampler : Currently implemented nested sampler alternative.

    Notes
    -----
    To contribute an implementation:
    1. Install ultranest: `pip install ultranest`
    2. Implement run() method following NautilusSampler pattern
    3. Build prior_transform from PriorDict
    4. Handle checkpointing via config.log_dir and config.resume
    """

    requires_gradients = False
    provides_evidence = True
    config_class = NestedSamplerConfig

    def __init__(self, task: 'InferenceTask', config: NestedSamplerConfig):
        raise NotImplementedError(
            "UltraNestSampler is not yet implemented.\n\n"
            "Please use NautilusSampler for nested sampling:\n"
            "  >>> from kl_pipe.sampling import build_sampler, NestedSamplerConfig\n"
            "  >>> config = NestedSamplerConfig(n_live=500)\n"
            "  >>> sampler = build_sampler('nautilus', task, config)\n\n"
            "Or contribute an implementation! See the docstring for guidance."
        )

    def run(self) -> SamplerResult:
        """
        Run UltraNest sampler.

        Raises
        ------
        NotImplementedError
            Always raised - not yet implemented.
        """
        raise NotImplementedError("UltraNestSampler is not yet implemented.")
