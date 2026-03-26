"""
Base classes for MCMC sampling infrastructure.

This module defines:
- Sampler: Abstract base class for sampler backends
- SamplerResult: Unified result container for all samplers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Type, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask
    from kl_pipe.sampling.configs import BaseSamplerConfig


@dataclass
class SamplerResult:
    """
    Unified result container for all sampler backends.

    Provides consistent interface regardless of which sampler was used.
    Optional fields are None if not applicable to that sampler type.

    Attributes
    ----------
    samples : np.ndarray
        Posterior samples with shape (n_samples, n_params).
        For ensemble samplers, chains are flattened after burn-in removal.
    log_prob : np.ndarray
        Log posterior probability for each sample, shape (n_samples,).
    param_names : list of str
        Names of sampled parameters in order.
    fixed_params : dict
        Dictionary of fixed parameter values.
    evidence : float, optional
        Log evidence estimate (for nested samplers like nautilus).
    evidence_error : float, optional
        Uncertainty on log evidence.
    chains : np.ndarray, optional
        Full chains before flattening, shape (n_iterations, n_walkers, n_params).
        Only available for ensemble samplers if requested.
    blobs : np.ndarray, optional
        Additional per-sample data from likelihood (emcee blobs).
    acceptance_fraction : float, optional
        Mean acceptance fraction (for MCMC samplers).
    autocorr_time : np.ndarray, optional
        Autocorrelation times per parameter.
    converged : bool
        Whether sampler converged (based on backend-specific diagnostics).
    diagnostics : dict
        Backend-specific diagnostic information.
    metadata : dict
        Additional metadata (timing, backend name, etc.).
    """

    # Core outputs (always present)
    samples: np.ndarray
    log_prob: np.ndarray
    param_names: List[str]
    fixed_params: Dict[str, float]

    # Optional outputs
    evidence: Optional[float] = None
    evidence_error: Optional[float] = None
    chains: Optional[np.ndarray] = None
    blobs: Optional[np.ndarray] = None
    acceptance_fraction: Optional[float] = None
    autocorr_time: Optional[np.ndarray] = None

    # Diagnostics
    converged: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Total number of posterior samples."""
        return self.samples.shape[0]

    @property
    def n_params(self) -> int:
        """Number of sampled parameters."""
        return self.samples.shape[1]

    def get_chain(self, param_name: str) -> np.ndarray:
        """
        Get samples for a specific parameter.

        Parameters
        ----------
        param_name : str
            Name of parameter.

        Returns
        -------
        np.ndarray
            Samples for this parameter, shape (n_samples,).

        Raises
        ------
        KeyError
            If parameter name is not found.
        """
        if param_name in self.param_names:
            idx = self.param_names.index(param_name)
            return self.samples[:, idx]
        elif param_name in self.fixed_params:
            return np.full(self.n_samples, self.fixed_params[param_name])
        else:
            raise KeyError(f"Unknown parameter: {param_name}")

    def get_summary(
        self,
        quantiles: Tuple[float, ...] = (0.16, 0.5, 0.84),
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute summary statistics for each parameter.

        Parameters
        ----------
        quantiles : tuple of float
            Quantiles to compute (default: 16th, 50th, 84th percentiles).

        Returns
        -------
        dict
            Dictionary with parameter names as keys, containing:
            - 'mean': Mean value
            - 'std': Standard deviation
            - 'quantiles': Dict mapping quantile values to computed values
        """
        summary = {}
        for i, name in enumerate(self.param_names):
            chain = self.samples[:, i]
            summary[name] = {
                'mean': float(np.mean(chain)),
                'std': float(np.std(chain)),
                'quantiles': {q: float(np.quantile(chain, q)) for q in quantiles},
            }
        return summary

    def get_rhat(
        self, param_name: Optional[str] = None
    ) -> Optional[Union[float, Dict[str, float]]]:
        """
        Get R-hat (Gelman-Rubin) convergence diagnostic.

        R-hat compares within-chain and between-chain variance. Values close
        to 1.0 indicate convergence. R-hat < 1.01 is typically considered
        good; R-hat > 1.1 suggests poor convergence.

        Requires multiple chains to compute.

        Parameters
        ----------
        param_name : str, optional
            If provided, return R-hat for this parameter only.
            If None, return dict of all parameter R-hats.

        Returns
        -------
        float or dict or None
            R-hat value(s), or None if not available in diagnostics.

        Examples
        --------
        >>> result.get_rhat()
        {'vcirc': 1.002, 'cosi': 1.001, ...}
        >>> result.get_rhat('vcirc')
        1.002
        """
        if 'r_hat' not in self.diagnostics:
            return None
        if param_name is None:
            return self.diagnostics['r_hat']
        return self.diagnostics['r_hat'].get(param_name)

    def get_ess(
        self, param_name: Optional[str] = None
    ) -> Optional[Union[float, Dict[str, float]]]:
        """
        Get effective sample size (ESS).

        ESS estimates the number of independent samples, accounting for
        autocorrelation. Higher is better; ESS > 400 per chain is typically
        sufficient for reliable posterior estimates.

        Parameters
        ----------
        param_name : str, optional
            If provided, return ESS for this parameter only.
            If None, return dict of all parameter ESS values.

        Returns
        -------
        float or dict or None
            ESS value(s), or None if not available in diagnostics.

        Examples
        --------
        >>> result.get_ess()
        {'vcirc': 1523.4, 'cosi': 892.1, ...}
        >>> result.get_ess('vcirc')
        1523.4
        """
        if 'ess' not in self.diagnostics:
            return None
        if param_name is None:
            return self.diagnostics['ess']
        return self.diagnostics['ess'].get(param_name)

    def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Parameters
        ----------
        include_samples : bool
            If True, include full sample arrays. If False, only include
            summary statistics (useful for large sample sets).

        Returns
        -------
        dict
            Serializable dictionary representation.
        """
        result = {
            'param_names': self.param_names,
            'fixed_params': self.fixed_params,
            'n_samples': self.n_samples,
            'n_params': self.n_params,
            'acceptance_fraction': self.acceptance_fraction,
            'evidence': self.evidence,
            'evidence_error': self.evidence_error,
            'converged': self.converged,
            'diagnostics': self.diagnostics,
            'metadata': self.metadata,
            'summary': self.get_summary(),
        }
        if include_samples:
            result['samples'] = self.samples.tolist()
            result['log_prob'] = self.log_prob.tolist()
        return result


class Sampler(ABC):
    """
    Abstract base class for MCMC sampler backends.

    Defines the contract that all sampler implementations must follow.
    Each backend translates the InferenceTask and config to its native API.

    Class Attributes
    ----------------
    requires_gradients : bool
        Does this sampler need gradients? (e.g., HMC/NUTS)
    provides_evidence : bool
        Does this sampler provide evidence estimates? (e.g., nested sampling)
    config_class : Type[BaseSamplerConfig]
        Which config class this sampler expects.

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : BaseSamplerConfig
        Sampler configuration options.
    """

    requires_gradients: bool = False
    provides_evidence: bool = False
    config_class: Type['BaseSamplerConfig'] = None  # Set by subclasses

    def __init__(self, task: 'InferenceTask', config: 'BaseSamplerConfig'):
        self.task = task
        self.config = config
        self._validate()

    def _validate(self) -> None:
        """
        Validate that the task is compatible with this sampler.

        Override in subclasses for backend-specific validation.
        """
        if self.requires_gradients:
            # Verify gradients are computable
            try:
                _ = self.task.get_log_posterior_and_grad_fn()
            except Exception as e:
                raise ValueError(
                    f"{self.__class__.__name__} requires gradients but "
                    f"gradient computation failed: {e}"
                )

        # Validate config type if config_class is specified
        if self.config_class is not None:
            if not isinstance(self.config, self.config_class):
                raise TypeError(
                    f"{self.__class__.__name__} expects config of type "
                    f"{self.config_class.__name__}, got {type(self.config).__name__}"
                )

    @abstractmethod
    def run(self) -> SamplerResult:
        """
        Run the sampler and return results.

        Returns
        -------
        SamplerResult
            Posterior samples and diagnostics.
        """
        pass

    @property
    def name(self) -> str:
        """Name of this sampler backend."""
        return self.__class__.__name__
