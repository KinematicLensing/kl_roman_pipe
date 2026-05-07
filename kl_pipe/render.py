"""
Rendering configuration for k-space intensity model rendering.

``RenderConfig`` controls FFT grid sizing: oversample factor, padding,
and folding/maxk thresholds. It can be auto-computed from model
parameters (standalone renders) or pre-built from prior bounds
(JIT-compatible inference).

The primary use case is inference with a static PSF and pixel response,
where the grid is determined once at setup time from the worst-case
prior bounds and frozen for all MCMC samples.

Trade-off: smaller ``cosi`` in priors → larger worst-case maxk →
larger FFT grid → slower inference per sample. Document this when
setting up inference tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax
import numpy as np

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel
    from kl_pipe.pixel import PixelResponse
    from kl_pipe.priors import PriorDict


@dataclass(frozen=True)
class RenderConfig:
    """K-space rendering grid parameters.

    Controls FFT grid sizing for intensity model rendering. Auto-computed
    from ``model.maxk``/``model.stepk`` when not provided. Pre-built from
    prior bounds for JIT-compatible inference.

    Parameters
    ----------
    oversample : int
        K-grid oversampling factor. Extends effective Nyquist to
        ``oversample × π / pixel_scale``. Derived from ``effective_maxk``
        when using factory methods.
    pad_factor : int
        FFT zero-padding factor for anti-aliasing. Default 2.
    folding_threshold : float
        Maximum flux fraction allowed to fold (alias) due to DFT periodic
        boundaries. Controls ``stepk``. Default 5e-3.
    maxk_threshold : float
        FT amplitude threshold for maxk computation. Default 1e-3.
    effective_maxk : float, optional
        Computed effective maxk for the full rendering chain
        (profile × pixel × PSF). None if not yet computed.
    stepk : float, optional
        Computed minimum k-spacing for flux containment. None if not yet
        computed.
    """

    oversample: int = 1
    pad_factor: int = 2
    folding_threshold: float = 5e-3
    maxk_threshold: float = 1e-3
    effective_maxk: Optional[float] = None
    stepk: Optional[float] = None

    @classmethod
    def for_model(
        cls,
        model: 'IntensityModel',
        params: dict,
        pixel_scale: float,
        pixel_response: 'PixelResponse' = None,
        psf=None,
        folding_threshold: float = 5e-3,
        maxk_threshold: float = 1e-3,
    ) -> 'RenderConfig':
        """Compute RenderConfig from specific model parameters.

        For standalone renders where theta is known. Computes maxk/stepk
        from the given params and derives oversample/pad_factor.

        Parameters
        ----------
        model : IntensityModel
            Model with ``maxk`` and ``stepk`` methods.
        params : dict
            Profile parameters (must include scale length and ``cosi``).
        pixel_scale : float
            Coarse pixel scale (arcsec/pixel).
        pixel_response : PixelResponse, optional
            Pixel response function.
        psf : galsim.GSObject, optional
            PSF profile.
        folding_threshold : float
            Max flux folding fraction. Default 5e-3.
        maxk_threshold : float
            FT amplitude threshold. Default 1e-3.
        """
        eff_maxk = compute_effective_maxk(
            model,
            params,
            pixel_response=pixel_response,
            psf=psf,
            threshold=maxk_threshold,
        )
        stepk = model.stepk(params, folding_threshold=folding_threshold)
        oversample = _oversample_from_maxk(eff_maxk, pixel_scale)

        return cls(
            oversample=oversample,
            pad_factor=2,
            folding_threshold=folding_threshold,
            maxk_threshold=maxk_threshold,
            effective_maxk=eff_maxk,
            stepk=stepk,
        )

    @classmethod
    def for_priors(
        cls,
        model: 'IntensityModel',
        priors: 'PriorDict',
        pixel_scale: float,
        pixel_response: 'PixelResponse' = None,
        psf=None,
        folding_threshold: float = 5e-3,
        maxk_threshold: float = 1e-3,
    ) -> 'RenderConfig':
        """Compute worst-case RenderConfig from prior bounds.

        For JIT-compatible inference. Evaluates at the prior extremes
        that produce the most demanding grid: smallest scale length ×
        smallest cosi (highest maxk), largest scale length (smallest stepk).

        Parameters
        ----------
        model : IntensityModel
            Model with ``maxk`` and ``stepk`` methods.
        priors : PriorDict
            Prior specifications.
        pixel_scale : float
            Coarse pixel scale.
        pixel_response : PixelResponse, optional
            Pixel response function.
        psf : galsim.GSObject, optional
            PSF profile.
        folding_threshold : float
            Max flux folding. Default 5e-3.
        maxk_threshold : float
            FT amplitude threshold. Default 1e-3.
        """
        worst_maxk_params, worst_stepk_params = _extract_worst_case_params(
            model, priors
        )

        try:
            eff_maxk = compute_effective_maxk(
                model,
                worst_maxk_params,
                pixel_response=pixel_response,
                psf=psf,
                threshold=maxk_threshold,
            )
        except (KeyError, NotImplementedError):
            eff_maxk = np.pi / pixel_scale  # fallback: Nyquist

        try:
            stepk = model.stepk(worst_stepk_params, folding_threshold=folding_threshold)
        except (KeyError, NotImplementedError):
            stepk = np.pi / (5.0 * pixel_scale)  # fallback

        oversample = _oversample_from_maxk(eff_maxk, pixel_scale)

        return cls(
            oversample=oversample,
            pad_factor=2,
            folding_threshold=folding_threshold,
            maxk_threshold=maxk_threshold,
            effective_maxk=eff_maxk,
            stepk=stepk,
        )

    def __repr__(self) -> str:
        parts = [f'oversample={self.oversample}', f'pad_factor={self.pad_factor}']
        if self.effective_maxk is not None:
            parts.append(f'effective_maxk={self.effective_maxk:.1f}')
        if self.stepk is not None:
            parts.append(f'stepk={self.stepk:.4f}')
        return f"RenderConfig({', '.join(parts)})"


# ============================================================================
# JAX pytree registration
# ============================================================================
# All fields are static aux — no traced children.


def _render_config_flatten(rc):
    return (), (
        rc.oversample,
        rc.pad_factor,
        rc.folding_threshold,
        rc.maxk_threshold,
        rc.effective_maxk,
        rc.stepk,
    )


def _render_config_unflatten(aux, children):
    return RenderConfig(
        oversample=aux[0],
        pad_factor=aux[1],
        folding_threshold=aux[2],
        maxk_threshold=aux[3],
        effective_maxk=aux[4],
        stepk=aux[5],
    )


jax.tree_util.register_pytree_node(
    RenderConfig, _render_config_flatten, _render_config_unflatten
)


# ============================================================================
# Grid computation utilities
# ============================================================================


def compute_effective_maxk(
    model,
    params: dict,
    pixel_response=None,
    psf=None,
    threshold: float = 1e-3,
) -> float:
    """Compute effective maxk for the full rendering chain.

    Scans wavenumbers from 0 to ``model.maxk(params, threshold)`` and
    returns the largest k where the product
    ``|profile_FT(k) × pixel_FT(k) × PSF_FT(k)|`` remains above ``threshold``.

    Includes whatever factors are present (profile + any of pixel/PSF).
    The product approach is correct for grid sizing: at the min of any two
    individual maxks, both factors are at most threshold, so their product
    is at most threshold² — meaning the product crosses below threshold
    well *before* either single factor.

    PSF FT evaluated at ``(kx=0, ky=k)`` -- assumes radially symmetric PSF
    (true for all common GalSim PSFs: Gaussian, Moffat, Airy without
    anisotropy). For asymmetric PSFs, this is the radial slice along the
    +ky direction, a defensible bandlimit proxy.

    Parameters
    ----------
    model : IntensityModel
        Model with ``maxk(params, threshold)`` and ``_ft_envelope(k, params)``
        methods.
    params : dict
        Profile parameters (must include scale length and ``cosi``).
    pixel_response : PixelResponse, optional
        Any subclass implementing ``ft_radial(k)``.
    psf : galsim.GSObject, optional
        PSF profile. Folded into the product scan via ``psf.kValue``.
    threshold : float
        FT amplitude threshold.

    Returns
    -------
    float
        Effective maxk in rad/arcsec.
    """
    # bare profile maxk as scan upper bound
    profile_maxk = model.maxk(params, threshold=threshold)

    if not hasattr(model, '_ft_envelope'):
        # model doesn't provide forward FT eval; fall back to bare maxk
        # (further tightened by PSF if present).
        if psf is not None:
            profile_maxk = min(profile_maxk, float(psf.maxk))
        return profile_maxk

    n_scan = 500
    # scan from 0 (not 0.1) so degenerate small-profile cases don't return
    # 0.1 > profile_maxk.
    k_scan = np.linspace(0.0, profile_maxk, n_scan)

    # build the product scan from whichever factors are present
    product = np.array([model._ft_envelope(k, params) for k in k_scan])

    if pixel_response is not None:
        product = product * np.asarray(pixel_response.ft_radial(k_scan))

    if psf is not None:
        import galsim

        psf_ft = np.array(
            [abs(psf.kValue(galsim.PositionD(0.0, float(k)))) for k in k_scan]
        )
        product = product * psf_ft

    above = k_scan[product > threshold]
    if len(above) == 0:
        # nothing crosses threshold (shouldn't happen at k=0 where DC is 1)
        # but if it does, return 0 as a safe upper bound that won't oversize.
        return 0.0
    return float(above[-1])


def _oversample_from_maxk(effective_maxk: float, pixel_scale: float) -> int:
    """Derive oversample factor from effective maxk and pixel scale.

    Returns the smallest odd integer oversample such that the effective
    Nyquist frequency ``oversample × π / pixel_scale`` exceeds maxk.
    """
    k_nyquist = np.pi / pixel_scale
    if effective_maxk <= k_nyquist:
        return 1
    oversample = int(np.ceil(effective_maxk / k_nyquist))
    if oversample % 2 == 0:
        oversample += 1
    return oversample


def _extract_worst_case_params(model, priors) -> tuple:
    """Extract worst-case params from priors for maxk and stepk.

    For maxk: smallest scale length × smallest cosi (most compact,
    most inclined → highest maxk).
    For stepk: largest scale length (most extended → smallest stepk).

    Returns
    -------
    worst_maxk_params : dict
    worst_stepk_params : dict
    """
    param_names = model.PARAMETER_NAMES
    worst_maxk_params = {}
    worst_stepk_params = {}

    for name in param_names:
        spec = priors._param_spec.get(name)
        if spec is None:
            continue

        # fixed scalar
        if isinstance(spec, (int, float)):
            worst_maxk_params[name] = float(spec)
            worst_stepk_params[name] = float(spec)
            continue

        prior = spec
        if hasattr(prior, 'low') and hasattr(prior, 'high'):
            if name in ('int_rscale', 'int_hlr'):
                # smallest scale → highest maxk
                worst_maxk_params[name] = prior.low
                worst_stepk_params[name] = prior.high
            elif name == 'cosi':
                # smallest cosi (most edge-on) → highest maxk
                worst_maxk_params[name] = prior.low
                worst_stepk_params[name] = prior.low
            elif name == 'n_sersic':
                # highest n → slowest FT decay → highest maxk
                worst_maxk_params[name] = prior.high
                worst_stepk_params[name] = prior.high
            elif name == 'nu':
                # most negative nu → slowest FT decay → highest maxk
                worst_maxk_params[name] = prior.low
                worst_stepk_params[name] = prior.low
            else:
                mid = 0.5 * (prior.low + prior.high)
                worst_maxk_params[name] = mid
                worst_stepk_params[name] = mid
        else:
            # non-uniform prior: use mean if available
            if hasattr(prior, 'mean'):
                worst_maxk_params[name] = prior.mean
                worst_stepk_params[name] = prior.mean

    return worst_maxk_params, worst_stepk_params
