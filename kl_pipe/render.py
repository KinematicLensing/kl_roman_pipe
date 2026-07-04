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
    from kl_pipe.dispersion import GrismPars
    from kl_pipe.model import IntensityModel
    from kl_pipe.parameters import ImagePars
    from kl_pipe.pixel import PixelResponse
    from kl_pipe.priors import PriorDict
    from kl_pipe.source import SourceModel


@dataclass(frozen=True)
class RenderConfig:
    """K-space rendering grid parameters.

    Controls FFT grid sizing for intensity model rendering. Auto-computed
    from ``model.maxk``/``model.stepk`` when not provided. Pre-built from
    prior bounds for JIT-compatible inference.

    Parameters
    ----------
    oversample : int
        K-grid oversampling factor (spatial). Extends effective Nyquist to
        ``oversample × π / pixel_scale``. Derived from ``effective_maxk``
        when using factory methods.
    pad_factor : int
        FFT zero-padding factor for anti-aliasing. Default 2.
    folding_threshold : float
        Maximum flux fraction allowed to fold (alias) due to DFT periodic
        boundaries. Controls ``stepk``. Default 5e-3.
    maxk_threshold : float
        FT amplitude threshold for maxk computation. Default 1e-3.
    spectral_oversample : int
        Wavelength sub-bin count for cube assembly in
        ``SourceModel.build_cube`` / ``render_grism``. Default 15; the
        former default of 5 was found to produce parameter bias 5-25x
        ``sigma_Fisher`` at SNR=10000 for spectrally-sensitive params
        (see ``docs/oversampling_convergence.md``). osf=15 reduces that to
        ~sigma_Fisher; cube-pixel convergence at osf=15 vs osf=25 is
        <1e-3. Cost scales linearly with osf. Only consulted when the
        obs (or render_grism caller) does not pass an explicit value,
        and only when ``spectral_method='oversample'``. Applies to
        grism obs only; ignored for image/velocity obs that carry
        RenderConfig.
    spectral_method : str
        Spectral bin-integration method for cube assembly: ``'erf'``
        (default; exact analytic Gaussian bin integrals, no spectral
        discretization error, ``spectral_oversample`` ignored) or
        ``'oversample'`` (midpoint fine-grid sampling controlled by
        ``spectral_oversample``; retained for convergence/comparison
        studies). Applies to grism obs only.
    psf_mode : str
        Where the (wavelength-independent) PSF convolution is applied in
        the grism pathway: ``'post_dispersion'`` (default; disperse the
        raw cube, then one exact padded convolution of the 2D dispersed
        image — mathematically identical up to bounded stamp-boundary
        truncation terms) or
        ``'per_slice'`` (convolve every wavelength slice before
        dispersion; the general reference path, required for future
        wavelength-dependent PSFs). Applies to grism obs only.
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
    spectral_oversample: int = 15
    spectral_method: str = 'erf'
    psf_mode: str = 'post_dispersion'
    effective_maxk: Optional[float] = None
    stepk: Optional[float] = None

    def __post_init__(self):
        if self.spectral_method not in ('erf', 'oversample'):
            raise ValueError(
                f"spectral_method must be 'erf' or 'oversample', got "
                f"{self.spectral_method!r}"
            )
        if self.psf_mode not in ('post_dispersion', 'per_slice'):
            raise ValueError(
                f"psf_mode must be 'post_dispersion' or 'per_slice', got "
                f"{self.psf_mode!r}"
            )

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

    @classmethod
    def for_grism_priors(
        cls,
        intensity_model: 'IntensityModel',
        velocity_model: 'VelocityModel',
        intensity_priors: 'PriorDict',
        velocity_priors: 'PriorDict',
        sigma_v_min: float,
        coarse_pixel_scale: float,
        psf=None,
        folding_threshold: float = 5e-3,
        maxk_threshold: float = 1e-3,
    ) -> 'RenderConfig':
        """Compute worst-case ``RenderConfig`` for one emission line's grism
        cube fine-grid sizing.

        Uses the Minkowski-sum bandwidth bound derived in
        ``docs/notes/grism_cube_bandwidth.tex``: the cube-slice
        ``M = I * G`` has support up to ``maxk(I) + maxk(G)`` (Minkowski sum
        of supports of the convolved FTs), bounded above by the PSF cutoff
        when PSF is present. No pixel-response factor (post-dispersion only).

        Parameters
        ----------
        intensity_model : IntensityModel
            Emission line's spatial intensity model.
        velocity_model : VelocityModel
            Velocity model providing ``grad_bandwidth(params)``.
        intensity_priors : PriorDict
            Prior specifications keyed by ``intensity_model.PARAMETER_NAMES``.
        velocity_priors : PriorDict
            Prior specifications keyed by ``velocity_model.PARAMETER_NAMES``.
        sigma_v_min : float
            Lower prior bound on intrinsic kinematic dispersion (km/s). The
            worst-case (narrowest line) value.
        coarse_pixel_scale : float
            Coarse detector pixel scale (arcsec).
        psf : galsim.GSObject, optional
            PSF profile for per-slice damping. Passed through to
            ``compute_effective_maxk_grism``.
        folding_threshold : float
            Max flux folding for ``stepk`` (default 5e-3); same convention as
            ``for_priors``.
        maxk_threshold : float
            FT amplitude threshold for the maxk product walk (default 1e-3).

        Returns
        -------
        RenderConfig
            With ``oversample``, ``effective_maxk``, ``stepk`` derived from
            the worst-case bandwidth.
        """
        int_worst_maxk, int_worst_stepk = _extract_worst_case_params(
            intensity_model, intensity_priors
        )
        vel_worst, _ = _extract_worst_case_params(velocity_model, velocity_priors)

        try:
            grad_v_max = velocity_model.grad_bandwidth(vel_worst)
        except KeyError:
            # velocity priors missing one of (vcirc, rscale, cosi) -- assume
            # no velocity-modulation bandwidth contribution
            grad_v_max = 0.0

        try:
            eff_maxk = compute_effective_maxk_grism(
                intensity_model,
                int_worst_maxk,
                sigma_v_min,
                grad_v_max,
                psf=psf,
                threshold=maxk_threshold,
            )
        except (KeyError, NotImplementedError):
            eff_maxk = np.pi / coarse_pixel_scale  # fallback: coarse Nyquist

        try:
            stepk = intensity_model.stepk(
                int_worst_stepk, folding_threshold=folding_threshold
            )
        except (KeyError, NotImplementedError):
            stepk = np.pi / (5.0 * coarse_pixel_scale)  # fallback

        oversample = _oversample_from_maxk(eff_maxk, coarse_pixel_scale)

        return cls(
            oversample=oversample,
            pad_factor=2,
            folding_threshold=folding_threshold,
            maxk_threshold=maxk_threshold,
            effective_maxk=eff_maxk,
            stepk=stepk,
        )

    def __repr__(self) -> str:
        parts = [
            f'oversample={self.oversample}',
            f'pad_factor={self.pad_factor}',
            f'spectral_method={self.spectral_method}',
        ]
        if self.spectral_method == 'oversample':
            parts.append(f'spectral_oversample={self.spectral_oversample}')
        if self.psf_mode != 'post_dispersion':
            parts.append(f'psf_mode={self.psf_mode}')
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
        rc.spectral_oversample,
        rc.spectral_method,
        rc.psf_mode,
        rc.effective_maxk,
        rc.stepk,
    )


def _render_config_unflatten(aux, children):
    return RenderConfig(
        oversample=aux[0],
        pad_factor=aux[1],
        folding_threshold=aux[2],
        maxk_threshold=aux[3],
        spectral_oversample=aux[4],
        spectral_method=aux[5],
        psf_mode=aux[6],
        effective_maxk=aux[7],
        stepk=aux[8],
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
        # the DC term (k=0) of a flux-normalized profile x pixel x PSF product
        # is ~1, so it must exceed threshold; an empty result means one factor
        # is not normalized (a real bug). Returning 0 would silently undersize
        # the FFT grid and alias the render -- raise loudly instead.
        raise ValueError(
            f"effective-maxk scan found no k with product > threshold "
            f"({threshold:g}); DC product={product[0]:g} (expected ~1 for a "
            f"flux-normalized profile x pixel x PSF). Model "
            f"{type(model).__name__!r} _ft_envelope, pixel_response, or PSF is "
            f"likely not normalized. params={params}"
        )
    return float(above[-1])


def compute_effective_maxk_grism(
    intensity_model,
    intensity_params: dict,
    sigma_v: float,
    grad_v_max: float,
    psf=None,
    threshold: float = 1e-3,
) -> float:
    """Effective maxk for one emission line's grism cube fine-grid sizing.

    The cube-slice intensity at fixed wavelength is
    ``M(x, y) = I(x, y) * G(x, y)`` (pointwise product). By the FT
    convolution theorem, ``FT[M] = FT[I] (*) FT[G]`` (convolution in
    k-space), and the support of the convolution is the Minkowski sum of
    supports: ``maxk(M) = maxk(I) + maxk(G)``. The cube slice is then
    convolved with the PSF in real space, so ``FT[Conv[M, PSF]] = FT[M] *
    FT[PSF]`` (multiplication). The effective post-PSF bandwidth is bounded
    by the smaller of:

    - the cube-slice bandwidth ``maxk(I) + maxk(G)``, and
    - the largest k below which the PSF FT remains above threshold.

    No pixel-response factor is included: the grism rendering chain applies
    the detector BoxPixel sinc post-dispersion (on the dispersed 2D
    observable), not per cube-slice cell. See
    ``docs/notes/grism_cube_bandwidth.tex`` for the derivation.

    Parameters
    ----------
    intensity_model : IntensityModel
        The emission line's spatial intensity model. Must expose
        ``maxk(params, threshold)``.
    intensity_params : dict
        Worst-case intensity parameters (typically the output of
        ``_extract_worst_case_params(intensity_model, priors)[0]``).
    sigma_v : float
        Worst-case (minimum) intrinsic kinematic dispersion sigma_v in km/s.
    grad_v_max : float
        Worst-case maximum ``|grad v_LOS|`` in km/s/arcsec across the
        spatial domain. From ``velocity_model.grad_bandwidth(params)``
        at worst-case velocity parameters.
    psf : galsim.GSObject, optional
        PSF profile. Damps the post-cube product at high k via ``psf.kValue``.
    threshold : float
        FT amplitude threshold. Default 1e-3.

    Returns
    -------
    float
        Effective maxk in rad/arcsec.

    Raises
    ------
    ValueError
        If the PSF FT never exceeds ``threshold`` (including at k=0), which
        indicates a non-normalized PSF rather than a legitimate bandwidth.
    """
    if sigma_v <= 0:
        raise ValueError(
            f"sigma_v must be positive (got {sigma_v}); the velocity-modulated "
            f"Gaussian factor is ill-defined at sigma_v = 0"
        )

    # cube-slice bandwidth bound (Minkowski sum of supports of FT[I] and FT[G])
    k_I = intensity_model.maxk(intensity_params, threshold=threshold)
    if grad_v_max > 0:
        k_G = np.sqrt(-2.0 * np.log(threshold)) * grad_v_max / sigma_v
    else:
        # uniform velocity field -> G has no spatial dependence -> no bandwidth
        # contribution from the modulation factor
        k_G = 0.0
    k_cube_bare = k_I + k_G

    if psf is None:
        return float(k_cube_bare)

    # PSF damping: post-PSF amplitude at k is bounded by FT[PSF](k). Find the
    # largest k <= k_cube_bare at which FT[PSF] remains above threshold.
    n_scan = 500
    k_scan = np.linspace(0.0, k_cube_bare, n_scan)

    import galsim

    psf_ft = np.array(
        [abs(psf.kValue(galsim.PositionD(0.0, float(k)))) for k in k_scan]
    )

    above = k_scan[psf_ft > threshold]
    if len(above) == 0:
        # FT[PSF](k=0) is ~1 for a flux-normalized PSF, so it must exceed
        # threshold; an empty result means the PSF is not normalized (a real
        # bug). Returning 0 would silently undersize the grism fine-grid and
        # alias the render -- raise loudly instead.
        raise ValueError(
            f"grism effective-maxk scan found no k with FT[PSF] > threshold "
            f"({threshold:g}); FT[PSF](0)={psf_ft[0]:g} (expected ~1 for a "
            f"flux-normalized PSF). The supplied PSF is likely not normalized."
        )
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
            if name in ('rscale', 'hlr'):
                # smallest scale → highest maxk (also the velocity-gradient
                # worst case: smallest rscale → steepest gradient → highest k_G)
                worst_maxk_params[name] = prior.low
                worst_stepk_params[name] = prior.high
            elif name == 'cosi':
                # smallest cosi (most edge-on) → highest maxk for intensity
                # AND highest sin(i) for velocity-gradient bandwidth
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
            elif name == 'vcirc':
                # largest vcirc → steepest velocity gradient → highest k_G
                worst_maxk_params[name] = prior.high
                worst_stepk_params[name] = prior.high
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


# ============================================================================
# Top-level builders for obs-aware RenderConfig sizing
# ============================================================================
#
# The classmethods ``RenderConfig.for_priors`` / ``RenderConfig.for_grism_priors``
# are the underlying primitives; they operate on a flat ``model.PARAMETER_NAMES``
# namespace and bare priors. The builders below are the user-facing entry points
# for the SourceModel API: they accept dotted-key ``PriorDict`` and the same
# primitives the ``build_*_obs`` factories take (image_pars / grism_pars, psf,
# pixel_response, broadband_key), so users construct an rc and an obs with the
# same arguments and thread the rc into the obs.
#
#   rc  = build_image_render_config(source, priors, image_pars, broadband_key='F087', psf=psf)
#   obs = build_image_obs(image_pars, psf=psf, broadband_key='F087', render_config=rc, data=..., variance=...)
#
# Skipping the builder yields ``build_image_obs``'s safe-default rc
# (``RenderConfig()`` -- point-sampled, oversample=1); appropriate for tutorials
# and quick renders, but undersized for inference with tight priors.


def build_image_render_config(
    source: 'SourceModel',
    priors: 'PriorDict',
    image_pars: 'ImagePars',
    broadband_key: str,
    *,
    psf=None,
    pixel_response: 'PixelResponse' = None,
    folding_threshold: float = 5e-3,
    maxk_threshold: float = 1e-3,
) -> RenderConfig:
    """Worst-case ``RenderConfig`` for an ``ImageObs`` channel.

    Looks up ``source.broadband_models[broadband_key]``, extracts the
    band-specific subset of the dotted-key ``priors`` (resolution
    ``<broadband_key>.<bare>`` → ``<bare>`` → verbatim), runs model-specific
    prior safety (e.g. Spergel cusp regime), and delegates to
    ``RenderConfig.for_priors``.

    Parameters
    ----------
    source : SourceModel
        Must have ``broadband_key`` in ``source.broadband_models``.
    priors : PriorDict
        Dotted-key SourceModel priors.
    image_pars : ImagePars
        Coarse pixel grid; ``pixel_scale`` is read from here.
    broadband_key : str
        Band name to size the rc for. Must exist in
        ``source.broadband_models``.
    psf : galsim.GSObject, optional
        PSF for the worst-case maxk product walk; matches what gets
        passed to ``build_image_obs``.
    pixel_response : PixelResponse, optional
        Defaults to ``BoxPixel(image_pars.pixel_scale)`` to match
        ``build_image_obs``'s default. Pass ``None`` to skip pixel
        damping in the worst-case scan (rarely what you want for
        inference).
    folding_threshold, maxk_threshold : float
        Passed through to ``RenderConfig.for_priors``.

    Returns
    -------
    RenderConfig
        Sized for the worst-case maxk and stepk in the prior bounds.

    Raises
    ------
    ValueError
        If ``broadband_key`` is not in ``source.broadband_models``.
    """
    from kl_pipe.pixel import BoxPixel
    from kl_pipe.source import _component_priors_for_intensity

    if broadband_key not in source.broadband_models:
        raise ValueError(
            f"broadband_key '{broadband_key}' not in source.broadband_models "
            f"(have: {sorted(source.broadband_models)})"
        )

    model = source.broadband_models[broadband_key]
    sub_priors = _component_priors_for_intensity(
        priors, broadband_key, model.PARAMETER_NAMES
    )

    # model-specific prior validation (loud failure before the worst-case scan)
    if hasattr(model, 'check_priors_safe'):
        model.check_priors_safe(sub_priors)

    if pixel_response is None:
        pixel_response = BoxPixel(image_pars.pixel_scale)

    return RenderConfig.for_priors(
        model,
        sub_priors,
        image_pars.pixel_scale,
        pixel_response=pixel_response,
        psf=psf,
        folding_threshold=folding_threshold,
        maxk_threshold=maxk_threshold,
    )


def build_grism_render_config(
    source: 'SourceModel',
    priors: 'PriorDict',
    grism_pars: 'GrismPars',
    *,
    psf=None,
    folding_threshold: float = 5e-3,
    maxk_threshold: float = 1e-3,
) -> RenderConfig:
    """Worst-case ``RenderConfig`` for a ``GrismObs`` cube.

    Iterates ``source.emission_lines``; for each line that owns a spatial
    profile (lines borrowing via ``intensity_key`` are validated through
    their owner), computes the per-line cube-slice rc via
    ``RenderConfig.for_grism_priors`` (Minkowski-sum bandwidth bound,
    intensity-FT support ⊕ velocity-modulation Gaussian support, damped
    by PSF). Returns the line with the largest required oversample
    (single shared spatial grid -> the most demanding line dictates).

    Math: see ``docs/notes/grism_cube_bandwidth.tex``.

    Parameters
    ----------
    source : SourceModel
        Must have ``velocity_model`` set and ``emission_lines`` non-empty.
    priors : PriorDict
        Dotted-key SourceModel priors. Required keys:
        ``<line>.<int_param>`` for each spatial-owner line,
        ``vel.<vel_param>`` for the velocity model,
        ``<disp_owner>.dispersion`` for each line's sigma_v (``disp_owner``
        is ``line.dispersion_key`` if set else the line key itself),
        plus shared ``cosi``, ``theta_int``, ``g1``, ``g2``.
    grism_pars : GrismPars
        Coarse pixel scale is read from ``grism_pars.image_pars.pixel_scale``.
    psf : galsim.GSObject, optional
        Per-slice PSF; passed through to
        ``compute_effective_maxk_grism``.
    folding_threshold, maxk_threshold : float
        Passed through to ``RenderConfig.for_grism_priors``.

    Returns
    -------
    RenderConfig
        Worst-case across spatial-owner emission lines.

    Raises
    ------
    ValueError
        If ``source.velocity_model`` is None or ``source.emission_lines``
        is empty or no line owns a spatial profile.
    KeyError
        If a required ``<line>.dispersion`` prior key is missing.
    """
    from kl_pipe.source import _component_priors_for_intensity

    if source.velocity_model is None:
        raise ValueError(
            "build_grism_render_config requires source.velocity_model to be set"
        )
    if not source.emission_lines:
        raise ValueError(
            "build_grism_render_config requires non-empty source.emission_lines"
        )

    coarse_pixel_scale = grism_pars.image_pars.pixel_scale
    velocity_priors = _component_priors_for_intensity(
        priors, 'vel', source.velocity_model.PARAMETER_NAMES
    )

    worst_rc = None
    for line_key, line in source.emission_lines.items():
        if line.intensity is None:
            # spatial profile borrowed via intensity_key; validated via owner
            continue
        intensity_model = line.intensity
        intensity_priors = _component_priors_for_intensity(
            priors, line_key, intensity_model.PARAMETER_NAMES
        )

        # model-specific prior safety (e.g. Spergel cusp regime)
        if hasattr(intensity_model, 'check_priors_safe'):
            intensity_model.check_priors_safe(intensity_priors)

        # sigma_v lookup: per-line dispersion (or shared via dispersion_key)
        disp_owner = (
            line.dispersion_key if line.dispersion_key is not None else line_key
        )
        disp_key = f'{disp_owner}.dispersion'
        if disp_key not in priors._param_spec:
            raise KeyError(
                f"emission line '{line_key}' requires prior key '{disp_key}'; "
                f"available: {sorted(priors._param_spec)}"
            )
        spec = priors._param_spec[disp_key]
        if hasattr(spec, 'low'):
            sigma_v_min = float(spec.low)
        else:
            sigma_v_min = float(spec)

        rc = RenderConfig.for_grism_priors(
            intensity_model=intensity_model,
            velocity_model=source.velocity_model,
            intensity_priors=intensity_priors,
            velocity_priors=velocity_priors,
            sigma_v_min=sigma_v_min,
            coarse_pixel_scale=coarse_pixel_scale,
            psf=psf,
            folding_threshold=folding_threshold,
            maxk_threshold=maxk_threshold,
        )
        if worst_rc is None or rc.oversample > worst_rc.oversample:
            worst_rc = rc

    if worst_rc is None:
        raise ValueError(
            "no emission line owns a spatial profile (all use intensity_key); "
            "cannot derive grism RenderConfig"
        )
    return worst_rc
