"""
Observation types for kinematic lensing models.

Bundles instrument state (PSF, grids, oversampling) and optional data
into frozen containers that models accept for rendering and likelihood.

Three types:
- ImageObs: broadband / narrowband 2D imaging
- VelocityObs(ImageObs): velocity map with flux weighting for PSF
- GrismObs: dispersed spectroscopy (grism)

Factory functions replace the old Model.configure_psf() family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel

from kl_pipe.parameters import ImagePars
from kl_pipe.pixel import BoxPixel, PixelResponse, _PIXEL_RESPONSE_UNSET
from kl_pipe.psf import PSFData
from kl_pipe.render import RenderConfig
from kl_pipe.utils import build_map_grid_from_image_pars


# default oversample for builders when no explicit RenderConfig is supplied.
# RenderConfig is the single source of truth for oversampling; this constant
# only sets the fallback recipe used for bespoke (non-inference) rendering.
# inference ignores it (InferenceTask.from_obs rebuilds with a priors-sized rc
# whenever the obs was built with a default RenderConfig).
DEFAULT_OVERSAMPLE = 5


# ============================================================================
# Observation types
# ============================================================================


@dataclass(frozen=True)
class ImageObs:
    """2D imaging observation (broadband, narrowband, etc.).

    Parameters
    ----------
    image_pars : ImagePars
        Pixel grid metadata (shape, pixel_scale).
    X, Y : jnp.ndarray
        Pre-computed coarse-scale coordinate grids.
    render_config : RenderConfig
        Rendering recipe (oversample, pad_factor, maxk_threshold, etc.).
        Canonical source for grid sizing -- ``obs.oversample`` is a
        property that reads from this. When ``build_image_obs`` is called
        without ``render_config``, the obs carries a builder-default rc
        marked ``_rc_was_default=True`` on the obs: ``InferenceTask.from_obs`` detects
        this and rebuilds the obs internally with a priors-sized rc via
        ``obs.with_render_config(...)``. For bespoke rendering outside
        ``from_obs``, either accept the builder default or pass an
        explicit ``RenderConfig`` (e.g.
        ``build_image_render_config(source, priors, image_pars,
        broadband_key, psf=psf)``).
    psf_data : PSFData, optional
        Pre-computed PSF FFT for convolve_fft.
    fine_X, fine_Y : jnp.ndarray, optional
        Fine-scale grids when render_config.oversample > 1.
    data : jnp.ndarray, optional
        Observed data (None = rendering-only, required for likelihood).
    variance : jnp.ndarray or float, optional
        Noise variance (same shape as data, or scalar).
    mask : jnp.ndarray, optional
        Boolean mask (True=valid). Same shape as data.
    kspace_psf_fft : jnp.ndarray, optional
        Fused k-space PSF kernel for k-space intensity rendering. Built
        from psf + render_config; co-derived with fine_X/fine_Y.
    pixel_response : PixelResponse, optional
        Pixel response function for k-space intensity rendering.
        Default BoxPixel is created by build_image_obs. None disables
        pixel integration (for testing or point-sampled comparisons).
    psf : galsim.GSObject, optional
        Original galsim PSF object retained so prior-based grid validation
        can include PSF damping in the worst-case maxk product scan. The
        rendered/precomputed PSF lives in ``psf_data``/``kspace_psf_fft``;
        this field is the source-of-truth galsim object kept for off-grid
        evaluation. Stored as static pytree aux.
    """

    image_pars: ImagePars
    X: jnp.ndarray
    Y: jnp.ndarray
    render_config: RenderConfig = None  # set by build_image_obs; never None at runtime
    psf_data: Optional[PSFData] = None
    fine_X: Optional[jnp.ndarray] = None
    fine_Y: Optional[jnp.ndarray] = None
    data: Optional[jnp.ndarray] = None
    variance: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    kspace_psf_fft: Optional[jnp.ndarray] = None
    pixel_response: Optional[PixelResponse] = None
    psf: Optional[object] = None  # galsim.GSObject; static aux for grid validation
    # psf_kernel_size: pinned kernel stamp size (fine pixels); preserved by
    # with_render_config so rebuilds keep the pinned shape. None = auto size.
    psf_kernel_size: Optional[int] = None
    # broadband_key: key into source.broadband_models that this obs renders
    # (used by SourceModel-based inference; None for rendering-only obs).
    broadband_key: Optional[str] = None
    # _rc_was_default: internal flag — True when build_image_obs supplied the
    # default render_config (caller passed render_config=None), False when the
    # caller passed an explicit one. Read by InferenceTask.from_obs to decide
    # whether to auto-derive a priors-sized rc + rebuild via with_render_config.
    # init=False keeps it out of the constructor kwargs; build_image_obs sets
    # it via object.__setattr__ (the standard escape hatch for frozen
    # dataclasses), and pytree unflatten restores it the same way.
    _rc_was_default: bool = field(default=False, init=False, repr=False)

    @property
    def oversample(self) -> int:
        """Oversample factor; canonical source is render_config.oversample."""
        return self.render_config.oversample if self.render_config is not None else 1

    def with_render_config(
        self, new_rc: 'RenderConfig', *, int_model=None
    ) -> 'ImageObs':
        """Return a new ImageObs with ``new_rc`` and freshly-recomputed grids.

        Used by ``InferenceTask.from_obs`` when the obs was constructed with
        a builder-default ``render_config`` (``obs._rc_was_default=True``) and the
        priors imply a different oversample. The returned obs has fresh
        ``psf_data`` (PSF FFT at ``new_rc.oversample``), fresh ``fine_X`` /
        ``fine_Y``, and fresh ``kspace_psf_fft`` (when ``int_model`` is
        supplied and supports k-space rendering). All other fields
        (``image_pars``, ``X``, ``Y``, ``data``, ``variance``, ``mask``,
        ``pixel_response``, ``psf``, ``broadband_key``) are preserved.

        Parameters
        ----------
        new_rc : RenderConfig
            The replacement rendering recipe. Determines the new
            ``oversample`` for grid sizing.
        int_model : IntensityModel, optional
            When supplied and the model has ``_kspace_pad_factor``, the
            fused k-space PSF kernel is recomputed for the band's rendering
            pipeline. Mirrors the ``int_model`` arg on ``build_image_obs``.

        Returns
        -------
        ImageObs
            A new instance (same dataclass type) with ``new_rc`` and
            recomputed precomputed grids.
        """
        import dataclasses

        oversample = new_rc.oversample

        new_psf_data = None
        new_fine_X = None
        new_fine_Y = None
        new_kspace_psf_fft = None

        if self.psf is not None:
            from kl_pipe.psf import precompute_psf_fft

            new_psf_data = precompute_psf_fft(
                self.psf,
                image_pars=self.image_pars,
                oversample=oversample,
                kernel_size=self.psf_kernel_size,
            )

            if int_model is not None and hasattr(int_model, '_kspace_pad_factor'):
                from kl_pipe.psf import precompute_psf_kspace_fft

                N = max(oversample, 1)
                fine_ps = self.image_pars.pixel_scale / N
                base_pad_sq = next_fast_len(
                    int_model._kspace_pad_factor
                    * max(self.image_pars.Nrow, self.image_pars.Ncol)
                )
                pad_sq = base_pad_sq * N
                new_kspace_psf_fft = precompute_psf_kspace_fft(
                    self.psf, (pad_sq, pad_sq), fine_ps
                )

        if oversample > 1:
            new_fine_X, new_fine_Y = _build_fine_spatial_grids(
                self.image_pars, oversample
            )

        return dataclasses.replace(
            self,
            render_config=new_rc,
            psf_data=new_psf_data,
            fine_X=new_fine_X,
            fine_Y=new_fine_Y,
            kspace_psf_fft=new_kspace_psf_fft,
        )


@dataclass(frozen=True)
class VelocityObs(ImageObs):
    """2D velocity observation with flux weighting for PSF convolution.

    Velocity PSF requires: v_obs = Conv(I*v, PSF) / Conv(I, PSF).
    Two modes for intensity source:

    - flux_model + flux_theta: evaluate intensity model with fixed params
    - flux_image: pre-rendered intensity map (upsampled to fine scale if needed)
    """

    flux_model: Optional['IntensityModel'] = None
    flux_theta: Optional[jnp.ndarray] = None
    flux_image: Optional[jnp.ndarray] = None
    # flux_weight_key: key into source.emission_lines whose intensity profile
    # weights the PSF for velocity rendering. None means no flux weighting
    # (velocity-only inference; mirrors flux_image=None semantics).
    flux_weight_key: Optional[str] = None


@dataclass(frozen=True)
class GrismObs:
    """Grism observation — dispersed spectroscopy.

    .. note::
       The current implementation uses one shared PSF (``psf_data`` /
       ``psf``) and one shared spatial+spectral cube across **all** emission
       lines on the source. Real instruments have wavelength-dependent PSFs
       and may benefit from per-line sub-cubes (to avoid wasted slices
       between widely-separated lines). Both are tracked as an open
       architectural item — see issue #51 — and require substantive
       changes to ``GrismObs`` (e.g. ``psf_data: Dict[line_key, PSFData]``),
       ``SourceModel.build_cube``, and ``SourceModel.render_grism``.
       Deferred past Phase 3.

    Parameters
    ----------
    grism_pars : GrismPars
        Dispersion parameters.
    cube_pars : CubePars
        Pre-computed wavelength grid at concrete redshift.
    psf_data : PSFData, optional
        Pre-computed PSF FFT for per-slice convolution.
    render_config : RenderConfig
        Rendering recipe (oversample, pad_factor, maxk_threshold, etc.).
        Canonical source for grid sizing -- ``obs.oversample`` is a
        property that reads from this. When ``build_grism_obs`` is called
        without ``render_config``, the obs carries a builder-default rc
        marked ``_rc_was_default=True`` on the obs: ``InferenceTask.from_obs`` detects
        this and rebuilds the obs internally with a priors-sized rc via
        ``obs.with_render_config(...)``. For bespoke rendering outside
        ``from_obs``, either accept the builder default or pass an
        explicit ``RenderConfig`` (e.g.
        ``build_grism_render_config(source, priors, grism_pars, psf=psf)``).
    fine_image_pars : ImagePars, optional
        Fine spatial grid (oversample > 1).
    data : jnp.ndarray, optional
        Observed grism data.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask (True=valid).
    pixel_response_fft : jnp.ndarray, optional
        Precomputed BoxPixel sinc on the fine k-grid, used by the
        post-dispersion 2D pixel-response step in ``SourceModel.render_grism``.
        Set by ``build_grism_obs`` when ``oversample > 1``; None at
        ``oversample == 1`` (no fine-grid sinc needed).
    psf : galsim.GSObject, optional
        Original galsim PSF retained for prior-grid validation so the
        worst-case maxk scan can include PSF damping (mirrors
        ``ImageObs.psf``). The rendered/precomputed PSF lives in
        ``psf_data``; this field is the source-of-truth galsim object
        used for off-grid evaluation.
    """

    grism_pars: object  # GrismPars — avoid circular import
    cube_pars: object  # CubePars — avoid circular import
    psf_data: Optional[PSFData] = None
    render_config: RenderConfig = None  # set by build_grism_obs; never None at runtime
    fine_image_pars: Optional[ImagePars] = None
    data: Optional[jnp.ndarray] = None
    variance: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    pixel_response_fft: Optional[jnp.ndarray] = None
    psf: Optional[object] = None  # galsim.GSObject; static aux for grid validation
    # psf_kernel_size: pinned kernel stamp size (fine pixels); preserved by
    # with_render_config so rebuilds keep the pinned shape. None = auto size.
    psf_kernel_size: Optional[int] = None
    # _rc_was_default: internal flag — True when build_grism_obs supplied the
    # default render_config (caller passed render_config=None), False when the
    # caller passed an explicit one. Read by InferenceTask.from_obs to decide
    # whether to auto-derive a priors-sized rc + rebuild via with_render_config.
    # init=False keeps it out of the constructor kwargs; build_grism_obs sets
    # it via object.__setattr__ (the standard escape hatch for frozen
    # dataclasses), and pytree unflatten restores it the same way.
    _rc_was_default: bool = field(default=False, init=False, repr=False)

    @property
    def oversample(self) -> int:
        """Oversample factor; canonical source is render_config.oversample."""
        return self.render_config.oversample if self.render_config is not None else 1

    @property
    def spectral_oversample(self) -> int:
        """Wavelength sub-bin count; canonical source is render_config.spectral_oversample."""
        return (
            self.render_config.spectral_oversample
            if self.render_config is not None
            else 15
        )

    @property
    def spectral_method(self) -> str:
        """Spectral bin-integration method; canonical source is render_config.spectral_method."""
        return (
            self.render_config.spectral_method
            if self.render_config is not None
            else 'erf'
        )

    @property
    def psf_mode(self) -> str:
        """Grism PSF pathway; canonical source is render_config.psf_mode."""
        return (
            self.render_config.psf_mode
            if self.render_config is not None
            else 'post_dispersion'
        )

    @property
    def cube_mode(self) -> str:
        """Cube-sharing strategy; canonical source is render_config.cube_mode."""
        return (
            self.render_config.cube_mode if self.render_config is not None else 'shared'
        )

    @property
    def dispersal_method(self) -> str:
        """Grism dispersal pathway; canonical source is render_config.dispersal_method."""
        return (
            self.render_config.dispersal_method
            if self.render_config is not None
            else 'slice'
        )

    @property
    def line_window_halfwidth(self):
        """Analytic deposit window half-width; canonical source is render_config."""
        return (
            self.render_config.line_window_halfwidth
            if self.render_config is not None
            else None
        )

    def with_render_config(self, new_rc: 'RenderConfig') -> 'GrismObs':
        """Return a new GrismObs with ``new_rc`` and freshly-recomputed grids.

        Used by ``InferenceTask.from_obs`` when the obs was constructed with
        a builder-default ``render_config`` (``obs._rc_was_default=True``) and the
        priors imply a different oversample. The returned obs has fresh
        ``psf_data`` (PSF FFT at ``new_rc.oversample``), fresh
        ``fine_image_pars``, and fresh ``pixel_response_fft``. All other
        fields (``grism_pars``, ``cube_pars``, ``data``, ``variance``,
        ``mask``, ``psf``) are preserved.

        Parameters
        ----------
        new_rc : RenderConfig
            The replacement rendering recipe. Determines the new
            ``oversample`` for grid sizing.

        Returns
        -------
        GrismObs
            A new instance with ``new_rc`` and recomputed precomputed grids.
        """
        import dataclasses

        oversample = new_rc.oversample

        new_psf_data = None
        new_fine_image_pars = None
        new_pixel_response_fft = None

        if self.psf is not None:
            from kl_pipe.psf import precompute_psf_fft

            new_psf_data = precompute_psf_fft(
                self.psf,
                image_pars=self.cube_pars.image_pars,
                oversample=oversample,
                kernel_size=self.psf_kernel_size,
            )

        if oversample > 1:
            new_fine_image_pars = self.cube_pars.image_pars.make_fine_scale(oversample)
            new_pixel_response_fft = _fine_pixel_response_fft(
                new_fine_image_pars, self.cube_pars.image_pars.pixel_scale
            )

        return dataclasses.replace(
            self,
            render_config=new_rc,
            psf_data=new_psf_data,
            fine_image_pars=new_fine_image_pars,
            pixel_response_fft=new_pixel_response_fft,
        )


# ============================================================================
# JAX pytree registration
# ============================================================================


def _image_obs_flatten(obs):
    children = (
        obs.X,
        obs.Y,
        obs.psf_data,
        obs.fine_X,
        obs.fine_Y,
        obs.data,
        obs.variance,
        obs.mask,
        obs.kspace_psf_fft,
        obs.pixel_response,
    )
    aux = (
        obs.image_pars,
        obs.render_config,
        obs.psf,
        obs.broadband_key,
        obs._rc_was_default,
        obs.psf_kernel_size,
    )
    return children, aux


def _image_obs_unflatten(aux, children):
    obs = ImageObs(
        image_pars=aux[0],
        render_config=aux[1],
        X=children[0],
        Y=children[1],
        psf_data=children[2],
        fine_X=children[3],
        fine_Y=children[4],
        data=children[5],
        variance=children[6],
        mask=children[7],
        kspace_psf_fft=children[8],
        pixel_response=children[9],
        psf=aux[2],
        broadband_key=aux[3],
        psf_kernel_size=aux[5],
    )
    # _rc_was_default is field(init=False); restore via frozen-dataclass bypass.
    object.__setattr__(obs, '_rc_was_default', aux[4])
    return obs


jax.tree_util.register_pytree_node(ImageObs, _image_obs_flatten, _image_obs_unflatten)


def _velocity_obs_flatten(obs):
    children = (
        obs.X,
        obs.Y,
        obs.psf_data,
        obs.fine_X,
        obs.fine_Y,
        obs.data,
        obs.variance,
        obs.mask,
        obs.kspace_psf_fft,
        obs.flux_theta,
        obs.flux_image,
    )
    aux = (
        obs.image_pars,
        obs.render_config,
        obs.flux_model,
        obs.psf,
        obs.broadband_key,
        obs.flux_weight_key,
        obs._rc_was_default,
    )
    return children, aux


def _velocity_obs_unflatten(aux, children):
    obs = VelocityObs(
        image_pars=aux[0],
        render_config=aux[1],
        X=children[0],
        Y=children[1],
        psf_data=children[2],
        fine_X=children[3],
        fine_Y=children[4],
        data=children[5],
        variance=children[6],
        mask=children[7],
        kspace_psf_fft=children[8],
        flux_model=aux[2],
        flux_theta=children[9],
        flux_image=children[10],
        psf=aux[3],
        broadband_key=aux[4],
        flux_weight_key=aux[5],
    )
    object.__setattr__(obs, '_rc_was_default', aux[6])
    return obs


jax.tree_util.register_pytree_node(
    VelocityObs, _velocity_obs_flatten, _velocity_obs_unflatten
)


def _grism_obs_flatten(obs):
    children = (
        obs.psf_data,
        obs.data,
        obs.variance,
        obs.mask,
        obs.pixel_response_fft,
    )
    aux = (
        obs.grism_pars,
        obs.cube_pars,
        obs.render_config,
        obs.fine_image_pars,
        obs.psf,
        obs._rc_was_default,
        obs.psf_kernel_size,
    )
    return children, aux


def _grism_obs_unflatten(aux, children):
    obs = GrismObs(
        grism_pars=aux[0],
        cube_pars=aux[1],
        render_config=aux[2],
        fine_image_pars=aux[3],
        psf_data=children[0],
        data=children[1],
        variance=children[2],
        mask=children[3],
        pixel_response_fft=children[4],
        psf=aux[4],
        psf_kernel_size=aux[6],
    )
    object.__setattr__(obs, '_rc_was_default', aux[5])
    return obs


jax.tree_util.register_pytree_node(GrismObs, _grism_obs_flatten, _grism_obs_unflatten)


# ============================================================================
# Factory functions
# ============================================================================


def _cast_and_validate_obs_arrays(data, variance, mask):
    """Cast data/variance/mask to JAX arrays and validate loudly.

    Checks performed eagerly at construction (not inside JIT):
    - ``variance`` is strictly positive everywhere. A zero/negative entry
      yields inf/NaN in the Gaussian log-likelihood, and a zero even on a
      masked-out pixel still poisons reverse-mode gradients through the
      ``jnp.where`` in ``_gaussian_log_likelihood``.
    - array ``variance`` and ``mask`` shapes match the data shape (a mismatch
      would otherwise surface only as a cryptic broadcast error inside the
      JITed likelihood). Scalar variance is allowed and broadcasts.

    Grism data is dispersed (shape differs from ``image_pars``), so shapes are
    cross-checked against ``data`` only, never against the grid.
    """
    if data is not None:
        data = jnp.asarray(data)
    if variance is not None:
        variance = jnp.asarray(variance)
        if not bool(jnp.all(variance > 0)):
            raise ValueError(
                "variance must be strictly positive everywhere; found "
                "non-positive entries (zero/negative variance yields inf/NaN "
                "in the Gaussian log-likelihood and NaN gradients)."
            )
        if data is not None and variance.ndim > 0 and variance.shape != data.shape:
            raise ValueError(
                f"variance shape {variance.shape} does not match data shape "
                f"{data.shape} (a scalar variance is allowed and broadcasts)."
            )
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
        if data is not None and mask.shape != data.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match data shape {data.shape}."
            )
    return data, variance, mask


def _build_fine_spatial_grids(image_pars, oversample):
    """Fine-scale coordinate grids for spatial oversampling.

    Returns ``(fine_X, fine_Y)``. Callers guard ``oversample > 1``.
    """
    fine_image_pars = image_pars.make_fine_scale(oversample)
    return build_map_grid_from_image_pars(fine_image_pars)


def _fine_pixel_response_fft(fine_image_pars, coarse_pixel_scale):
    """BoxPixel sinc FT on the fine k-grid, for post-dispersion pixel response.

    The pixel response is a coarse-detector property, so the BoxPixel side is
    ``coarse_pixel_scale`` even though the k-grid is at the fine pixel scale.
    """
    fine_ps = fine_image_pars.pixel_scale
    Nrow_f, Ncol_f = fine_image_pars.Nrow, fine_image_pars.Ncol
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(Ncol_f, d=fine_ps)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Nrow_f, d=fine_ps)
    KY, KX = jnp.meshgrid(ky, kx, indexing='ij')
    return BoxPixel(coarse_pixel_scale).ft(KX, KY)


def _validate_psf_kernel_size_args(
    psf_kernel_size: Optional[int], psf, rc_was_default: bool
) -> None:
    """Guard psf_kernel_size usage in the obs builders.

    A pinned kernel size is meaningless without a PSF, and it is defined in
    fine-scale pixels (pixel_scale / oversample), so it must be tied to an
    explicitly chosen render_config: builder-default configs are rebuilt by
    ``InferenceTask`` at a priors-derived oversample via
    ``with_render_config``, which preserves the pinned size but at a
    DIFFERENT fine pixel scale -- the pinned pixel count would then denote a
    different angular size than the caller computed. Explicit-rc rebuilds
    (e.g. the analytic-path line_window_halfwidth fill) keep the oversample,
    so preserving the pinned size there is exact.
    """
    if psf_kernel_size is None:
        return
    if psf is None:
        raise ValueError("psf_kernel_size given without a psf")
    if rc_was_default:
        raise ValueError(
            "psf_kernel_size requires an explicit render_config; the "
            "builder-default render_config is rebuilt by InferenceTask via "
            "with_render_config at a priors-derived oversample, which "
            "preserves the pinned pixel count but at a different fine pixel "
            "scale -- so the same count would denote a different angular "
            "size than the caller computed"
        )


def build_image_obs(
    image_pars: ImagePars,
    *,
    psf=None,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
    int_model=None,
    pixel_response=_PIXEL_RESPONSE_UNSET,
    render_config=None,
    broadband_key: Optional[str] = None,
    psf_kernel_size: Optional[int] = None,
) -> ImageObs:
    """Build imaging observation. Replaces Model.configure_psf().

    Parameters
    ----------
    image_pars : ImagePars
        Pixel grid metadata.
    psf : galsim.GSObject, optional
        PSF profile. None = no PSF convolution.
    gsparams : galsim.GSParams, optional
        GalSim rendering parameters.
    data : jnp.ndarray, optional
        Observed data. None = rendering-only.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask (True=valid).
    int_model : InclinedExponentialModel, optional
        When provided and has _kspace_pad_factor, also pre-compute
        fused k-space PSF kernel for the InclinedExponentialModel path.
    pixel_response : PixelResponse or None, optional
        Pixel response function for k-space rendering. Default (sentinel):
        auto-construct ``BoxPixel(image_pars.pixel_scale)``. Pass
        ``pixel_response=None`` explicitly to disable pixel integration
        (for testing or point-sampled comparisons).
    render_config : RenderConfig, optional
        Rendering recipe (single source of truth for oversampling, PSF FFT
        sizing, and fine-grid construction). When omitted, defaults to
        ``RenderConfig(oversample=DEFAULT_OVERSAMPLE)`` and the obs is marked
        ``_rc_was_default=True``: ``InferenceTask.from_obs`` will derive a
        priors-sized rc and rebuild the obs internally. For bespoke
        (non-inference) rendering with tight priors, pass an explicit
        ``build_image_render_config(...)`` result.
    psf_kernel_size : int, optional
        Explicit PSF kernel stamp size in fine-scale pixels
        (``pixel_scale / oversample``), passed to ``precompute_psf_fft``.
        Pins the kernel array shape (odd, >= automatic good size; raises
        otherwise). Requires an explicit ``render_config``: the
        builder-default config is rebuilt by ``InferenceTask`` via
        ``with_render_config`` at a priors-derived oversample, which
        preserves the pinned pixel count but at a different fine pixel
        scale, so the same count would denote a different angular size.
    """
    rc_was_default = render_config is None
    if rc_was_default:
        render_config = RenderConfig(oversample=DEFAULT_OVERSAMPLE)
    oversample = render_config.oversample
    _validate_psf_kernel_size_args(psf_kernel_size, psf, rc_was_default)

    X, Y = build_map_grid_from_image_pars(image_pars)

    # pixel response: default to BoxPixel from pixel_scale
    if pixel_response is _PIXEL_RESPONSE_UNSET:
        pixel_response = BoxPixel(image_pars.pixel_scale)

    psf_data = None
    fine_X = None
    fine_Y = None
    kspace_psf_fft = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=image_pars,
            oversample=oversample,
            gsparams=gsparams,
            kernel_size=psf_kernel_size,
        )

        # fused k-space PSF kernel for k-space intensity models
        if int_model is not None and hasattr(int_model, '_kspace_pad_factor'):
            from kl_pipe.psf import precompute_psf_kspace_fft

            N = max(oversample, 1)
            fine_ps = image_pars.pixel_scale / N
            # for wrap-compatible grids: compute base pad first, then
            # multiply by oversample so the fused PSF grid is an exact
            # multiple of the base grid (required by _wrap_kspace)
            base_pad_sq = next_fast_len(
                int_model._kspace_pad_factor * max(image_pars.Nrow, image_pars.Ncol)
            )
            pad_sq = base_pad_sq * N
            kspace_psf_fft = precompute_psf_kspace_fft(
                psf, (pad_sq, pad_sq), fine_ps, gsparams=gsparams
            )

    # fine grids: create when oversample > 1, regardless of PSF.
    # needed for velocity models (spatial oversampling) even without PSF.
    if oversample > 1:
        fine_X, fine_Y = _build_fine_spatial_grids(image_pars, oversample)

    data, variance, mask = _cast_and_validate_obs_arrays(data, variance, mask)

    obs = ImageObs(
        image_pars=image_pars,
        X=X,
        Y=Y,
        render_config=render_config,
        psf_data=psf_data,
        fine_X=fine_X,
        fine_Y=fine_Y,
        data=data,
        variance=variance,
        mask=mask,
        kspace_psf_fft=kspace_psf_fft,
        pixel_response=pixel_response,
        psf=psf,
        psf_kernel_size=psf_kernel_size,
        broadband_key=broadband_key,
    )
    # _rc_was_default is field(init=False) so it stays out of the constructor
    # kwargs; set it here via the standard frozen-dataclass escape hatch.
    if rc_was_default:
        object.__setattr__(obs, '_rc_was_default', True)
    return obs


def build_velocity_obs(
    image_pars: ImagePars,
    *,
    psf=None,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
    flux_model=None,
    flux_theta=None,
    flux_image=None,
    flux_image_pars=None,
    render_config=None,
    flux_weight_key: Optional[str] = None,
) -> VelocityObs:
    """Build velocity observation. Replaces VelocityModel.configure_velocity_psf().

    Parameters
    ----------
    image_pars : ImagePars
        Pixel grid metadata.
    psf : galsim.GSObject, optional
        PSF profile.
    gsparams : galsim.GSParams, optional
        GalSim rendering parameters.
    data : jnp.ndarray, optional
        Observed velocity data.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask.
    flux_model : IntensityModel, optional
        Intensity model for PSF flux weighting.
    flux_theta : jnp.ndarray, optional
        Fixed intensity params (used with flux_model).
    flux_image : ndarray, optional
        Pre-rendered intensity map for PSF flux weighting.
    flux_image_pars : ImagePars, optional
        Image parameters of flux_image (for resampling if shape differs).
    render_config : RenderConfig, optional
        Rendering recipe (single source of truth for oversampling). When
        omitted, defaults to ``RenderConfig(oversample=DEFAULT_OVERSAMPLE)``.

        Note: there is no ``build_velocity_render_config`` helper (unlike the
        image/grism cases). Velocity rendering flux-weights an intensity
        profile for PSF convolution, so a priors-sized rc is derived from that
        intensity component's helper (``build_image_render_config``). TODO:
        add a dedicated velocity helper if velocity-only PSF sizing ever needs
        its own path.
    """
    if render_config is None:
        render_config = RenderConfig(oversample=DEFAULT_OVERSAMPLE)
    oversample = render_config.oversample

    X, Y = build_map_grid_from_image_pars(image_pars)

    psf_data = None
    fine_X = None
    fine_Y = None
    processed_flux_image = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

        if flux_model is None and flux_image is None and flux_weight_key is None:
            raise ValueError(
                "Velocity PSF requires a flux source. Provide flux_model + "
                "flux_theta, flux_image (pre-rendered), or flux_weight_key "
                "(SourceModel emission line reference)."
            )

        # process flux_image: resample + upsample if needed
        if flux_image is not None:
            target_shape = (image_pars.Nrow, image_pars.Ncol)

            if flux_image.shape != target_shape:
                if flux_image_pars is None:
                    raise ValueError(
                        f"flux_image shape {flux_image.shape} != velocity grid "
                        f"{target_shape}. Provide flux_image_pars for resampling."
                    )
                from kl_pipe.psf import _resample_to_grid

                flux_image = _resample_to_grid(
                    flux_image,
                    flux_image_pars,
                    target_shape=target_shape,
                    target_pixel_scale=image_pars.pixel_scale,
                )

            if oversample > 1:
                from kl_pipe.psf import _resample_to_grid

                coarse_pars = ImagePars(
                    shape=target_shape,
                    pixel_scale=image_pars.pixel_scale,
                    indexing='ij',
                )
                fine_shape = (
                    target_shape[0] * oversample,
                    target_shape[1] * oversample,
                )
                fine_ps = image_pars.pixel_scale / oversample
                flux_image = _resample_to_grid(
                    flux_image,
                    coarse_pars,
                    target_shape=fine_shape,
                    target_pixel_scale=fine_ps,
                )

            processed_flux_image = jnp.asarray(flux_image)

    # fine grids: create when oversample > 1, regardless of PSF
    if oversample > 1:
        fine_X, fine_Y = _build_fine_spatial_grids(image_pars, oversample)

    data, variance, mask = _cast_and_validate_obs_arrays(data, variance, mask)
    if flux_theta is not None:
        flux_theta = jnp.asarray(flux_theta)

    return VelocityObs(
        image_pars=image_pars,
        X=X,
        Y=Y,
        render_config=render_config,
        psf_data=psf_data,
        fine_X=fine_X,
        fine_Y=fine_Y,
        data=data,
        variance=variance,
        mask=mask,
        kspace_psf_fft=None,
        flux_model=flux_model,
        flux_theta=flux_theta,
        flux_image=processed_flux_image,
        psf=psf,
        flux_weight_key=flux_weight_key,
    )


def build_grism_obs(
    grism_pars,
    z: float,
    *,
    psf=None,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
    render_config: Optional[RenderConfig] = None,
    velocity_window_kms: float = 3000.0,
    n_lambda: Optional[int] = None,
    slice_width_kms: Optional[float] = None,
    psf_kernel_size: Optional[int] = None,
) -> GrismObs:
    """Build grism observation.

    Parameters
    ----------
    grism_pars : GrismPars
        Dispersion parameters.
    z : float
        Concrete redshift for pre-computing cube_pars.
    psf : galsim.GSObject, optional
        PSF profile for per-slice convolution.
    gsparams : galsim.GSParams, optional
        GalSim rendering parameters.
    data : jnp.ndarray, optional
        Observed grism data.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask.
    render_config : RenderConfig, optional
        Rendering recipe (single source of truth for oversampling). When
        omitted, defaults to ``RenderConfig(oversample=DEFAULT_OVERSAMPLE)``
        and the obs is marked ``_rc_was_default=True``:
        ``InferenceTask.from_obs`` will derive a priors-sized rc and
        rebuild the obs internally. For bespoke (non-inference) rendering
        with tight priors, pass an explicit
        ``build_grism_render_config(source, priors, grism_pars, psf=psf)``
        result (from ``kl_pipe.render``).
    velocity_window_kms : float, optional
        Half-width of the wavelength-grid velocity window (km/s), passed
        to ``grism_pars.to_cube_pars``. Default 3000.
    n_lambda : int, optional
        Wavelength slice count, passed to ``grism_pars.to_cube_pars``.
        When neither this nor ``slice_width_kms`` is given, slices are
        sized ~1 dispersion pixel apart (~250 km/s at Roman-like
        configs). That default is fine for exploration but too coarse
        for production: each slice is dispersed rigidly to one detector
        offset, so the dispersed position of velocity structure is
        quantized at the slice width. Cost scales linearly with the
        slice count.
    slice_width_kms : float, optional
        Preferred way to size the grid: the slice width in velocity
        units. Choose a small fraction (roughly 1/20 to 1/30) of the
        largest line-of-sight velocity span the priors allow -- about 40
        km/s for vcirc priors reaching ~300 km/s, 60-80 km/s for
        exploration. Mutually exclusive with ``n_lambda``.
    psf_kernel_size : int, optional
        Explicit PSF kernel stamp size in fine-scale pixels
        (``pixel_scale / oversample``), passed to ``precompute_psf_fft``.
        Pins the kernel array shape (odd, >= automatic good size; raises
        otherwise) -- e.g. to keep wavelength-dependent PSF kernels at one
        shape across an ensemble of redshifts. Requires an explicit
        ``render_config``: the builder-default config is rebuilt by
        ``InferenceTask`` via ``with_render_config``, which would silently
        drop the pinned size.
    """
    rc_was_default = render_config is None
    if rc_was_default:
        render_config = RenderConfig(oversample=DEFAULT_OVERSAMPLE)
    oversample = render_config.oversample  # canonical
    _validate_psf_kernel_size_args(psf_kernel_size, psf, rc_was_default)

    cube_pars = grism_pars.to_cube_pars(
        z,
        velocity_window_kms=velocity_window_kms,
        n_lambda=n_lambda,
        slice_width_kms=slice_width_kms,
    )

    psf_data = None
    fine_image_pars = None
    pixel_response_fft = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=cube_pars.image_pars,
            oversample=oversample,
            gsparams=gsparams,
            kernel_size=psf_kernel_size,
        )

    # fine grid: create when oversample > 1, regardless of PSF. The BoxPixel
    # sinc is the post-dispersion pixel response (consumed by
    # SourceModel.render_grism).
    if oversample > 1:
        fine_image_pars = cube_pars.image_pars.make_fine_scale(oversample)
        pixel_response_fft = _fine_pixel_response_fft(
            fine_image_pars, cube_pars.image_pars.pixel_scale
        )

    data, variance, mask = _cast_and_validate_obs_arrays(data, variance, mask)

    obs = GrismObs(
        grism_pars=grism_pars,
        cube_pars=cube_pars,
        psf_data=psf_data,
        render_config=render_config,
        fine_image_pars=fine_image_pars,
        data=data,
        variance=variance,
        mask=mask,
        pixel_response_fft=pixel_response_fft,
        psf=psf,
        psf_kernel_size=psf_kernel_size,
    )
    # _rc_was_default is field(init=False) so it stays out of the constructor
    # kwargs; set it here via the standard frozen-dataclass escape hatch.
    if rc_was_default:
        object.__setattr__(obs, '_rc_was_default', True)
    return obs


# ============================================================================
# Shared-cube grouping: group grism obs that can share one model cube
# ============================================================================


#: relative tolerance below which two unequal lambda grids are treated as
#: accidental float drift (loud error) rather than intentionally distinct
#: line windows, which differ by many orders of magnitude more
_LAMBDA_NEAR_MISS_RTOL = 1e-9


def group_grism_obs_by_cube_compat(
    grism_obs: Dict[str, 'GrismObs'],
) -> List[Dict[str, 'GrismObs']]:
    """Group grism observations that can share one celestial-frame cube.

    Two obs are cube-compatible when their unconvolved model cubes are
    identical by construction: same wavelength grid, same spatial shape and
    pixel scale, same oversample factor, and same spectral method. The obs
    WCS rotation, data, variance, mask, and PSF are deliberately excluded:
    the roll rotation is fused into the dispersion sampling and the PSF is
    applied per-roll after dispersion, so neither touches the shared cube.

    Different emission-line complexes have wavelength grids that differ by
    far more than float noise and simply form separate groups (each gets
    its own shared cube) -- that is correct behavior, not an error. Grids
    that are unequal but agree to within ``_LAMBDA_NEAR_MISS_RTOL`` are
    almost certainly meant to be identical (per-roll construction drift)
    and raise loudly instead of silently falling back to per-roll cubes.

    Obs with ``dispersal_method='analytic'`` always form singleton
    groups: the analytic path builds no cube, so each rolled obs
    renders independently. This is routing, not an approximation --
    the result matches the per-roll path exactly.

    Parameters
    ----------
    grism_obs : dict[str, GrismObs]
        Per-roll grism observations keyed by name.

    Returns
    -------
    list[dict[str, GrismObs]]
        Groups (insertion-ordered); singleton groups are common and valid.
    """
    groups: List[Dict[str, 'GrismObs']] = []
    group_reps: List['GrismObs'] = []

    for key, obs in grism_obs.items():
        # analytic dispersal builds no cube; always a singleton group
        if obs.dispersal_method == 'analytic':
            groups.append({key: obs})
            group_reps.append(obs)
            continue
        placed = False
        for group, rep in zip(groups, group_reps):
            if _cube_compatible(obs, rep):
                group[key] = obs
                placed = True
                break
        if not placed:
            # near-miss check against every existing representative
            for rep_group, rep in zip(groups, group_reps):
                _check_lambda_near_miss(obs, rep, key, next(iter(rep_group)))
            groups.append({key: obs})
            group_reps.append(obs)
    return groups


def _cube_compatible(a: 'GrismObs', b: 'GrismObs') -> bool:
    """True when two grism obs can share one celestial-frame cube."""
    # analytic dispersal builds no cube; never joins a shared group
    if a.dispersal_method == 'analytic' or b.dispersal_method == 'analytic':
        return False
    ip_a, ip_b = a.grism_pars.image_pars, b.grism_pars.image_pars
    if (ip_a.Nrow, ip_a.Ncol) != (ip_b.Nrow, ip_b.Ncol):
        return False
    if ip_a.pixel_scale != ip_b.pixel_scale:
        return False
    if a.oversample != b.oversample:
        return False
    if a.spectral_method != b.spectral_method:
        return False
    if a.spectral_oversample != b.spectral_oversample:
        return False
    lam_a = np.asarray(a.cube_pars.lambda_grid)
    lam_b = np.asarray(b.cube_pars.lambda_grid)
    return lam_a.shape == lam_b.shape and bool((lam_a == lam_b).all())


def _check_lambda_near_miss(
    obs: 'GrismObs', rep: 'GrismObs', obs_key: str, rep_key: str
) -> None:
    """Raise when two incompatible obs have suspiciously close lambda grids."""
    lam_a = np.asarray(obs.cube_pars.lambda_grid)
    lam_b = np.asarray(rep.cube_pars.lambda_grid)
    if lam_a.shape != lam_b.shape:
        return
    if (lam_a == lam_b).all():
        return  # identical grids; incompatibility is elsewhere (shape/os/...)
    rel = np.max(np.abs(lam_a - lam_b) / np.maximum(np.abs(lam_b), 1e-300))
    if rel < _LAMBDA_NEAR_MISS_RTOL:
        raise ValueError(
            f"grism obs '{obs_key}' and '{rep_key}' have wavelength grids "
            f"that differ by max rel {rel:.2e} (< {_LAMBDA_NEAR_MISS_RTOL}) "
            f"-- almost certainly accidental float drift in per-roll "
            f"construction. Build all rolls of a line complex with an "
            f"identical lambda grid (or set cube_mode='per_roll' to disable "
            f"sharing)."
        )


def validate_shared_cube_group(obs_group: Dict[str, 'GrismObs'], psf_mode: str) -> None:
    """Loud-error guards for a multi-obs shared-cube group.

    Raises
    ------
    ValueError
        If psf_mode is not 'post_dispersion' (the shared celestial cube is
        unconvolved; per-slice convolution requires detector-frame cubes),
        if any obs WCS carries a parity flip (a flip is not a rotation and
        cannot be fused into the dispersion sampling), or if PSF presence
        is mixed across the group (fine/coarse cube grids would differ).
    """
    from kl_pipe.coordinates import wcs_is_flipped

    if len(obs_group) < 2:
        return
    keys = list(obs_group)
    if psf_mode != 'post_dispersion':
        raise ValueError(
            f"cube_mode='shared' with multiple grism obs {keys} requires "
            f"psf_mode='post_dispersion' (got {psf_mode!r}): the shared "
            f"celestial-frame cube is unconvolved, and per-slice "
            f"convolution needs a detector-frame cube per roll. Use "
            f"cube_mode='per_roll' for the per_slice reference path."
        )
    for key, obs in obs_group.items():
        wcs = obs.grism_pars.image_pars.wcs
        if wcs is not None and wcs_is_flipped(wcs):
            raise ValueError(
                f"grism obs '{key}' has a parity-flipped WCS (det(PC) < 0); "
                f"a flip is not a rotation and is not supported by the "
                f"shared-cube pathway. Use cube_mode='per_roll'."
            )
    psf_presence = {key: obs.psf_data is not None for key, obs in obs_group.items()}
    if len(set(psf_presence.values())) > 1:
        raise ValueError(
            f"shared-cube group has mixed PSF presence {psf_presence}; all "
            f"obs must have a PSF or none (the cube's fine/coarse spatial "
            f"grid depends on it)."
        )


def build_group_dispersion_operators(
    obs_group: Dict[str, 'GrismObs'],
) -> Dict[str, object]:
    """Precompute the per-obs dispersion operators for a shared-cube group.

    The shared cube is anchored in the FIRST obs's detector frame, so each
    operator carries the RELATIVE rotation ``rot_obs - rot_anchor`` (zero
    for the anchor: no rotational resampling for that roll). Operators are
    model- and galaxy-independent; build once per group at likelihood
    construction and reuse across every likelihood evaluation (and across
    galaxies sharing the grid/wavelength configuration).
    """
    from kl_pipe.coordinates import image_rotation_from_wcs
    from kl_pipe.dispersion import precompute_dispersion_operator

    first = next(iter(obs_group.values()))
    rot_anchor = image_rotation_from_wcs(first.grism_pars.image_pars.wcs)
    operators = {}
    for key, obs in obs_group.items():
        rot = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
        operators[key] = precompute_dispersion_operator(
            obs.grism_pars,
            obs.cube_pars.lambda_grid,
            oversample=obs.oversample,
            image_rotation=rot - rot_anchor,
        )
    return operators
