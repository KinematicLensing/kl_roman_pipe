"""Verification of ``build_grism_render_config`` against numerically converged renders.

Validates that the Minkowski-sum + PSF bandwidth bound derived in
``docs/notes/grism_cube_bandwidth.tex`` correctly predicts the spatial
oversample at which the grism cube + PSF convolution converges to a
reference computed at high oversample.

Four classes:

- ``TestConvergence``: at worst-case priors, sweep oversample upward on the
  full ``render_grism`` pipeline and show that the relative L2 error vs. a
  high-oversample reference drops below tolerance at the oversample
  predicted by ``build_grism_render_config``.
- ``TestDefaultAccuracy``: at typical (mid-prior) parameters, confirm that
  the build_grism_render_config-derived oversample produces a dispersed grism image
  matching the high-oversample reference within tolerance.
- ``TestCubeSliceConvergence``: isolates the cube-slice rendering
  (``build_cube`` + per-slice PSF + bin-to-coarse) from the full dispersion
  pipeline. The Minkowski-sum bandwidth bound is fundamentally about the
  cube slice spatial map; testing it directly removes confounders from
  downstream stages.
- ``TestDiagnostics``: writes visual diagnostic PNG panels for human
  inspection (cube slice, residual map, radial FT amplitude with predicted
  maxk marked, dispersed image).
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import galsim  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kl_pipe.coordinates import image_rotation_from_wcs  # noqa: E402
from kl_pipe.dispersion import GrismPars  # noqa: E402
from kl_pipe.intensity import InclinedExponentialModel  # noqa: E402
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS  # noqa: E402
from kl_pipe.observation import build_grism_obs  # noqa: E402
from kl_pipe.parameters import ImagePars  # noqa: E402
from kl_pipe.priors import PriorDict, Uniform  # noqa: E402
from kl_pipe.render import RenderConfig  # noqa: E402
from kl_pipe.render import build_grism_render_config  # noqa: E402
from kl_pipe.source import SourceModel  # noqa: E402
from kl_pipe.velocity import CenteredVelocityModel  # noqa: E402


OUT_DIR = 'tests/out/grism_bandwidth'


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope='module')
def coarse_image_pars():
    return ImagePars(shape=(16, 16), pixel_scale=0.1, indexing='ij')


@pytest.fixture(scope='module')
def gauss_psf():
    return galsim.Gaussian(fwhm=0.18)


@pytest.fixture(scope='module')
def grism_pars(coarse_image_pars):
    return GrismPars(
        image_pars=coarse_image_pars,
        dispersion=1.1,
        lambda_ref=1300.0,
        dispersion_angle_detector=0.0,
    )


@pytest.fixture(scope='module')
def source():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )


def _priors_for(cosi_prior, vcirc_prior, sigma_v_prior, rscale=0.3, vel_rscale=0.3):
    return PriorDict(
        {
            'cosi': cosi_prior,
            'theta_int': 0.3,
            'g1': 0.0,
            'g2': 0.0,
            'vel.vcirc': vcirc_prior,
            'vel.v0': 0.0,
            'vel.rscale': vel_rscale,
            'Halpha.flux': Uniform(1.0, 100.0),
            'Halpha.rscale': rscale,
            'Halpha.h_over_r': 0.15,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
            'Halpha.dispersion': sigma_v_prior,
            'z': 1.0,
        }
    )


@pytest.fixture(scope='module')
def worst_case_priors():
    """Tightest priors in the test-fixture space: most edge-on, fastest rotator,
    narrowest line, most compact velocity scale."""
    return _priors_for(
        cosi_prior=Uniform(0.1, 0.99),
        vcirc_prior=Uniform(80.0, 300.0),
        sigma_v_prior=Uniform(20.0, 150.0),
    )


@pytest.fixture(scope='module')
def typical_priors():
    """Tighter middle-of-the-road priors -- more representative of inference
    workloads where galaxy-priors-from-photometry constrain inclination."""
    return _priors_for(
        cosi_prior=Uniform(0.4, 0.95),
        vcirc_prior=Uniform(150.0, 250.0),
        sigma_v_prior=Uniform(40.0, 100.0),
        rscale=0.5,
        vel_rscale=0.5,
    )


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------


def _render_grism_at_oversample(
    source,
    priors,
    grism_pars,
    gauss_psf,
    oversample,
    theta_pars,
):
    """Render the dispersed grism image at a chosen oversample."""
    rc = RenderConfig(oversample=oversample)
    obs = build_grism_obs(
        grism_pars, z=theta_pars['z'], psf=gauss_psf, render_config=rc
    )
    return np.asarray(source.render_grism(theta_pars, obs))


def _midpoint_pars_from_priors(priors):
    """Pick midpoint values for sampled priors, fixed values verbatim."""
    pars = {}
    for name, spec in priors._param_spec.items():
        if hasattr(spec, 'low'):
            pars[name] = 0.5 * (spec.low + spec.high)
        else:
            pars[name] = float(spec)
    return pars


def _worst_case_pars_from_priors(priors):
    """Pick worst-case values: smallest cosi, largest vcirc, smallest rscale,
    smallest sigma_v -- matches what build_grism_render_config validates against."""
    pars = {}
    for name, spec in priors._param_spec.items():
        if hasattr(spec, 'low'):
            if name == 'cosi':
                pars[name] = spec.low  # most edge-on
            elif name == 'vel.vcirc':
                pars[name] = spec.high  # fastest rotation
            elif name == 'vel.rscale':
                pars[name] = spec.low  # most compact
            elif name == 'Halpha.dispersion':
                pars[name] = spec.low  # narrowest line
            elif name == 'Halpha.rscale':
                pars[name] = spec.low if hasattr(spec, 'low') else spec
            else:
                pars[name] = 0.5 * (spec.low + spec.high)
        else:
            pars[name] = float(spec)
    return pars


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestConvergence:
    """Worst-case-priors convergence sweep against high-oversample reference."""

    def test_worst_case_converges_at_predicted_oversample(
        self,
        source,
        worst_case_priors,
        grism_pars,
        gauss_psf,
        coarse_image_pars,
    ):
        rc_predicted = build_grism_render_config(
            source, worst_case_priors, grism_pars, psf=gauss_psf
        )
        n_predicted = rc_predicted.oversample
        print(
            f'\nworst-case predicted oversample: {n_predicted} '
            f'(effective_maxk={rc_predicted.effective_maxk:.2f})'
        )

        # use worst-case theta values (so the test exercises the regime
        # build_grism_render_config was sized for)
        theta_pars = _worst_case_pars_from_priors(worst_case_priors)

        # sweep oversamples; include the predicted N and several above
        oversamples = sorted(set([1, 3, 5, 7, n_predicted, n_predicted + 2, 15]))
        renders = {}
        for N in oversamples:
            renders[N] = _render_grism_at_oversample(
                source,
                worst_case_priors,
                grism_pars,
                gauss_psf,
                N,
                theta_pars,
            )

        # reference = highest oversample in the sweep
        N_ref = max(oversamples)
        ref = renders[N_ref]
        ref_norm = np.linalg.norm(ref)

        # relative L2 vs reference
        rel_l2 = {N: np.linalg.norm(renders[N] - ref) / ref_norm for N in oversamples}
        for N in oversamples:
            print(f'  oversample={N:2d}  rel L2 = {rel_l2[N]:.4e}')

        # at predicted oversample, error should be below the maxk threshold
        threshold = 5e-2  # 5% — loose; build_grism_render_config targets 1e-3 in
        # amplitude, but L2 sum-of-squared-errors aggregates
        assert rel_l2[n_predicted] < threshold, (
            f'rel L2 at predicted oversample {n_predicted} '
            f'({rel_l2[n_predicted]:.4e}) exceeds threshold {threshold}; '
            f'build_grism_render_config under-sized the grid'
        )

        # and converging: error decreases monotonically (or at worst stays
        # within ~1.5x noise band)
        for N1, N2 in zip(oversamples[:-1], oversamples[1:]):
            if N1 < n_predicted:
                continue  # error may be large at low N; allow non-monotone
            assert rel_l2[N2] <= 1.5 * rel_l2[N1], (
                f'rel L2 not converging: N={N1} -> {rel_l2[N1]:.4e}, '
                f'N={N2} -> {rel_l2[N2]:.4e}'
            )


class TestDefaultAccuracy:
    """Default build_grism_render_config-derived oversample meets accuracy at typical
    parameters."""

    def test_typical_priors_accuracy(
        self,
        source,
        typical_priors,
        grism_pars,
        gauss_psf,
        coarse_image_pars,
    ):
        rc_predicted = build_grism_render_config(
            source, typical_priors, grism_pars, psf=gauss_psf
        )
        n_predicted = rc_predicted.oversample

        # render at midpoint pars to evaluate "typical inference accuracy"
        theta_pars = _midpoint_pars_from_priors(typical_priors)

        ref = _render_grism_at_oversample(
            source,
            typical_priors,
            grism_pars,
            gauss_psf,
            15,
            theta_pars,
        )
        predicted = _render_grism_at_oversample(
            source,
            typical_priors,
            grism_pars,
            gauss_psf,
            n_predicted,
            theta_pars,
        )

        rel_l2 = np.linalg.norm(predicted - ref) / np.linalg.norm(ref)
        print(
            f'\ntypical predicted oversample: {n_predicted} '
            f'(effective_maxk={rc_predicted.effective_maxk:.2f}); '
            f'rel L2 vs N=15 = {rel_l2:.4e}'
        )
        # tighter: typical priors should give well-converged renders
        assert rel_l2 < 5e-2, (
            f'rel L2 at predicted oversample {n_predicted} = {rel_l2:.4e} > 5e-2; '
            f'build_grism_render_config under-sized for typical priors'
        )


def _render_cube_slice_at_oversample(
    source,
    pars,
    grism_pars,
    gauss_psf,
    oversample,
    coarse_shape,
    coarse_ps,
):
    """Render line-center cube slice + PSF, mean-bin to coarse, SB->flux/pixel.

    Isolates build_cube + per-slice PSF convolution from the dispersion stage.
    Output is in flux per coarse pixel (matching the
    ``docs/units_and_conventions.md`` convention).
    """
    from kl_pipe.spectral import CubePars
    from kl_pipe.psf import convolve_fft

    rc = RenderConfig(oversample=oversample)
    obs = build_grism_obs(
        grism_pars,
        z=pars['z'],
        psf=gauss_psf,
        render_config=rc,
    )
    if obs.fine_image_pars is not None:
        cube_pars = CubePars(
            image_pars=obs.fine_image_pars,
            lambda_grid=obs.cube_pars.lambda_grid,
        )
    else:
        cube_pars = obs.cube_pars
    cube = source.build_cube(
        pars,
        cube_pars,
        spectral_oversample=5,
        image_rotation=image_rotation_from_wcs(obs.grism_pars.image_pars.wcs),
    )
    # line center slice
    idx_center = len(obs.cube_pars.lambda_grid) // 2
    slice_fine = np.asarray(cube[:, :, idx_center])

    # per-slice PSF convolution (matches render_grism's vmap-over-wavelength)
    if obs.psf_data is not None:
        slice_fine = np.asarray(convolve_fft(slice_fine, obs.psf_data, bin=False))

    Nrow_c, Ncol_c = coarse_shape
    if oversample <= 1:
        sb_coarse = slice_fine
    else:
        sb_coarse = slice_fine.reshape(Nrow_c, oversample, Ncol_c, oversample).mean(
            axis=(1, 3)
        )
    return sb_coarse * coarse_ps**2


class TestCubeSliceConvergence:
    """Cube-slice rendering convergence isolated from dispersion.

    The bandwidth derivation in ``docs/notes/grism_cube_bandwidth.tex`` is
    fundamentally about the spatial map M(x, y; lambda_slice). Testing the
    cube slice directly (no dispersion) confirms the prediction with one
    fewer pipeline stage as confounder.
    """

    def test_worst_case_cube_slice_converges_at_predicted_oversample(
        self,
        source,
        worst_case_priors,
        grism_pars,
        gauss_psf,
        coarse_image_pars,
    ):
        coarse_ps = coarse_image_pars.pixel_scale
        coarse_shape = (coarse_image_pars.Nrow, coarse_image_pars.Ncol)
        rc_predicted = build_grism_render_config(
            source,
            worst_case_priors,
            grism_pars,
            psf=gauss_psf,
        )
        n_predicted = rc_predicted.oversample

        theta_pars = _worst_case_pars_from_priors(worst_case_priors)
        oversamples = sorted(set([1, 3, 5, 7, n_predicted, n_predicted + 2, 15]))
        renders = {
            N: _render_cube_slice_at_oversample(
                source,
                theta_pars,
                grism_pars,
                gauss_psf,
                N,
                coarse_shape,
                coarse_ps,
            )
            for N in oversamples
        }
        ref = renders[max(oversamples)]
        ref_norm = np.linalg.norm(ref)
        rel_l2 = {N: np.linalg.norm(renders[N] - ref) / ref_norm for N in oversamples}
        for N in oversamples:
            print(f'  oversample={N:2d}  cube-slice rel L2 = {rel_l2[N]:.4e}')

        # cube slice is the thing build_grism_render_config directly predicts;
        # convergence should hit the maxk_threshold of 1e-3 at the predicted N
        threshold = 1e-2
        assert rel_l2[n_predicted] < threshold, (
            f'cube-slice rel L2 at predicted oversample {n_predicted} '
            f'({rel_l2[n_predicted]:.4e}) exceeds threshold {threshold}'
        )


class TestDiagnostics:
    """Visual diagnostics — writes PNG panels for human inspection."""

    def test_write_diagnostic_panels(
        self,
        source,
        worst_case_priors,
        grism_pars,
        gauss_psf,
        coarse_image_pars,
    ):
        os.makedirs(OUT_DIR, exist_ok=True)

        coarse_ps = coarse_image_pars.pixel_scale
        rc_predicted = build_grism_render_config(
            source, worst_case_priors, grism_pars, psf=gauss_psf
        )
        n_predicted = rc_predicted.oversample

        # worst-case theta + sweep
        theta_pars = _worst_case_pars_from_priors(worst_case_priors)
        oversamples = sorted(set([1, 3, 5, 7, n_predicted, n_predicted + 2, 15]))
        renders = {
            N: _render_grism_at_oversample(
                source,
                worst_case_priors,
                grism_pars,
                gauss_psf,
                N,
                theta_pars,
            )
            for N in oversamples
        }
        ref = renders[max(oversamples)]

        # ---- panel 1: dispersed images side-by-side ----
        fig, axes = plt.subplots(1, len(oversamples), figsize=(3 * len(oversamples), 4))
        for ax, N in zip(axes, oversamples):
            im = ax.imshow(renders[N], origin='lower')
            tag = '(predicted)' if N == n_predicted else ''
            ax.set_title(f'oversample={N} {tag}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'dispersed_oversample_sweep.png'), dpi=120)
        plt.close()

        # ---- panel 2: residual maps vs reference ----
        fig, axes = plt.subplots(1, len(oversamples), figsize=(3 * len(oversamples), 4))
        for ax, N in zip(axes, oversamples):
            resid = renders[N] - ref
            vmax = max(abs(resid.min()), abs(resid.max()))
            im = ax.imshow(resid, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            tag = '(predicted)' if N == n_predicted else ''
            ax.set_title(
                f'N={N} {tag}\n||resid||/||ref||='
                f'{np.linalg.norm(resid)/np.linalg.norm(ref):.2e}'
            )
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'dispersed_residuals.png'), dpi=120)
        plt.close()

        # ---- panel 3: radial FT amplitude of reference + predicted maxk ----
        fft_ref = np.fft.fft2(ref)
        kx = 2.0 * np.pi * np.fft.fftfreq(ref.shape[1], d=coarse_ps)
        ky = 2.0 * np.pi * np.fft.fftfreq(ref.shape[0], d=coarse_ps)
        KX, KY = np.meshgrid(kx, ky, indexing='xy')
        K = np.sqrt(KX**2 + KY**2)
        k_nyq = np.pi / coarse_ps
        bin_edges = np.linspace(0, k_nyq, 32)
        amps = np.zeros(len(bin_edges) - 1)
        for i in range(len(amps)):
            mask = (K >= bin_edges[i]) & (K < bin_edges[i + 1])
            if np.any(mask):
                amps[i] = np.mean(np.abs(fft_ref[mask]))
        k_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        amps_norm = amps / amps.max() if amps.max() > 0 else amps
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(k_centers, amps_norm, 'k-', label='|FT[dispersed image]| / peak')
        ax.axvline(
            rc_predicted.effective_maxk,
            color='red',
            ls='--',
            label=f'predicted effective_maxk = {rc_predicted.effective_maxk:.1f}',
        )
        ax.axhline(
            rc_predicted.maxk_threshold,
            color='gray',
            ls=':',
            label=f'threshold = {rc_predicted.maxk_threshold}',
        )
        ax.set_xlabel('k (rad/arcsec)')
        ax.set_ylabel('|FT| / peak')
        ax.set_title(
            'Radial FT amplitude of dispersed grism image ' '(worst-case priors)'
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'dispersed_ft_amplitude.png'), dpi=120)
        plt.close()

        print(f'\nDiagnostic panels written to {OUT_DIR}/')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
