"""Flagship integration test: joint Roman-like broadband + grism inference.

The headline scientific claim:

    At Roman-like noise + PSF + dispersion, joint broadband-photometric +
    slitless-grism inference with NUTS recovers the full 16-dim model
    parameter set within joint Nsigma < 3, demonstrating end-to-end
    correctness of the kinematic lensing forward model and inference
    pipeline.

This is the **top-level visual regression test** for the repo: every
diagnostic plot doubles as a presentation / paper asset. Outputs sort first
alphabetically in ``tests/out/`` via the ``00_flagship/`` directory.

Configuration (Roman-like):
- F087 broadband: 32x32 image at 0.11 arcsec/pixel (native Roman), PSF FWHM 0.18"
- F184 grism: 32x32 dispersed image, dispersion 1.1 nm/pix, lambda_ref derived
  from Halpha at z=1.0 (lambda_obs ~ 1313 nm, well inside the F184 band)
- Galaxy: z=1.0, vcirc=200 km/s, hlr~0.3", inc~53 deg (cosi=0.6), theta_int=pi/4
- Shear: g1=0.02, g2=-0.01
- SNR (matched-filter): 100 broadband, 50 grism

Sample space (17-dim):
- Geometry:        cosi, theta_int, g1, g2
- Velocity:        vel.v0, vel.vcirc, vel.rscale (CenteredVelocityModel -- no vel.x0/y0)
- Broadband F087:  F087.flux, F087.rscale, F087.x0, F087.y0   (h_over_r fixed)
- Emission line:   Halpha.flux, Halpha.rscale, Halpha.x0, Halpha.y0,
                   Halpha.dispersion                          (h_over_r fixed)
- Continuum under line:
                   Halpha.cont.flux_per_nm                           (others fixed to line spatial truth)
The continuum is intentionally non-zero so the flat continuum trace is
visible above the noise floor in the dispersed grism image.

Pass criterion (single, matching existing diagnostic convention):
    joint Nsigma > 3 -> pytest.fail
    2 < joint Nsigma <= 3 -> warnings.warn
    <= 2 -> pass silently

R-hat / ESS / per-param recovery are computed and saved as diagnostics but
do NOT gate the test. They surface in the plots for human review.

Marked ``slow`` -- excluded from ``make test-basic``. Run via
``make test-flagship`` (or pytest tests/test_flagship.py directly).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict

import galsim
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kl_pipe.diagnostics.imaging import (
    compute_joint_nsigma,
    plot_data_comparison_panels,
)
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.observation import (
    build_grism_obs,
    build_image_obs,
    build_velocity_obs,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.priors import Gaussian, PriorDict, TruncatedNormal
from kl_pipe.sampling import (
    InferenceTask,
    NumpyroSamplerConfig,
    build_sampler,
)
from kl_pipe.sampling.diagnostics import (
    plot_corner,
    plot_recovery,
    plot_trace,
)
from kl_pipe.source import SourceModel
from kl_pipe.synthetic import SyntheticIntensity
from kl_pipe.utils import get_test_dir
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.render import RenderConfig


pytestmark = pytest.mark.slow


# Roman-like configuration (module-scope constants -- referenced in plot titles)
Z = 1.0
PIXEL_SCALE = 0.11  # arcsec/pixel, Roman native
IMAGE_SHAPE = (32, 32)
PSF_FWHM = 0.18  # arcsec, Roman F087/F184-like
SNR_BROADBAND = 100.0
SNR_GRISM = 150.0
GRISM_DISPERSION_NM_PER_PIX = 1.1

# Spatial oversample factor for both broadband and grism obs. Default is 5;
# 3 is acceptable here because PSF FWHM (0.18") is ~1.6 pixels at 0.11"/pix
# and the resulting profile is band-limited enough that osf=3 gives sub-1%
# bias on rendered images. Cuts FFT work ~3x vs default.
SPATIAL_OVERSAMPLE = 3

# Sampler settings. Uses the Laplace preconditioner (precondition='laplace'):
# NUTS starts at the MAP with a fixed inverse-Hessian mass matrix, so warmup is
# short (just step-size adaptation) instead of the ~300 iters dense-mass needs
# to climb from an identity metric. Validated ~5x faster (8.5 vs 42.5 min) with
# equal recovery + better convergence; see experiments/sweverett/flagship_speedup.
#
# Two configs selectable via the --flagship-long pytest flag (see conftest.py):
#   short (default): fast gating run.
#   long: production-depth run for cleaner posteriors (make test-flagship-long).
# Only sampler depth differs between them -- truth, priors, and the joint-Nsigma
# pass criterion are identical.
SHORT_CONFIG = {
    'n_warmup': 50,
    'n_samples': 300,
    'n_chains': 2,
    'max_tree_depth': 8,
    'chain_method': 'vectorized',  # 2 chains batched -- modest peak memory
}
# Long mode: 4 chains vectorized, 2000 samples, depth 10 -- production-depth for
# clean posteriors (8000 draws). Earlier this config was SIGKILLed at the end of
# sampling ("zsh: killed" at 100%); root cause was NOT the sampler but the
# wrapper computing log_prob by vmap-ing the FFT-rendering likelihood over all
# n_samples*n_chains at once (a transient that scaled with sample count and
# OOM'd). Fixed in numpyro.py via chunked evaluation (_batched_log_posterior_
# chunked); peak memory now bounded regardless of sample count.
LONG_CONFIG = {
    'n_warmup': 200,
    'n_samples': 2000,
    'n_chains': 4,
    'max_tree_depth': 10,
    'chain_method': 'vectorized',
}


# Headline parameter subset for the small corner plot.
HEADLINE_PARAMS = [
    'g1',
    'g2',
    'cosi',
    'theta_int',
    'vel.vcirc',
    'vel.rscale',
    'F087.rscale',
    'Halpha.rscale',
]


@pytest.fixture(scope='module')
def output_dir():
    """Top-of-alphabet output directory so diagnostics surface first."""
    out_dir = get_test_dir() / 'out' / '00_flagship'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _true_pars_dotted() -> Dict[str, float]:
    return {
        # Geometry
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': 0.02,
        'g2': -0.01,
        # Velocity (CenteredVelocityModel -- vel.x0, vel.y0 are not parameters)
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 0.3,
        # Broadband F087
        'F087.flux': 100.0,
        'F087.rscale': 0.3,
        'F087.h_over_r': 0.1,
        'F087.x0': 0.0,
        'F087.y0': 0.0,
        # Halpha emission line
        'Halpha.flux': 100.0,
        'Halpha.rscale': 0.25,
        'Halpha.h_over_r': 0.1,
        'Halpha.x0': 0.0,
        'Halpha.y0': 0.0,
        'Halpha.dispersion': 50.0,
        # Continuum under Halpha line. Spatial profile fixed equal to the line
        # spatial truth; only the continuum flux is sampled. The continuum
        # produces a flat trace across the dispersed grism image visible above
        # the noise floor at this SNR.
        'Halpha.cont.flux_per_nm': 25.0,
        'Halpha.cont.rscale': 0.25,
        'Halpha.cont.h_over_r': 0.1,
        'Halpha.cont.x0': 0.0,
        'Halpha.cont.y0': 0.0,
        # Redshift (fixed in prior)
        'z': Z,
    }


def _flagship_priors(true: Dict[str, float]) -> PriorDict:
    """Moderately-narrow TruncatedNormal priors centered on truth; fix h_over_r + z.

    theta_int is restricted to [0, pi/2] (single position-angle quadrant) --
    the full [0, pi] range exposes the classic sin(2theta) two-mode
    degeneracy that destroys NUTS mixing at this dimensionality.
    """
    return PriorDict(
        {
            # Geometry
            'cosi': TruncatedNormal(0.6, 0.15, 0.05, 0.99),
            'theta_int': TruncatedNormal(np.pi / 4, 0.3, 0.0, np.pi / 2),
            'g1': TruncatedNormal(0.0, 0.04, -0.1, 0.1),
            'g2': TruncatedNormal(0.0, 0.04, -0.1, 0.1),
            # Velocity
            'vel.v0': Gaussian(10.0, 10.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 80.0, 400.0),
            'vel.rscale': TruncatedNormal(0.3, 0.1, 0.05, 1.0),
            # Broadband F087
            'F087.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
            'F087.rscale': TruncatedNormal(0.3, 0.08, 0.05, 1.0),
            'F087.h_over_r': 0.1,  # fixed
            'F087.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'F087.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            # Halpha emission line
            'Halpha.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
            'Halpha.rscale': TruncatedNormal(0.25, 0.08, 0.05, 1.0),
            'Halpha.h_over_r': 0.1,  # fixed
            'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
            # Continuum under Halpha (only flux sampled; spatial profile fixed
            # to line spatial truth so the inference doesn't have to also solve
            # a continuum-line spatial degeneracy at this SNR).
            'Halpha.cont.flux_per_nm': TruncatedNormal(25.0, 15.0, 0.0, 200.0),
            'Halpha.cont.rscale': true['Halpha.cont.rscale'],
            'Halpha.cont.h_over_r': true['Halpha.cont.h_over_r'],
            'Halpha.cont.x0': true['Halpha.cont.x0'],
            'Halpha.cont.y0': true['Halpha.cont.y0'],
            # Redshift fixed
            'z': Z,
        }
    )


def _plot_ground_truth_overview(
    data_F087_noisy: np.ndarray,
    data_F087_true: np.ndarray,
    data_grism_noisy: np.ndarray,
    data_grism_true: np.ndarray,
    velocity_true: np.ndarray,
    intensity_true: np.ndarray,
    output_path: Path,
):
    """6-panel composite: noisy data, noiseless truth, and intrinsic fields."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    panels = [
        (axes[0, 0], data_F087_noisy, 'F087 broadband (data, with noise)', 'viridis'),
        (axes[0, 1], data_F087_true, 'F087 broadband (truth)', 'viridis'),
        (
            axes[0, 2],
            intensity_true,
            'Intrinsic F087 intensity (no noise, no PSF)',
            'viridis',
        ),
        (axes[1, 0], data_grism_noisy, 'Halpha grism (data, with noise)', 'magma'),
        (axes[1, 1], data_grism_true, 'Halpha grism (truth)', 'magma'),
        (axes[1, 2], velocity_true, 'LOS velocity field (km/s)', 'RdBu_r'),
    ]
    for ax, img, title, cmap in panels:
        if cmap == 'RdBu_r':
            vmax = float(np.max(np.abs(img)))
            im = ax.imshow(
                np.asarray(img), cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower'
            )
        else:
            im = ax.imshow(np.asarray(img), cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f'Flagship: Roman-like joint F087 + Halpha grism inference at z={Z:.2f} '
        f'(SNR {SNR_BROADBAND:.0f}/{SNR_GRISM:.0f})',
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_summary_txt(
    output_path: Path,
    result,
    runtime_sec: float,
    true_pars: Dict[str, float],
    joint_nsigma: float,
    sampler_config: Dict[str, int],
):
    """Plain-text dump: sampler stats, R-hat, ESS, per-param recovery."""
    summary = result.get_summary()
    rhats = result.get_rhat() if sampler_config['n_chains'] > 1 else None
    ess = result.get_ess()
    lines = []
    lines.append('=' * 80)
    lines.append('FLAGSHIP TEST SUMMARY')
    lines.append('=' * 80)
    lines.append(f'Sampler:     numpyro NUTS')
    lines.append(
        f'Chains:      {sampler_config["n_chains"]}    '
        f'Warmup: {sampler_config["n_warmup"]}    '
        f'Samples/chain: {sampler_config["n_samples"]}'
    )
    lines.append(f'Runtime:     {runtime_sec:.1f} s')
    lines.append(f'Joint Nsigma: {joint_nsigma:.3f}')
    if result.acceptance_fraction is not None:
        lines.append(f'Acceptance:  {result.acceptance_fraction:.2%}')
    lines.append('')
    lines.append(
        f"{'Parameter':<22} {'truth':>10} {'mean':>10} {'std':>10} "
        f"{'R-hat':>8} {'ESS':>8}  err"
    )
    lines.append('-' * 80)
    for name in result.param_names:
        s = summary[name]
        truth = true_pars.get(name, float('nan'))
        rhat = rhats[name] if rhats is not None else float('nan')
        ess_val = ess[name] if ess is not None else float('nan')
        if abs(truth) > 1e-9:
            err = abs(s['mean'] - truth) / abs(truth)
            err_str = f'{err:.1%}'
        else:
            err_str = 'N/A'
        lines.append(
            f'{name:<22} {truth:>10.4f} {s["mean"]:>10.4f} {s["std"]:>10.4f} '
            f'{rhat:>8.3f} {ess_val:>8.0f}  {err_str}'
        )
    lines.append('=' * 80)
    output_path.write_text('\n'.join(lines))


class TestFlagship:
    """Top-level visual regression: joint Roman-like broadband + grism inference."""

    def test_recover_joint_phot_grism(self, output_dir, request):
        """End-to-end: synth Roman-like data, run NUTS, validate via joint Nsigma."""
        long_mode = request.config.getoption('--flagship-long')
        sampler_config = LONG_CONFIG if long_mode else SHORT_CONFIG
        print(
            f"\nFlagship sampler config: {'long' if long_mode else 'short'} "
            f"-> {sampler_config}"
        )

        pars_dotted = _true_pars_dotted()

        image_pars = ImagePars(
            shape=IMAGE_SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij'
        )
        psf = galsim.Gaussian(fwhm=PSF_FWHM)

        # Grism: place Halpha rest=656.28 nm at observed lambda via z
        grism_pars = build_grism_pars_for_line(
            LINE_LAMBDAS['Halpha'],
            redshift=Z,
            image_pars=image_pars,
            dispersion=GRISM_DISPERSION_NM_PER_PIX,
        )

        # SourceModel (same instance used for synth and inference)
        f087_int = InclinedExponentialModel()
        halpha_int = InclinedExponentialModel()
        halpha_cont = InclinedExponentialModel()
        source = SourceModel(
            velocity_model=CenteredVelocityModel(),
            broadband_models={'F087': f087_int},
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=halpha_int,
                    continuum=halpha_cont,
                )
            },
        )

        # ====================================================================
        # Synthetic data
        # ====================================================================

        # F087 broadband via SyntheticIntensity (PSF-aware, GalSim backend)
        F087_pars_flat = {
            'cosi': pars_dotted['cosi'],
            'theta_int': pars_dotted['theta_int'],
            'g1': pars_dotted['g1'],
            'g2': pars_dotted['g2'],
            'flux': pars_dotted['F087.flux'],
            'rscale': pars_dotted['F087.rscale'],
            'h_over_r': pars_dotted['F087.h_over_r'],
            'x0': pars_dotted['F087.x0'],
            'y0': pars_dotted['F087.y0'],
        }
        synth_F087 = SyntheticIntensity(
            F087_pars_flat, model_type='exponential', seed=42, psf=psf
        )
        data_F087_noisy = synth_F087.generate(
            image_pars, snr=SNR_BROADBAND, seed=42, include_poisson=False
        )
        data_F087_true = synth_F087.data_true
        var_F087 = synth_F087.variance

        # Grism via SourceModel.render_grism + Gaussian noise sized to SNR
        grism_obs_clean = build_grism_obs(grism_pars, z=Z, psf=psf)
        data_grism_true = np.asarray(source.render_grism(pars_dotted, grism_obs_clean))
        signal_power = float(np.sum(data_grism_true**2))
        var_grism = signal_power / SNR_GRISM**2
        rng = np.random.default_rng(43)
        data_grism_noisy = data_grism_true + rng.normal(
            0.0, np.sqrt(var_grism), size=data_grism_true.shape
        )

        # Truth-side intrinsic velocity + intensity images (for the hero plot)
        vel_obs_clean = build_velocity_obs(image_pars)
        img_obs_clean = build_image_obs(image_pars, broadband_key='F087')
        velocity_true = np.asarray(source.render_velocity(pars_dotted, vel_obs_clean))
        intensity_true = np.asarray(
            source.render_broadband(pars_dotted, img_obs_clean, 'F087')
        )

        # ====================================================================
        # Build obs WITH data + InferenceTask
        # ====================================================================

        obs_F087 = build_image_obs(
            image_pars,
            psf=psf,
            render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
            data=jnp.asarray(data_F087_noisy),
            variance=var_F087,
            int_model=f087_int,
            broadband_key='F087',
        )
        obs_grism = build_grism_obs(
            grism_pars,
            z=Z,
            psf=psf,
            render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
            data=jnp.asarray(data_grism_noisy),
            variance=float(var_grism),
        )

        priors = _flagship_priors(pars_dotted)
        task = InferenceTask.from_obs(
            source,
            priors,
            image_obs={'F087': obs_F087},
            grism_obs={'roll0': obs_grism},
        )

        # ====================================================================
        # Hero plot (before sampling, so something useful exists even on failure)
        # ====================================================================
        _plot_ground_truth_overview(
            data_F087_noisy=data_F087_noisy,
            data_F087_true=data_F087_true,
            data_grism_noisy=data_grism_noisy,
            data_grism_true=data_grism_true,
            velocity_true=velocity_true,
            intensity_true=intensity_true,
            output_path=output_dir / 'ground_truth_overview.png',
        )

        # ====================================================================
        # Run NUTS
        # ====================================================================

        config = NumpyroSamplerConfig(
            n_samples=sampler_config['n_samples'],
            n_warmup=sampler_config['n_warmup'],
            n_chains=sampler_config['n_chains'],
            chain_method=sampler_config['chain_method'],
            seed=42,
            progress=False,
            reparam_strategy='prior',
            dense_mass=True,  # (inert under precondition='laplace', which uses
            # the fixed inv-Hessian metric; kept for the non-preconditioned path)
            precondition='laplace',  # MAP init + fixed Laplace mass -> ~5x faster
            # warmup on this correlated joint posterior (see flagship_speedup)
            target_accept_prob=0.8,
            max_tree_depth=sampler_config['max_tree_depth'],
            init_strategy='prior',  # narrow priors are centered on truth
        )
        sampler = build_sampler('numpyro', task, config)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = sampler.run()
        runtime = time.time() - start
        print(f'\nFlagship NUTS run: {result.n_samples} samples in {runtime:.1f}s')

        # ====================================================================
        # Recovery diagnostics
        # ====================================================================

        # MAP from samples
        map_idx = int(np.argmax(result.log_prob))
        map_pars_dotted = {
            name: float(result.samples[map_idx, i])
            for i, name in enumerate(result.param_names)
        }
        # Merge truth fixed params (priors fixed values) so render has full keys
        for name, val in task.fixed_params.items():
            map_pars_dotted.setdefault(name, val)

        # Model evaluations at MAP (for the recovery panels)
        model_F087_at_MAP = np.asarray(
            source.render_broadband(map_pars_dotted, obs_F087, 'F087')
        )
        model_grism_at_MAP = np.asarray(source.render_grism(map_pars_dotted, obs_grism))

        # Per-channel data-comparison panels
        plot_data_comparison_panels(
            data_noisy=np.asarray(data_F087_noisy),
            data_true=np.asarray(data_F087_true),
            model_eval=model_F087_at_MAP,
            test_name='flagship',
            output_dir=output_dir,
            data_type='F087_broadband',
            variance=np.asarray(var_F087),
            n_params=task.n_params,
        )
        plot_data_comparison_panels(
            data_noisy=np.asarray(data_grism_noisy),
            data_true=np.asarray(data_grism_true),
            model_eval=model_grism_at_MAP,
            test_name='flagship',
            output_dir=output_dir,
            data_type='grism',
            variance=float(var_grism),
            n_params=task.n_params,
        )

        # Full 16-dim corner (large, but readable on monitor)
        sampler_info = {
            'name': 'numpyro NUTS',
            'runtime': runtime,
            'settings': {
                'chains': sampler_config['n_chains'],
                'warmup': sampler_config['n_warmup'],
                'samples': sampler_config['n_samples'],
                'SNR_F087': SNR_BROADBAND,
                'SNR_grism': SNR_GRISM,
                'z': Z,
            },
        }
        # smooth/smooth1d (bin units) de-jag the marginals for presentation; mild
        # (1.0) so structure is preserved -- these are slide/paper assets.
        fig_corner = plot_corner(
            result,
            true_values=pars_dotted,
            map_values=map_pars_dotted,
            sampler_info=sampler_info,
            smooth=1.0,
            smooth1d=1.0,
        )
        fig_corner.savefig(output_dir / 'corner_full.png', dpi=120, bbox_inches='tight')
        plt.close(fig_corner)

        # Headline subset corner (presentation-ready)
        fig_headline = plot_corner(
            result,
            params=HEADLINE_PARAMS,
            true_values=pars_dotted,
            map_values=map_pars_dotted,
            sampler_info=sampler_info,
            include_derived=True,
            smooth=1.0,
            smooth1d=1.0,
        )
        fig_headline.savefig(
            output_dir / 'corner_headline.png', dpi=150, bbox_inches='tight'
        )
        plt.close(fig_headline)

        # Truth-vs-recovered + joint Nsigma
        fig_recov, recovery_stats = plot_recovery(
            result,
            pars_dotted,
            output_path=output_dir / 'truth_vs_recovered.png',
            sampler_name='numpyro NUTS',
        )
        plt.close(fig_recov)

        # Trace
        fig_trace = plot_trace(result)
        fig_trace.savefig(output_dir / 'trace.png', dpi=120, bbox_inches='tight')
        plt.close(fig_trace)

        # Text summary (truth, mean, std, R-hat, ESS, error)
        joint_nsigma = float(recovery_stats['joint_nsigma'])
        _save_summary_txt(
            output_dir / 'summary.txt',
            result,
            runtime_sec=runtime,
            true_pars=pars_dotted,
            joint_nsigma=joint_nsigma,
            sampler_config=sampler_config,
        )

        # ====================================================================
        # Pass criterion (single, from existing diagnostic convention)
        # ====================================================================
        print(f'Joint Nsigma = {joint_nsigma:.3f}')
        if joint_nsigma > 3.0:
            pytest.fail(
                f'Flagship recovery failed: joint Nsigma = {joint_nsigma:.2f} > 3.0. '
                f'See {output_dir} for diagnostics.'
            )
        elif joint_nsigma > 2.0:
            warnings.warn(
                f'Flagship recovery marginal: joint Nsigma = {joint_nsigma:.2f} '
                f'(>2, <=3). See {output_dir} for diagnostics.'
            )
