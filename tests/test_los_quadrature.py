"""LOS quadrature gates: tanh-substituted rule vs the windowed reference.

The inclined 3D models integrate rho = rho0 * radial(R) * sech^2(z/h_z)
along the line of sight. Two selectable rules (``los_quadrature``):

- 'tanh' (default): t = tanh(z/h_z) absorbs the vertical profile into the
  integration measure exactly; only the smooth radial factor is
  quadratured. No truncation window, no cosi clip. Per-class default node
  counts: 32 (exponential), 256 (cuspy Sersic/Spergel family).
- 'legendre' (reference): the original 200-point Gauss-Legendre rule on a
  truncated +/-5 h_z / max(cosi, 0.1) window. Retained for convergence
  comparisons; its window/clip construction loses flux near edge-on (a
  KNOWN model approximation, pinned by a canary below).

Every tolerance is measure-then-freeze: the comment carries the measured
value (2026-07-04, fp64, 48x48 @ 0.1"/pix) and the frozen bound gives
~3x headroom. These gates are the loud tripwire for the quadrature
default change -- if any fails, the LOS evaluation regressed; do NOT
loosen without re-running the A5 Fisher gate
(experiments/sweverett/production_speedups/a5_los_eval/).
"""

import itertools

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from kl_pipe.intensity import (
    BulgeDiskModel,
    InclinedExponentialModel,
    InclinedSersicModel,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.transformation import cen2source, obs2cen, source2gal
from kl_pipe.utils import build_map_grid_from_image_pars

_IP = ImagePars(shape=(48, 48), pixel_scale=0.1, indexing='ij')


@pytest.fixture(scope='module')
def grid():
    return build_map_grid_from_image_pars(_IP)


def _theta_exp(cosi, rscale, h_over_r):
    return jnp.array([cosi, 0.7, 0.02, -0.01, 100.0, rscale, h_over_r, 0.02, -0.03])


def _theta_sersic(cosi, hlr, h_over_hlr, n=4.0):
    return jnp.array([cosi, 0.7, 0.02, -0.01, 100.0, hlr, h_over_hlr, n, 0.02, -0.03])


def _max_rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def _flux_rel(a, b):
    return abs(float((jnp.sum(a) - jnp.sum(b)) / jnp.sum(b)))


def _l1_rel(a, b):
    return float(jnp.sum(jnp.abs(a - b)) / jnp.sum(jnp.abs(b)))


def test_invalid_los_quadrature_raises():
    with pytest.raises(ValueError, match="los_quadrature"):
        InclinedExponentialModel(los_quadrature='simpson')


def test_constructor_settings_honored():
    m = InclinedExponentialModel(n_quad=48)
    assert m._los_quadrature == 'tanh'
    assert m._n_quad == 48
    assert m._u_nodes.shape == (48,)
    m_ref = InclinedExponentialModel(los_quadrature='legendre')
    assert m_ref._n_quad == 200
    assert m_ref._gl_nodes.shape == (200,)
    # per-class tanh defaults: smooth exponential 32, cuspy Sersic 256
    assert InclinedExponentialModel()._n_quad == 32
    assert InclinedSersicModel()._n_quad == 256


def test_exponential_tanh_matches_windowed_reference(grid):
    """tanh-32 vs converged windowed rule (GL-512) where the window's cosi
    clip is inactive (cosi >= 0.2): both rules compute the same integral."""
    X, Y = grid
    m = InclinedExponentialModel()
    ref = InclinedExponentialModel(n_quad=512, los_quadrature='legendre')
    worst = 0.0
    for cosi, rs, hor in itertools.product(
        (0.2, 0.6, 0.95), (0.1, 0.5), (0.05, 0.1, 0.2)
    ):
        th = _theta_exp(cosi, rs, hor)
        worst = max(worst, _max_rel(m(th, 'obs', X, Y), ref(th, 'obs', X, Y)))
    # measured worst 1.07e-3 (at cosi=0.2, rs=0.1, hor=0.1); frozen at 3x
    assert worst < 3e-3, f"tanh-32 vs GL-512 worst max-rel {worst:.2e}"


def test_exponential_edge_on_self_convergence(grid):
    """Near edge-on (inside the windowed rule's clip zone) the tanh rule
    must be self-converged: tanh-32 vs tanh-512."""
    X, Y = grid
    m = InclinedExponentialModel()
    ref = InclinedExponentialModel(n_quad=512)
    for cosi in (0.06, 0.08):
        th = _theta_exp(cosi, 0.25, 0.1)
        err = _max_rel(m(th, 'obs', X, Y), ref(th, 'obs', X, Y))
        # measured 1.77e-3 (cosi 0.06) / 8.9e-4 (0.08); frozen at ~3x
        assert err < 5e-3, f"cosi={cosi}: tanh self-convergence {err:.2e}"


def test_windowed_rule_edge_on_flux_canary(grid):
    """CANARY: the legendre reference path's +/-5 h_z window with its
    cosi clip at 0.1 loses flux near edge-on relative to the true
    integral -- a KNOWN, documented model approximation (measured
    -3.6e-4 total flux at cosi=0.06, up to -9e-3 at compact scales).

    This test asserts the difference EXISTS. If it ever fails, the
    legendre path's window/clip semantics changed -- update the A5
    record and re-freeze the reference-path expectations."""
    X, Y = grid
    th = _theta_exp(0.06, 0.25, 0.1)
    windowed = InclinedExponentialModel(n_quad=512, los_quadrature='legendre')
    true = InclinedExponentialModel(n_quad=512)
    diff = _flux_rel(windowed(th, 'obs', X, Y), true(th, 'obs', X, Y))
    assert 5e-5 < diff < 2e-2, (
        f"windowed-rule edge-on flux deficit changed: {diff:.2e} "
        f"(expected ~3.6e-4; window/clip semantics may have been modified)"
    )


def test_thin_disk_closed_form(grid):
    """h_z -> 0 limit: the tanh rule is EXACT (sech^2 becomes the measure;
    the integrand goes constant), matching the analytic thin inclined
    disk I = flux/(2 pi rs^2 cosi) exp(-R_thin/rs). The windowed rule
    structurally cannot resolve this regime (its historical tolerance for
    h_over_r <= 0.01 was 300%)."""
    X, Y = grid
    cosi, rs = 0.6, 0.3
    th = _theta_exp(cosi, rs, 1e-3)
    img = InclinedExponentialModel()(th, 'obs', X, Y)
    xp, yp = obs2cen(0.02, -0.03, X, Y)
    xp, yp = cen2source(0.02, -0.01, xp, yp)
    xp, yp = source2gal(0.7, xp, yp)
    R_thin = jnp.sqrt(xp**2 + (yp / cosi) ** 2)
    thin = 100.0 / (2 * jnp.pi * rs**2 * cosi) * jnp.exp(-R_thin / rs)
    err = _max_rel(img, thin)
    # measured 1.4e-6 at h_over_r=1e-3 (residual is the O(h^2) thickness
    # effect itself, not quadrature error); frozen at ~7x
    assert err < 1e-5, f"thin-disk closed-form mismatch {err:.2e}"


def test_sersic_bulge_tanh256_beats_old_default(grid):
    """de Vaucouleurs (n=4) cusp: NO generic 1D quadrature fully converges
    (even tanh-2048 vs 4096 differ at ~1e-3). The gate is therefore
    two-part at the measured worst corner (cosi=0.2, hlr=0.5, hoh=0.2):

    1. absolute: tanh-256 vs a tanh-4096 reference within frozen bounds;
    2. dominance: tanh-256 must be at least as good as the OLD default
       (windowed GL-200) on max-rel, total-flux, and L1 metrics -- the
       default switch must not regress bulge accuracy anywhere.
    """
    X, Y = grid
    th = _theta_sersic(0.2, 0.5, 0.2)
    ref = InclinedSersicModel(n_quad=4096)(th, 'obs', X, Y)
    new = InclinedSersicModel()(th, 'obs', X, Y)
    old = InclinedSersicModel(los_quadrature='legendre')(th, 'obs', X, Y)

    # measured: tanh-256 maxrel 9.3e-3 / flux 1.3e-3 / L1 1.3e-3
    assert _max_rel(new, ref) < 3e-2
    assert _flux_rel(new, ref) < 4e-3
    # measured old GL-200: maxrel 1.0e-1 / flux 2.9e-3 / L1 1.6e-2
    assert _max_rel(new, ref) < _max_rel(old, ref)
    assert _flux_rel(new, ref) < _flux_rel(old, ref)
    assert _l1_rel(new, ref) < _l1_rel(old, ref)


def test_bulge_disk_composite_accuracy(grid):
    """BulgeDiskModel with tanh defaults vs a converged tanh-1024
    composite over a bulge_frac x hlr x cosi grid."""
    X, Y = grid
    bd = BulgeDiskModel()
    bd_ref = BulgeDiskModel(n_quad=1024)
    names = bd.PARAMETER_NAMES
    base = {
        'cosi': 0.5,
        'theta_int': 0.7,
        'g1': 0.02,
        'g2': -0.01,
        'total_flux': 100.0,
        'bulge_frac': 0.3,
        'disk_rscale': 0.3,
        'disk_h_over_r': 0.1,
        'disk_x0': 0.0,
        'disk_y0': 0.0,
        'bulge_hlr': 0.15,
        'bulge_h_over_hlr': 0.1,
        'bulge_x0': 0.0,
        'bulge_y0': 0.0,
    }
    worst = 0.0
    for cosi, bf, hlr in itertools.product((0.2, 0.6, 0.95), (0.1, 0.4), (0.1, 0.4)):
        d = dict(base, cosi=cosi, bulge_frac=bf, bulge_hlr=hlr)
        th = jnp.array([d[n] for n in names])
        worst = max(worst, _max_rel(bd(th, 'obs', X, Y), bd_ref(th, 'obs', X, Y)))
    # measured worst 2.8e-4 (old GL-200 default measured 3.2e-4 on the
    # same grid -- the switch is accuracy-neutral-or-better); frozen ~3x
    assert worst < 1e-3, f"bulge_disk tanh vs converged ref {worst:.2e}"


def test_gradient_equivalence_exponential(grid):
    """Sampler-relevant check: gradients of a scalar functional of the
    rendered map agree between the rules (vector norm at a perturbed
    theta -- per-component comparisons near zero-crossings are the known
    invalid metric)."""
    X, Y = grid
    th = jnp.array([0.57, 0.75, 0.028, -0.018, 103.0, 0.28, 0.1, 0.03, -0.02])

    def grad_of(model):
        return jax.grad(lambda t: jnp.sum(model(t, 'obs', X, Y) ** 2))(th)

    g_new = grad_of(InclinedExponentialModel())
    g_ref = grad_of(InclinedExponentialModel(los_quadrature='legendre'))
    rel = float(jnp.linalg.norm(g_new - g_ref) / jnp.linalg.norm(g_ref))
    # measured 2.1e-4; frozen at ~5x
    assert rel < 1e-3, f"gradient vector rel diff {rel:.2e}"


# =============================================================================
# Posterior-level A/B (short seeded run) -- slow tier
# =============================================================================


@pytest.mark.slow
class TestPosteriorEquivalence:
    def test_seeded_posterior_ab_tanh_vs_legendre(self):
        """Seeded NUTS with identical data through both quadrature rules:
        posterior means shift < 0.1 sigma per parameter, widths within
        20%. Data is generated with the LEGENDRE (old default) source so
        the A/B measures the default switch against prior behavior.
        Screening config (24x24, 4 sampled params, 1 chain), mirroring
        the psf_mode protocol test.

        Sample count is sized to the gate: at 300 samples the per-param
        MC error on a mean shift is ~0.06 sigma, and a 4-seed sweep
        showed the worst-of-4-params statistic crosses the 0.1 gate on
        ~half of seeds with NO code change at all. 1500 samples brings
        the per-param MC error to ~0.027 sigma so a gate crossing
        indicates a real quadrature difference, not sampler noise."""
        import galsim

        from kl_pipe.dispersion import GrismPars
        from kl_pipe.lines import LINE_LAMBDAS, EmissionLine
        from kl_pipe.observation import build_grism_obs
        from kl_pipe.priors import PriorDict, Uniform
        from kl_pipe.render import RenderConfig
        from kl_pipe.sampling import InferenceTask, build_sampler
        from kl_pipe.sampling.configs import NumpyroSamplerConfig
        from kl_pipe.source import SourceModel
        from kl_pipe.velocity import CenteredVelocityModel

        z = 1.0
        image_pars = ImagePars(shape=(24, 24), pixel_scale=0.11, indexing='ij')
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=LINE_LAMBDAS['Halpha'] * (1 + z),
            dispersion_angle_detector=0.0,
        )
        base_pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.02,
            'g2': -0.03,
            'z': z,
            'vel.v0': 10.0,
            'vel.vcirc': 200.0,
            'vel.rscale': 0.5,
            'Halpha.flux': 100.0,
            'Halpha.rscale': 0.3,
            'Halpha.h_over_r': 0.1,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
            'Halpha.dispersion': 50.0,
        }

        def make_source(los_quadrature):
            return SourceModel(
                velocity_model=CenteredVelocityModel(),
                emission_lines={
                    'Halpha': EmissionLine(
                        intensity=InclinedExponentialModel(
                            los_quadrature=los_quadrature
                        )
                    )
                },
            )

        source_old = make_source('legendre')
        source_new = make_source('tanh')

        obs_clean = build_grism_obs(
            gp,
            z=z,
            psf=galsim.Gaussian(fwhm=0.11),
            render_config=RenderConfig(oversample=3),
        )
        truth = source_old.render_grism(base_pars, obs_clean)
        peak = float(jnp.max(truth))
        sigma_noise = peak / 50.0  # SNR ~ 50 at the peak
        rng = np.random.default_rng(20260704)
        data = np.asarray(truth) + rng.normal(0.0, sigma_noise, truth.shape)
        obs = build_grism_obs(
            gp,
            z=z,
            psf=galsim.Gaussian(fwhm=0.11),
            render_config=RenderConfig(oversample=3),
            data=jnp.asarray(data),
            variance=sigma_noise**2,
        )

        sampled = {
            'cosi': Uniform(0.2, 0.8),
            'g1': Uniform(-0.1, 0.1),
            'g2': Uniform(-0.1, 0.1),
            'vel.vcirc': Uniform(100.0, 300.0),
        }
        priors = PriorDict(
            {
                **sampled,
                **{k: v for k, v in base_pars.items() if k not in sampled},
            }
        )
        config = NumpyroSamplerConfig(
            n_samples=1500, n_warmup=300, n_chains=1, seed=42, progress=False
        )

        results = {}
        for label, source in (('legendre', source_old), ('tanh', source_new)):
            task = InferenceTask.from_obs(source, priors, grism_obs={'roll0': obs})
            results[label] = build_sampler('numpyro', task, config).run()

        res_a, res_b = results['legendre'], results['tanh']
        assert res_a.param_names == res_b.param_names
        report = []
        for i, name in enumerate(res_a.param_names):
            mean_a = float(res_a.samples[:, i].mean())
            mean_b = float(res_b.samples[:, i].mean())
            std_a = float(res_a.samples[:, i].std())
            std_b = float(res_b.samples[:, i].std())
            shift_sigma = abs(mean_b - mean_a) / std_a
            width_ratio = std_b / std_a
            report.append(
                f"{name}: shift {shift_sigma:.3f} sigma, width x{width_ratio:.2f}"
            )
            # 0.1 sigma screening threshold (MC error ~0.06 sigma at 300
            # samples), matching the psf_mode posterior A/B protocol
            assert shift_sigma < 0.1, (
                f"{name}: posterior shift {shift_sigma:.3f} sigma between "
                f"quadrature rules (means {mean_a:.5g} vs {mean_b:.5g})"
            )
            assert (
                0.8 < width_ratio < 1.25
            ), f"{name}: posterior width changed x{width_ratio:.2f}"
        print("; ".join(report))
