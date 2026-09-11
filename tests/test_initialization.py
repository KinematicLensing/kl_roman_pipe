"""Unit tests for the fit-initialization toolkit (kl_pipe.sampling.initialization).

Covers: start sets and their historical PRNG streams, adaptive image moments
(exact on a Gaussian, floored on a sub-pixel object, loud on an empty
image), moment estimates against a rendered exponential disk (size,
inclination, position-angle convention, centroid, flux), the MAP finder
(the keyword wrapper and the piecewise route reproduce each other
exactly, the historical procedure stays reachable by explicit values, basin
clustering), the two eigenvalue-floor rules, the chain initial points and
the one-call ``InitConfig`` / ``Initializer`` / ``InitResult`` API.
"""

import jax

jax.config.update('jax_enable_x64', True)

import galsim  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kl_pipe.intensity import InclinedExponentialModel  # noqa: E402
from kl_pipe.observation import build_image_obs, build_velocity_obs  # noqa: E402
from kl_pipe.parameters import ImagePars  # noqa: E402
from kl_pipe.priors import (  # noqa: E402
    CircularUniform,
    Gaussian,
    LogNormal,
    PriorDict,
    TruncatedNormal,
    Uniform,
)
from kl_pipe.sampling.initialization import (  # noqa: E402
    adaptive_moments,
    build_preconditioner,
    chain_inits,
    clip_into_support,
    cluster_basins,
    combine_starts,
    EigenFloor,
    find_map,
    ImageMoments,
    InitConfig,
    initialization_columns,
    Initializer,
    InitResult,
    laplace_metric,
    moment_estimates,
    moment_starts,
    MomentsError,
    newton_polish,
    pa_stratified_starts,
    prior_starts,
    psf_moments,
    StartSet,
    support_bounds_scaled,
)
from kl_pipe.sampling.initialization import _THICK_DISK_Q0_PER_H  # noqa: E402
from kl_pipe.sampling.task import InferenceTask, LaplacePreconditioner  # noqa: E402
from kl_pipe.sampling.transforms import UnconstrainingTransform  # noqa: E402
from kl_pipe.source import SourceModel  # noqa: E402
from kl_pipe.synthetic import SyntheticVelocity  # noqa: E402
from kl_pipe.velocity import CenteredVelocityModel  # noqa: E402


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope='module')
def velocity_task():
    """Velocity-only task at SNR 1000 (same construction as test_numpyro)."""
    image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')
    true_flat = {
        'v0': 10.0,
        'vcirc': 200.0,
        'rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }
    synth = SyntheticVelocity(true_flat, model_type='arctan', seed=42)
    data = synth.generate(image_pars, snr=1000)
    priors = PriorDict(
        {
            'vel.v0': Gaussian(10.0, 5.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel.rscale': TruncatedNormal(5.0, 2.0, 0.4, 20.0),
            'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
            'theta_int': TruncatedNormal(0.785, 0.3, 0, np.pi),
            'g1': 0.02,
            'g2': -0.01,
        }
    )
    obs = build_velocity_obs(image_pars, data=jnp.array(data), variance=synth.variance)
    return InferenceTask.from_obs(
        SourceModel(velocity_model=CenteredVelocityModel()), priors, velocity_obs=obs
    )


IMAGE_TRUTH = {
    'cosi': 0.5,
    'theta_int': 0.7,
    'g1': 0.0,
    'g2': 0.0,
    'F087.flux': 100.0,
    'F087.rscale': 0.35,
    'F087.h_over_r': 0.15,
    'F087.x0': 0.12,
    'F087.y0': -0.08,
}


@pytest.fixture(scope='module')
def image_task():
    """Single-band inclined exponential with a Gaussian PSF, near-noiseless
    data (noise 1e-3 of the peak). Returns (task, image_obs dict)."""
    image_pars = ImagePars(shape=(40, 40), pixel_scale=0.1, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.18)
    src = SourceModel(broadband_models={'F087': InclinedExponentialModel()})
    priors = PriorDict(
        {
            'cosi': Uniform(0.05, 0.95),
            'theta_int': CircularUniform(),
            'g1': Gaussian(0.0, 0.05),
            'g2': Gaussian(0.0, 0.05),
            'F087.flux': Uniform(10.0, 500.0),
            'F087.rscale': LogNormal(np.log(0.3), 0.5),
            'F087.h_over_r': 0.15,
            'F087.x0': TruncatedNormal(0.0, 0.15, -0.5, 0.5),
            'F087.y0': TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        }
    )
    clean_obs = build_image_obs(image_pars, psf=psf, broadband_key='F087')
    clean = np.asarray(src.render_broadband(IMAGE_TRUTH, clean_obs, 'F087'))
    sigma = 1e-3 * clean.max()
    data = clean + np.random.RandomState(3).normal(0.0, sigma, clean.shape)
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.asarray(data),
        variance=sigma**2,
        broadband_key='F087',
    )
    task = InferenceTask.from_obs(src, priors, image_obs={'F087': obs})
    return task, {'F087': obs}


# ==============================================================================
# Start sets
# ==============================================================================


class TestStartSet:
    def test_combine_and_families(self):
        a = StartSet(np.zeros((2, 3)), ['prior'] * 2, ('a', 'b', 'c'))
        b = StartSet(np.ones((1, 3)), ['moments'], ('a', 'b', 'c'))
        c = combine_starts(a, None, b)
        assert len(c) == 3
        assert c.families() == {'prior': 2, 'moments': 1}
        assert c.labels == ['prior', 'prior', 'moments']

    def test_shape_and_label_validation(self):
        with pytest.raises(ValueError, match='columns'):
            StartSet(np.zeros((2, 2)), ['x', 'x'], ('a', 'b', 'c'))
        with pytest.raises(ValueError, match='labels'):
            StartSet(np.zeros((2, 3)), ['x'], ('a', 'b', 'c'))
        with pytest.raises(ValueError, match='finite'):
            StartSet(np.full((1, 3), np.nan), ['x'], ('a', 'b', 'c'))
        with pytest.raises(ValueError, match='names differ'):
            combine_starts(
                StartSet(np.zeros((1, 2)), ['x'], ('a', 'b')),
                StartSet(np.zeros((1, 2)), ['x'], ('a', 'c')),
            )


class TestPriorAndPAStarts:
    def test_prior_starts_use_historical_stream(self, velocity_task):
        s = prior_starts(velocity_task, 3, seed=11)
        expected = np.asarray(
            velocity_task.sample_prior(jax.random.PRNGKey(12), n_samples=3)
        )
        np.testing.assert_array_equal(s.points, expected)
        assert s.labels == ['prior'] * 3
        assert s.names == tuple(velocity_task.sampled_names)

    def test_pa_stratified_bounded_prior(self, velocity_task):
        s = pa_stratified_starts(velocity_task.priors, n_pa=4, seed=7)
        i = list(velocity_task.sampled_names).index('theta_int')
        np.testing.assert_allclose(s.points[:, i], (np.arange(4) + 0.5) * np.pi / 4)
        assert s.labels == ['pa_stratified'] * 4

    def test_pa_stratified_periodic_prior_covers_both_directions(self, image_task):
        task, _ = image_task
        s = pa_stratified_starts(task.priors, n_pa=4, seed=7)
        i = list(task.sampled_names).index('theta_int')
        assert len(s) == 8
        np.testing.assert_allclose(s.points[:, i], (np.arange(8) + 0.5) * np.pi / 4)

    def test_pa_stratified_none_without_theta(self):
        priors = PriorDict({'a': Uniform(0.0, 1.0)})
        assert pa_stratified_starts(priors) is None

    def test_clip_into_support(self, image_task):
        task, _ = image_task
        names = list(task.sampled_names)
        theta = np.zeros(len(names))
        theta[names.index('cosi')] = 5.0
        theta[names.index('F087.rscale')] = -1.0
        out = clip_into_support(task, theta)
        assert 0.05 < out[names.index('cosi')] < 0.95
        assert out[names.index('F087.rscale')] > 0
        # unbounded parameters untouched
        assert out[names.index('g1')] == 0.0


# ==============================================================================
# Image moments
# ==============================================================================


def _grid(n=64, scale=0.1):
    x = (np.arange(n) - n / 2 + 0.5) * scale
    Y, X = np.meshgrid(x, x, indexing='ij')  # x along columns, y along rows
    return X, Y


class TestAdaptiveMoments:
    def test_exact_on_elliptical_gaussian(self):
        X, Y = _grid()
        x0, y0, sa, sb, pa = 0.23, -0.31, 0.5, 0.25, 0.6
        c, s = np.cos(pa), np.sin(pa)
        u = (X - x0) * c + (Y - y0) * s
        v = -(X - x0) * s + (Y - y0) * c
        img = np.exp(-0.5 * (u**2 / sa**2 + v**2 / sb**2))
        m = adaptive_moments(img, X, Y)
        assert m.converged
        assert abs(m.x0 - x0) < 1e-3 and abs(m.y0 - y0) < 1e-3
        # pixelization adds p^2/12 per axis to the second moments
        p2 = 0.1**2 / 12
        assert abs(m.sigma_major - np.sqrt(sa**2 + p2)) / sa < 0.01
        assert abs(m.sigma_minor - np.sqrt(sb**2 + p2)) / sb < 0.01
        assert abs(m.pa - pa) < 0.01
        assert abs(m.flux - img.sum()) < 1e-12

    def test_sub_pixel_object_is_floored_not_collapsed(self):
        X, Y = _grid()
        img = np.exp(-0.5 * ((X / 0.6) ** 2 + (Y / 0.01) ** 2))
        m = adaptive_moments(img, X, Y)
        assert m.sigma_minor >= 0.5 * 0.1 - 1e-12
        assert 0.55 < m.sigma_major < 0.65

    def test_empty_image_raises(self):
        X, Y = _grid()
        with pytest.raises(MomentsError, match='weighted flux'):
            adaptive_moments(np.zeros_like(X), X, Y)
        with pytest.raises(MomentsError, match='no valid pixels'):
            adaptive_moments(np.ones_like(X), X, Y, mask=np.zeros_like(X, bool))

    def test_deconvolution_floor(self):
        obj = ImageMoments(0.0, 0.0, 1.0, 0.04, 0.0, 0.01, 3)
        psf = ImageMoments(0.0, 0.0, 1.0, 0.02, 0.0, 0.02, 3)
        d = obj.deconvolved(psf)
        assert abs(d.m_xx - 0.02) < 1e-12
        # minor axis narrower than the PSF: floored at 5% of the observed moment
        assert abs(d.m_yy - 0.05 * 0.01) < 1e-12


class TestMomentEstimates:
    """Budgets: measured on the 32 noiseless cosmos25_bank32 renders
    (2026-09-10): deconvolved sigma/rscale 1.31-1.56 around the 1.43
    calibration (+-10%), q - cosi corrected by the thick-disk term, PA within
    0.08 rad of theta_int, centroid bias 0.015" (0.14 px)."""

    def test_recovers_truth_within_budgets(self, image_task):
        task, image_obs = image_task
        est = moment_estimates(task, image_obs)
        assert abs(est.rscale / IMAGE_TRUTH['F087.rscale'] - 1.0) < 0.10
        assert abs(est.cosi - IMAGE_TRUTH['cosi']) < 0.10
        dpa = abs((est.pa - IMAGE_TRUTH['theta_int'] + np.pi / 2) % np.pi - np.pi / 2)
        assert dpa < 0.08
        assert abs(est.x0 - IMAGE_TRUTH['F087.x0']) < 0.05
        assert abs(est.y0 - IMAGE_TRUTH['F087.y0']) < 0.05
        assert abs(est.flux['F087'] / IMAGE_TRUTH['F087.flux'] - 1.0) < 0.05
        assert est.q0 == pytest.approx(_THICK_DISK_Q0_PER_H * 0.15)

    def test_psf_moments_match_kernel_width(self, image_task):
        _, image_obs = image_task
        m = psf_moments(image_obs['F087'])
        sigma_psf = 0.18 / (2 * np.sqrt(2 * np.log(2)))
        assert abs(m.sigma_major - sigma_psf) / sigma_psf < 0.05
        assert abs(m.sigma_minor - sigma_psf) / sigma_psf < 0.05
        assert abs(m.x0) < 0.01 and abs(m.y0) < 0.01

    def test_moment_starts_rows(self, image_task):
        task, image_obs = image_task
        s = moment_starts(task, image_obs, seed=0)
        names = list(task.sampled_names)
        assert len(s) == 2 and s.labels == ['moments', 'moments']
        ith = names.index('theta_int')
        # both rotation directions, half a turn apart on the circle
        assert abs(abs(s.points[0, ith] - s.points[1, ith]) - np.pi) < 1e-9
        for i, name in enumerate(names):
            if name in ('g1', 'g2'):
                assert s.points[:, i].tolist() == [0.0, 0.0]
        for i, (lo, hi) in enumerate(task.get_bounds()):
            if lo is not None:
                assert (s.points[:, i] > lo).all()
            if hi is not None:
                assert (s.points[:, i] < hi).all()
        # rows differ in theta_int only
        other = [i for i in range(len(names)) if i != ith]
        np.testing.assert_array_equal(s.points[0, other], s.points[1, other])

    def test_moment_starts_requires_images(self, image_task):
        task, _ = image_task
        with pytest.raises(ValueError, match='at least one'):
            moment_estimates(task, {})


# ==============================================================================
# MAP finder
# ==============================================================================


class TestFindMap:
    def test_wrapper_matches_toolkit_route_at_defaults(self, velocity_task):
        """``laplace_preconditioner`` and the piecewise route must agree
        exactly at the shared defaults (robust procedure: bounded search,
        8-step polish of 3 basins, prior floor 0.5; jobs 988356 / 988824 /
        990891)."""
        pre = velocity_task.laplace_preconditioner(n_starts=3, seed=0)
        starts = prior_starts(velocity_task, 3, seed=0)
        r = find_map(velocity_task, starts, seed=0)
        np.testing.assert_array_equal(r.theta_map, pre.map_point)
        assert r.n_converged == pre.n_starts_converged
        built = build_preconditioner(velocity_task, r)
        np.testing.assert_array_equal(
            built.inverse_mass_matrix, pre.inverse_mass_matrix
        )
        assert built.start_labels == ['prior'] * len(r.labels)
        assert built.basin_points.shape[1] == velocity_task.n_params
        assert pre.eig_floor_mode == 'prior' and pre.eig_floor_value == 0.5
        assert np.isfinite(pre.map_grad_norm) and pre.polish_gain >= 0

    def test_historical_procedure_by_explicit_values(self, velocity_task):
        """Passing the pre-2026-09-11 values (unbounded L-BFGS, no polish,
        relative floor 1e-4) reproduces that procedure: same endpoints as the
        unbounded ``find_map`` call, relative-floor metric with the condition
        number capped at 1e4, no polish records."""
        pre = velocity_task.laplace_preconditioner(
            n_starts=3,
            seed=0,
            bounded=False,
            polish_steps=0,
            eig_floor_mode='relative',
            eig_floor=1e-4,
        )
        starts = prior_starts(velocity_task, 3, seed=0)
        r = find_map(velocity_task, starts, seed=0, bounded=False, polish_steps=0)
        np.testing.assert_array_equal(r.theta_map, pre.map_point)
        np.testing.assert_array_equal(r.end_points, pre.start_map_points)
        built = build_preconditioner(
            velocity_task, r, floor=EigenFloor('relative', 1e-4)
        )
        np.testing.assert_array_equal(
            built.inverse_mass_matrix, pre.inverse_mass_matrix
        )
        assert pre.eig_floor_mode == 'relative' and pre.eig_floor_value == 1e-4
        assert pre.condition_number <= 1e4 * (1 + 1e-9)
        assert np.isnan(pre.map_grad_norm) and pre.polish_gain == 0.0
        # the robust default reaches at least as good an objective
        robust = velocity_task.laplace_preconditioner(n_starts=3, seed=0)
        assert robust.start_neg_logposts.min() <= pre.start_neg_logposts.min() + 1e-9

    def test_result_records_and_summary(self, velocity_task):
        starts = combine_starts(
            prior_starts(velocity_task, 2, seed=1),
            pa_stratified_starts(velocity_task.priors, n_pa=2, seed=1),
        )
        r = find_map(velocity_task, starts, seed=1)
        assert r.end_points.shape == (len(r.labels), velocity_task.n_params)
        assert r.objectives.min() == r.neg_logpost
        assert r.basin_ids[r.best_index] == 0
        assert r.winning_label in ('prior', 'pa_stratified')
        assert r.n_evals > 0 and r.wall_s > 0
        rows = r.summary_rows()
        assert rows[0]['gap_to_map'] == 0.0
        text = r.format_summary()
        assert 'basin(s)' in text and 'winning start family' in text
        if r.n_basins == 1:
            assert r.basin_margin == np.inf

    def test_name_mismatch_raises(self, velocity_task):
        bad = StartSet(np.zeros((1, velocity_task.n_params)), ['x'], tuple('abcde'))
        with pytest.raises(ValueError, match='names'):
            find_map(velocity_task, bad)


class TestClusterBasins:
    def test_two_basins_and_periodic_wrap(self):
        scale = np.array([1.0, 1.0])
        pts = np.array([[0.0, 0.1], [0.05, 0.1], [3.0, 0.1], [6.2, 0.1]])
        obj = np.array([5.0, 4.0, 10.0, 4.5])
        ids, bp, bo = cluster_basins(pts, obj, scale, [2 * np.pi, None], tol=0.25)
        # 6.2 wraps to -0.08 on the circle: same basin as 0.0 / 0.05
        assert ids.tolist() == [0, 0, 1, 0]
        np.testing.assert_array_equal(bp[0], [0.05, 0.1])
        assert bo.tolist() == [4.0, 10.0]

    def test_basins_ordered_by_objective(self):
        pts = np.array([[0.0], [10.0], [20.0]])
        obj = np.array([3.0, 1.0, 2.0])
        ids, bp, bo = cluster_basins(pts, obj, np.ones(1), [None])
        assert ids.tolist() == [2, 0, 1]
        assert bo.tolist() == [1.0, 2.0, 3.0]


# ==============================================================================
# Eigenvalue floor and Laplace metric
# ==============================================================================


class TestEigenFloor:
    def test_defaults_and_validation(self):
        # default rule = prior-unit floor 0.5 (988824 / 990891); the relative
        # rule keeps its historical 1e-4
        assert EigenFloor().mode == 'prior' and EigenFloor().value == 0.5
        assert EigenFloor('relative').value == 1e-4
        assert EigenFloor('prior').value == 0.5
        assert EigenFloor('prior', 0.2).value == 0.2
        with pytest.raises(ValueError, match='mode'):
            EigenFloor('bogus')
        with pytest.raises(ValueError, match='positive'):
            EigenFloor('relative', -1.0)

    def test_threshold_rules(self):
        w = np.array([0.01, 1.0, 1e4])
        assert EigenFloor('relative', 1e-4).threshold(w) == 1.0
        assert EigenFloor('prior', 0.5).threshold(w) == 0.5

    def test_prior_floor_leaves_constrained_directions_alone(self, velocity_task):
        """At SNR 1000 every direction is far stiffer than its prior, so the
        prior floor is inactive and the metric is the exact inverse Hessian,
        while the relative floor may clip the softest directions."""
        starts = prior_starts(velocity_task, 2, seed=0)
        r = find_map(velocity_task, starts, seed=0)
        m_prior = laplace_metric(
            velocity_task, r.theta_map, r.scale, floor=EigenFloor('prior')
        )
        assert m_prior.n_floored == 0
        assert m_prior.eigenvalues.min() > 0.5
        m_rel = laplace_metric(
            velocity_task, r.theta_map, r.scale, floor=EigenFloor('relative')
        )
        assert m_rel.condition_number <= 1e4 * (1 + 1e-9)
        assert m_rel.n_negative == m_prior.n_negative
        # both metrics symmetric positive definite
        for m in (m_prior, m_rel):
            im = m.inverse_mass_matrix
            assert np.allclose(im, im.T)
            assert np.all(np.linalg.eigvalsh(im) > 0)
        # with no clipping the prior-floor metric is the plain inverse Hessian
        Hn = (m_prior.eigenvectors * m_prior.eigenvalues) @ m_prior.eigenvectors.T
        inv_expected = (r.scale[:, None] * np.linalg.inv(Hn)) * r.scale[None, :]
        np.testing.assert_allclose(m_prior.inverse_mass_matrix, inv_expected, rtol=1e-8)

    def test_wrapper_exposes_floor_mode(self, velocity_task):
        pre = velocity_task.laplace_preconditioner(
            n_starts=2, seed=0, eig_floor_mode='prior', eig_floor=0.5
        )
        assert pre.eig_floor_mode == 'prior' and pre.eig_floor_value == 0.5
        assert pre.n_floored_eigenvalues == 0
        with pytest.raises(ValueError, match='mode'):
            velocity_task.laplace_preconditioner(n_starts=2, eig_floor_mode='bogus')


# ==============================================================================
# Chain initial points
# ==============================================================================


def _fake_pre(n_params=3, basins=((0.0, 5.0, 30.0),)):
    """Preconditioner stub with the MAP at zero and basins along axis 0."""
    objectives = np.array(basins[0])
    points = np.zeros((len(objectives), n_params))
    points[:, 0] = np.arange(len(objectives)) * 2.0
    return LaplacePreconditioner(
        map_point=np.zeros(n_params),
        inverse_mass_matrix=np.diag([1.0, 4.0, 9.0]),
        n_starts_converged=1,
        condition_number=9.0,
        basin_points=points,
        basin_neg_logposts=objectives,
    )


class TestChainInits:
    def test_jitter_reproduces_historical_formula(self):
        pre = _fake_pre()
        inits = chain_inits(pre, pre.inverse_mass_matrix, 4, seed=5)
        noise = np.asarray(jax.random.normal(jax.random.PRNGKey(5), (4, 3)))
        expected = 0.01 * np.sqrt(np.diag(pre.inverse_mass_matrix))[None, :] * noise
        np.testing.assert_allclose(inits, expected)

    def test_single_chain_sits_at_map(self):
        pre = _fake_pre()
        inits = chain_inits(pre, pre.inverse_mass_matrix, 1, mode='map_basins')
        np.testing.assert_array_equal(inits, np.zeros((1, 3)))

    def test_basins_mode_places_competing_basins(self):
        pre = _fake_pre()
        inits = chain_inits(
            pre, pre.inverse_mass_matrix, 4, mode='map_basins', seed=5, max_margin=20.0
        )
        np.testing.assert_array_equal(inits[0], np.zeros(3))  # exact MAP
        np.testing.assert_array_equal(inits[1], [2.0, 0.0, 0.0])  # 5-nat basin
        # the 30-nat basin is beyond the margin: chains 2 and 3 are jittered
        assert not np.array_equal(inits[2], [4.0, 0.0, 0.0])
        assert np.all(np.abs(inits[2:]) < 0.5)

    def test_basins_mode_needs_records(self):
        pre = _fake_pre()
        pre.basin_points = None
        with pytest.raises(ValueError, match='basin records'):
            chain_inits(pre, pre.inverse_mass_matrix, 2, mode='map_basins')
        with pytest.raises(ValueError, match='mode'):
            chain_inits(pre, pre.inverse_mass_matrix, 2, mode='bogus')

    def test_transform_maps_to_sampling_coordinates(self):
        priors = PriorDict(
            {'a': Uniform(-1.0, 1.0), 'b': Uniform(-1.0, 1.0), 'c': Uniform(-9.0, 9.0)}
        )
        transform = UnconstrainingTransform.from_priors(priors)
        pre = _fake_pre(basins=((0.0, 5.0),))
        pre.basin_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        inits = chain_inits(pre, np.eye(3), 2, mode='map_basins', transform=transform)
        eta_map, _ = transform.forward_clipped(np.zeros(3), u_margin=1e-6)
        eta_b1, _ = transform.forward_clipped(np.array([0.5, 0.0, 0.0]), u_margin=1e-6)
        np.testing.assert_allclose(inits[0], eta_map)
        np.testing.assert_allclose(inits[1], eta_b1)


# ==============================================================================
# Bounded search and Newton polish
# ==============================================================================


class TestBoundedAndPolish:
    def test_support_bounds_scaled(self, image_task):
        task, _ = image_task
        starts = prior_starts(task, 1, seed=0)
        r = find_map(task, starts, seed=0, maxiter=5)
        b = support_bounds_scaled(task, r.loc, r.scale)
        names = list(task.sampled_names)
        lo, hi = b[names.index('cosi')]
        # bounds map back to just inside the prior support
        assert (
            0.05
            < r.loc[names.index('cosi')] + r.scale[names.index('cosi')] * lo
            < 0.051
        )
        assert (
            0.949
            < r.loc[names.index('cosi')] + r.scale[names.index('cosi')] * hi
            < 0.95
        )
        assert b[names.index('theta_int')] == (None, None)
        assert b[names.index('F087.rscale')][1] is None
        assert b[names.index('F087.rscale')][0] is not None

    def test_bounded_search_stays_in_support(self, velocity_task):
        starts = prior_starts(velocity_task, 3, seed=0)
        r = find_map(velocity_task, starts, seed=0, bounded=True)
        for i, (lo, hi) in enumerate(velocity_task.get_bounds()):
            if lo is not None:
                assert (r.end_points[:, i] > lo).all()
            if hi is not None:
                assert (r.end_points[:, i] < hi).all()
        # the same mode as the unbounded search on this well-posed task
        r0 = find_map(velocity_task, starts, seed=0)
        assert abs(r.neg_logpost - r0.neg_logpost) < 1e-3

    def test_polish_reaches_stationary_point(self, velocity_task):
        """From a deliberately displaced point the polish must descend to a
        local maximum: small scaled gradient, positive-definite Hessian, and
        the same objective as the L-BFGS MAP."""
        starts = prior_starts(velocity_task, 2, seed=0)
        r = find_map(velocity_task, starts, seed=0)
        theta0 = r.theta_map + 0.3 * r.scale
        theta, f, gn, me, taken = newton_polish(
            velocity_task, theta0, r.scale, n_steps=20
        )
        assert taken >= 1
        assert f <= r.neg_logpost + 1e-6
        assert gn < 1e-3
        assert me > 0

    def test_find_map_polish_records(self, velocity_task):
        starts = prior_starts(velocity_task, 2, seed=0)
        r0 = find_map(velocity_task, starts, seed=0, polish_steps=0)
        r = find_map(velocity_task, starts, seed=0, polish_steps=5, polish_basins=2)
        assert r.neg_logpost <= r0.neg_logpost + 1e-9
        assert r.polish_gain >= 0
        assert np.isfinite(r.map_grad_norm) and np.isfinite(r.map_min_eigenvalue)
        assert r.n_evals > r0.n_evals
        pre = build_preconditioner(velocity_task, r)
        assert pre.map_grad_norm == r.map_grad_norm
        assert pre.polish_gain == r.polish_gain
        with pytest.raises(ValueError, match='polish'):
            find_map(velocity_task, starts, polish_steps=-1)

    def test_wrapper_passes_polish(self, velocity_task):
        pre = velocity_task.laplace_preconditioner(
            n_starts=2, seed=0, bounded=True, polish_steps=3
        )
        assert np.isfinite(pre.map_grad_norm)
        assert pre.n_negative_eigenvalues == 0


# ==============================================================================
# One-call API: InitConfig / Initializer / InitResult
# ==============================================================================


class TestInitializerAPI:
    def test_config_defaults_and_validation(self):
        cfg = InitConfig()
        assert (cfg.map_bounded, cfg.map_polish_steps, cfg.map_polish_basins) == (
            True,
            8,
            3,
        )
        assert cfg.eig_floor_mode == 'prior' and cfg.floor.value == 0.5
        assert cfg.chain_init == 'map_jitter' and cfg.map_moment_starts is False
        assert InitConfig(eig_floor_mode='relative').floor.value == 1e-4
        for bad in (
            dict(n_map_starts=0),
            dict(n_pa_starts=-1),
            dict(map_polish_steps=-1),
            dict(map_polish_basins=0),
            dict(hessian_method='bogus'),
            dict(chain_init='bogus'),
            dict(chain_init_max_margin=0.0),
            dict(eig_floor_mode='bogus'),
            dict(eig_floor=-1.0),
            dict(maxiter=0),
        ):
            with pytest.raises(ValueError):
                InitConfig(**bad)

    def test_run_matches_wrapper(self, velocity_task):
        """``Initializer.run()`` with the wrapper's start set must give the
        wrapper's preconditioner bit for bit; the stepwise methods give the
        same objects as ``run``."""
        cfg = InitConfig(n_map_starts=3, n_pa_starts=0)
        init = Initializer(velocity_task, cfg, seed=0)
        res = init.run()
        assert isinstance(res, InitResult)
        pre = velocity_task.laplace_preconditioner(n_starts=3, seed=0)
        np.testing.assert_array_equal(res.preconditioner.map_point, pre.map_point)
        np.testing.assert_array_equal(
            res.preconditioner.inverse_mass_matrix, pre.inverse_mass_matrix
        )
        assert res.starts.families() == {'prior': 3}
        assert res.metric.n_floored == res.preconditioner.n_floored_eigenvalues
        assert res.metric.condition_number == res.preconditioner.condition_number
        assert res.moment_starts_ok is None
        # stepwise route
        starts = init.starts()
        r = init.find_map(starts)
        np.testing.assert_array_equal(r.theta_map, res.map.theta_map)
        m = init.metric(r)
        np.testing.assert_array_equal(
            m.inverse_mass_matrix, res.metric.inverse_mass_matrix
        )
        np.testing.assert_array_equal(
            init.preconditioner(r, m).inverse_mass_matrix,
            res.preconditioner.inverse_mass_matrix,
        )

    def test_pa_starts_and_columns(self, velocity_task):
        res = Initializer(velocity_task, InitConfig(n_map_starts=2), seed=1).run()
        # bounded theta prior on this task: n_pa_starts grid points
        assert res.starts.families() == {'prior': 2, 'pa_stratified': 4}
        cols = res.columns()
        assert set(cols) == {
            'map_n_basins',
            'map_basin_margin',
            'map_winning_start',
            'map_moment_starts_ok',
            'precond_n_floored_eigenvalues',
            'precond_eig_floor_mode',
            'map_grad_norm',
            'map_min_eigenvalue',
            'map_polish_gain',
            'chain_init',
        }
        assert cols['map_n_basins'] == res.map.n_basins
        assert cols['map_winning_start'] in ('prior', 'pa_stratified')
        assert cols['map_moment_starts_ok'] == 'n/a'
        assert cols['precond_eig_floor_mode'] == 'prior'
        assert cols['chain_init'] == 'map_jitter'
        assert cols == initialization_columns(res.preconditioner, None, 'map_jitter')
        inv = res.preconditioner.inverse_mass_matrix
        pts = res.chain_points(inv, 4, seed=3)
        assert pts.shape == (4, velocity_task.n_params)
        np.testing.assert_allclose(pts, chain_inits(res.preconditioner, inv, 4, seed=3))

    def test_moment_starts_need_images_and_record_failure(self, image_task):
        import dataclasses

        task, image_obs = image_task
        with pytest.raises(ValueError, match='image_obs'):
            Initializer(task, InitConfig(map_moment_starts=True), seed=0)
        cfg = InitConfig(n_map_starts=1, map_moment_starts=True)
        init = Initializer(task, cfg, seed=0, image_obs=image_obs)
        starts = init.starts()
        assert init.moment_starts_ok is True and starts.families()['moments'] == 2
        # an empty stamp yields no moments: warning, flag False, search goes on
        obs = image_obs['F087']
        empty = {'F087': dataclasses.replace(obs, data=jnp.zeros_like(obs.data))}
        init_empty = Initializer(task, cfg, seed=0, image_obs=empty)
        with pytest.warns(RuntimeWarning, match='moment starts unavailable'):
            starts = init_empty.starts()
        assert init_empty.moment_starts_ok is False
        assert 'moments' not in starts.families()

    def test_from_spec_like_object(self):
        class Spec:
            n_map_starts = 6
            map_moment_starts = True
            map_bounded = False
            map_polish_steps = 0
            map_polish_basins = 1
            eig_floor_mode = 'relative'
            eig_floor = 1e-3
            chain_init = 'map_basins'
            chain_init_max_margin = 5.0
            hessian_method = 'fd'

        cfg = InitConfig.from_spec(Spec(), n_pa_starts=2)
        assert cfg.n_map_starts == 6 and cfg.n_pa_starts == 2
        assert cfg.map_bounded is False and cfg.map_polish_steps == 0
        assert cfg.floor == EigenFloor('relative', 1e-3)
        assert cfg.chain_init == 'map_basins' and cfg.chain_init_max_margin == 5.0
