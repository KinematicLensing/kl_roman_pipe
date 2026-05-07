"""
Unit tests for CompositeIntensityModel and BulgeDiskModel.

Tests cover:
- Parameter naming, structure, and roundtrips
- Component theta extraction and flux derivation
- K-space rendering (composite vs sum of individuals)
- Fixed-overrides-shared parameter handling
- JIT compilation and gradient flow
- Integration with KLModel
"""

import pytest
import galsim as gs
import jax
import jax.numpy as jnp
import numpy as np

from kl_pipe.intensity import (
    BulgeDiskModel,
    CompositeIntensityModel,
    ComponentSpec,
    InclinedExponentialModel,
    InclinedSersicModel,
    InclinedSpergelModel,
    build_intensity_model,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.noise import add_intensity_noise
from kl_pipe.utils import get_test_dir


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def output_dir():
    out_dir = get_test_dir() / 'out' / 'composite'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def bulge_disk():
    return BulgeDiskModel()


@pytest.fixture
def bulge_disk_shared():
    return BulgeDiskModel(shared_centroids=True)


@pytest.fixture
def bulge_disk_pars():
    """Parameter dict for BulgeDiskModel with independent centroids."""
    return {
        'cosi': 0.5,
        'theta_int': 0.3,
        'g1': 0.02,
        'g2': -0.01,
        'total_flux': 1e4,
        'bulge_frac': 0.25,
        'disk_rscale': 1.0,
        'disk_h_over_r': 0.1,
        'disk_x0': 0.0,
        'disk_y0': 0.0,
        'bulge_hlr': 0.3,
        'bulge_h_over_hlr': 0.3,
        'bulge_x0': 0.0,
        'bulge_y0': 0.0,
    }


@pytest.fixture
def bulge_disk_shared_pars():
    """Parameter dict for BulgeDiskModel with shared centroids."""
    return {
        'cosi': 0.5,
        'theta_int': 0.3,
        'g1': 0.02,
        'g2': -0.01,
        'int_x0': 0.0,
        'int_y0': 0.0,
        'total_flux': 1e4,
        'bulge_frac': 0.25,
        'disk_rscale': 1.0,
        'disk_h_over_r': 0.1,
        'bulge_hlr': 0.3,
        'bulge_h_over_hlr': 0.3,
    }


@pytest.fixture
def image_pars():
    return ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='xy')


# ==============================================================================
# Parameter structure tests
# ==============================================================================


class TestBulgeDiskParameterNames:
    """Verify PARAMETER_NAMES for BulgeDiskModel configurations."""

    def test_independent_centroids(self, bulge_disk):
        expected = (
            'cosi',
            'theta_int',
            'g1',
            'g2',
            'total_flux',
            'bulge_frac',
            'disk_rscale',
            'disk_h_over_r',
            'disk_x0',
            'disk_y0',
            'bulge_hlr',
            'bulge_h_over_hlr',
            'bulge_x0',
            'bulge_y0',
        )
        assert bulge_disk.PARAMETER_NAMES == expected

    def test_shared_centroids(self, bulge_disk_shared):
        pnames = bulge_disk_shared.PARAMETER_NAMES
        assert len(pnames) == 12
        assert 'int_x0' in pnames
        assert 'int_y0' in pnames
        assert 'disk_x0' not in pnames
        assert 'bulge_x0' not in pnames

    def test_n_params_independent(self, bulge_disk):
        assert len(bulge_disk.PARAMETER_NAMES) == 14

    def test_n_params_shared(self, bulge_disk_shared):
        assert len(bulge_disk_shared.PARAMETER_NAMES) == 12

    def test_no_n_sersic(self, bulge_disk):
        """n_sersic is fixed at 4.0, should not appear."""
        assert 'n_sersic' not in bulge_disk.PARAMETER_NAMES
        assert 'bulge_n_sersic' not in bulge_disk.PARAMETER_NAMES

    def test_no_flux(self, bulge_disk):
        """Individual fluxes replaced by total_flux + bulge_frac."""
        assert 'flux' not in bulge_disk.PARAMETER_NAMES
        assert 'disk_flux' not in bulge_disk.PARAMETER_NAMES
        assert 'bulge_flux' not in bulge_disk.PARAMETER_NAMES
        assert 'total_flux' in bulge_disk.PARAMETER_NAMES
        assert 'bulge_frac' in bulge_disk.PARAMETER_NAMES


class TestThetaParsRoundtrip:
    """Verify theta <-> pars conversion."""

    def test_roundtrip_independent(self, bulge_disk, bulge_disk_pars):
        theta = bulge_disk.pars2theta(bulge_disk_pars)
        pars_rt = bulge_disk.theta2pars(theta)
        theta_rt = bulge_disk.pars2theta(pars_rt)
        assert jnp.allclose(theta, theta_rt)

    def test_roundtrip_shared(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        pars_rt = bulge_disk_shared.theta2pars(theta)
        theta_rt = bulge_disk_shared.pars2theta(pars_rt)
        assert jnp.allclose(theta, theta_rt)

    def test_pars_values(self, bulge_disk, bulge_disk_pars):
        theta = bulge_disk.pars2theta(bulge_disk_pars)
        pars_rt = bulge_disk.theta2pars(theta)
        for key, val in bulge_disk_pars.items():
            assert abs(pars_rt[key] - val) < 1e-6, f'{key}: {pars_rt[key]} != {val}'


# ==============================================================================
# Component theta extraction
# ==============================================================================


class TestComponentThetaExtraction:
    """Verify _get_component_theta produces correct values."""

    def test_flux_derivation(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        ct0 = bulge_disk_shared._get_component_theta(theta, 0)
        ct1 = bulge_disk_shared._get_component_theta(theta, 1)

        total = bulge_disk_shared_pars['total_flux']
        bf = bulge_disk_shared_pars['bulge_frac']
        disk_flux = float(
            ct0[bulge_disk_shared._components[0].model._param_indices['flux']]
        )
        bulge_flux = float(
            ct1[bulge_disk_shared._components[1].model._param_indices['flux']]
        )

        assert abs(disk_flux - total * (1 - bf)) < 1e-3
        assert abs(bulge_flux - total * bf) < 1e-3
        assert abs(disk_flux + bulge_flux - total) < 1e-3

    def test_shared_geometry(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        ct0 = bulge_disk_shared._get_component_theta(theta, 0)
        ct1 = bulge_disk_shared._get_component_theta(theta, 1)

        disk_model = bulge_disk_shared._components[0].model
        bulge_model = bulge_disk_shared._components[1].model

        for p in ('cosi', 'theta_int', 'g1', 'g2'):
            v0 = float(ct0[disk_model._param_indices[p]])
            v1 = float(ct1[bulge_model._param_indices[p]])
            assert abs(v0 - v1) < 1e-6, f'{p}: disk={v0} != bulge={v1}'
            assert abs(v0 - bulge_disk_shared_pars[p]) < 1e-6

    def test_fixed_n_sersic(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        ct1 = bulge_disk_shared._get_component_theta(theta, 1)
        bulge_model = bulge_disk_shared._components[1].model
        n = float(ct1[bulge_model._param_indices['n_sersic']])
        assert abs(n - 4.0) < 1e-6


# ==============================================================================
# Rendering tests
# ==============================================================================


class TestCompositeRendering:
    """Verify k-space rendering correctness."""

    def test_render_shape(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        img = bulge_disk_shared._render_kspace(theta, 32, 32, 0.11)
        assert img.shape == (32, 32)

    def test_composite_vs_sum_of_individuals(
        self, bulge_disk_shared, bulge_disk_shared_pars
    ):
        """Composite render should match sum of individual renders."""
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        composite_img = bulge_disk_shared._render_kspace(theta, 64, 64, 0.11)

        total_flux = bulge_disk_shared_pars['total_flux']
        bf = bulge_disk_shared_pars['bulge_frac']

        disk = InclinedExponentialModel()
        disk_pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.02,
            'g2': -0.01,
            'flux': total_flux * (1 - bf),
            'int_rscale': 1.0,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        disk_img = disk._render_kspace(disk.pars2theta(disk_pars), 64, 64, 0.11)

        bulge = InclinedSersicModel()
        bulge_pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.02,
            'g2': -0.01,
            'flux': total_flux * bf,
            'int_hlr': 0.3,
            'int_h_over_hlr': 0.3,
            'n_sersic': 4.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        bulge_img = bulge._render_kspace(bulge.pars2theta(bulge_pars), 64, 64, 0.11)

        sum_img = disk_img + bulge_img
        max_val = float(jnp.abs(sum_img).max())
        max_diff = float(jnp.abs(composite_img - sum_img).max())
        rel_err = max_diff / max_val

        assert rel_err < 1e-4, f'relative error {rel_err:.2e} > 1e-4'

    def test_bulge_frac_zero(self, bulge_disk_shared):
        """bulge_frac=0 should match pure disk."""
        pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.0,
            'g2': 0.0,
            'total_flux': 1e4,
            'bulge_frac': 0.0,
            'disk_rscale': 1.0,
            'disk_h_over_r': 0.1,
            'bulge_hlr': 0.3,
            'bulge_h_over_hlr': 0.3,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        theta = bulge_disk_shared.pars2theta(pars)
        composite_img = bulge_disk_shared._render_kspace(theta, 64, 64, 0.11)

        disk = InclinedExponentialModel()
        disk_pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.0,
            'g2': 0.0,
            'flux': 1e4,
            'int_rscale': 1.0,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        disk_img = disk._render_kspace(disk.pars2theta(disk_pars), 64, 64, 0.11)

        max_val = float(jnp.abs(disk_img).max())
        max_diff = float(jnp.abs(composite_img - disk_img).max())
        rel_err = max_diff / max_val

        assert rel_err < 1e-4, f'relative error {rel_err:.2e} > 1e-4'

    def test_bulge_frac_one(self, bulge_disk_shared):
        """bulge_frac=1 should match pure bulge."""
        pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.0,
            'g2': 0.0,
            'total_flux': 1e4,
            'bulge_frac': 1.0,
            'disk_rscale': 1.0,
            'disk_h_over_r': 0.1,
            'bulge_hlr': 0.3,
            'bulge_h_over_hlr': 0.3,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        theta = bulge_disk_shared.pars2theta(pars)
        composite_img = bulge_disk_shared._render_kspace(theta, 64, 64, 0.11)

        bulge = InclinedSersicModel()
        bulge_pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.0,
            'g2': 0.0,
            'flux': 1e4,
            'int_hlr': 0.3,
            'int_h_over_hlr': 0.3,
            'n_sersic': 4.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        bulge_img = bulge._render_kspace(bulge.pars2theta(bulge_pars), 64, 64, 0.11)

        max_val = float(jnp.abs(bulge_img).max())
        max_diff = float(jnp.abs(composite_img - bulge_img).max())
        rel_err = max_diff / max_val

        assert rel_err < 1e-4, f'relative error {rel_err:.2e} > 1e-4'


# ==============================================================================
# Fixed-overrides-shared tests
# ==============================================================================


class TestFixedOverridesShared:
    """Verify fixed_params overrides shared_pars for specific component."""

    def test_zero_bulge_shear(self):
        model = CompositeIntensityModel(
            components=[
                ComponentSpec(InclinedExponentialModel(), prefix='disk'),
                ComponentSpec(
                    InclinedSersicModel(),
                    prefix='bulge',
                    fixed_params={'n_sersic': 4.0, 'g1': 0.0, 'g2': 0.0},
                ),
            ],
        )
        # g1, g2 should still be in PARAMETER_NAMES (disk needs them)
        assert 'g1' in model.PARAMETER_NAMES
        assert 'g2' in model.PARAMETER_NAMES

        pars = {
            'cosi': 0.5,
            'theta_int': 0.3,
            'g1': 0.05,
            'g2': -0.03,
            'total_flux': 1e4,
            'bulge_frac': 0.25,
            'disk_rscale': 1.0,
            'disk_h_over_r': 0.1,
            'disk_x0': 0.0,
            'disk_y0': 0.0,
            'bulge_hlr': 0.3,
            'bulge_h_over_hlr': 0.3,
            'bulge_x0': 0.0,
            'bulge_y0': 0.0,
        }
        theta = model.pars2theta(pars)

        ct0 = model._get_component_theta(theta, 0)
        ct1 = model._get_component_theta(theta, 1)

        disk_model = model._components[0].model
        bulge_model = model._components[1].model

        # disk gets sampled shear
        assert abs(float(ct0[disk_model._param_indices['g1']]) - 0.05) < 1e-6
        assert abs(float(ct0[disk_model._param_indices['g2']]) + 0.03) < 1e-6

        # bulge gets fixed zeros
        assert abs(float(ct1[bulge_model._param_indices['g1']])) < 1e-6
        assert abs(float(ct1[bulge_model._param_indices['g2']])) < 1e-6

    def test_shared_param_still_in_parameter_names(self):
        """A shared param fixed for ONE component must stay in PARAMETER_NAMES."""
        model = CompositeIntensityModel(
            components=[
                ComponentSpec(InclinedExponentialModel(), prefix='a'),
                ComponentSpec(
                    InclinedSersicModel(),
                    prefix='b',
                    fixed_params={'n_sersic': 4.0, 'g1': 0.0},
                ),
            ],
        )
        # g1 is shared, fixed for b, free for a -> must be in PARAMETER_NAMES
        assert 'g1' in model.PARAMETER_NAMES


# ==============================================================================
# JIT and gradient tests
# ==============================================================================


class TestJITAndGrad:
    """Verify JIT compilation and gradient flow."""

    def test_jit_render(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        fn = jax.jit(lambda t: bulge_disk_shared._render_kspace(t, 32, 32, 0.11))
        img = fn(theta)
        assert img.shape == (32, 32)
        assert not jnp.any(jnp.isnan(img))

    def test_grad_no_nan(self, bulge_disk_shared, bulge_disk_shared_pars):
        theta = bulge_disk_shared.pars2theta(bulge_disk_shared_pars)
        grad_fn = jax.jit(
            jax.grad(lambda t: bulge_disk_shared._render_kspace(t, 32, 32, 0.11).sum())
        )
        g = grad_fn(theta)
        assert g.shape == theta.shape
        assert not jnp.any(jnp.isnan(g)), 'gradient contains NaN'


# ==============================================================================
# Generic composite tests
# ==============================================================================


class TestGenericComposite:
    """Tests for arbitrary CompositeIntensityModel configurations."""

    def test_three_component(self):
        model = CompositeIntensityModel(
            components=[
                ComponentSpec(InclinedExponentialModel(), prefix='disk'),
                ComponentSpec(
                    InclinedSersicModel(),
                    prefix='bulge',
                    fixed_params={'n_sersic': 4.0},
                ),
                ComponentSpec(
                    InclinedSersicModel(),
                    prefix='bar',
                    fixed_params={'n_sersic': 1.5},
                ),
            ],
        )
        assert 'bulge_frac' in model.PARAMETER_NAMES
        assert 'bar_frac' in model.PARAMETER_NAMES
        assert 'total_flux' in model.PARAMETER_NAMES
        # 3 fracs: disk derived, bulge_frac and bar_frac explicit
        assert len([p for p in model.PARAMETER_NAMES if p.endswith('_frac')]) == 2

    def test_two_sersic_free_n(self):
        model = CompositeIntensityModel(
            components=[
                ComponentSpec(InclinedSersicModel(), prefix='thin'),
                ComponentSpec(InclinedSersicModel(), prefix='thick'),
            ],
        )
        assert 'thin_n_sersic' in model.PARAMETER_NAMES
        assert 'thick_n_sersic' in model.PARAMETER_NAMES

    def test_custom_shared_pars_centroids(self):
        model = CompositeIntensityModel(
            components=[
                ComponentSpec(InclinedExponentialModel(), prefix='a'),
                ComponentSpec(
                    InclinedSersicModel(),
                    prefix='b',
                    fixed_params={'n_sersic': 2.0},
                ),
            ],
            shared_pars={'cosi', 'theta_int', 'g1', 'g2', 'int_x0', 'int_y0'},
        )
        assert 'int_x0' in model.PARAMETER_NAMES
        assert 'a_x0' not in model.PARAMETER_NAMES
        assert 'b_x0' not in model.PARAMETER_NAMES

    def test_minimum_two_components(self):
        with pytest.raises(ValueError, match='requires >= 2'):
            CompositeIntensityModel(
                components=[ComponentSpec(InclinedExponentialModel(), prefix='solo')],
            )

    def test_invalid_fixed_param(self):
        with pytest.raises(ValueError, match='not in'):
            CompositeIntensityModel(
                components=[
                    ComponentSpec(InclinedExponentialModel(), prefix='a'),
                    ComponentSpec(
                        InclinedExponentialModel(),
                        prefix='b',
                        fixed_params={'nonexistent': 1.0},
                    ),
                ],
            )

    def test_invalid_shared_param(self):
        with pytest.raises(ValueError, match='not found in any component'):
            CompositeIntensityModel(
                components=[
                    ComponentSpec(InclinedExponentialModel(), prefix='a'),
                    ComponentSpec(InclinedExponentialModel(), prefix='b'),
                ],
                shared_pars={'cosi', 'theta_int', 'nonexistent'},
            )


# ==============================================================================
# Integration tests
# ==============================================================================


class TestKLModelIntegration:
    """Verify composite works inside KLModel."""

    def test_klmodel_parameter_dedup(self):
        from kl_pipe.model import KLModel
        from kl_pipe.velocity import OffsetVelocityModel

        vel = OffsetVelocityModel()
        intensity = BulgeDiskModel(shared_centroids=True)
        kl = KLModel(vel, intensity, shared_pars={'cosi', 'theta_int', 'g1', 'g2'})

        # shared geometric params should appear once
        pnames = kl.PARAMETER_NAMES
        assert pnames.count('cosi') == 1
        assert pnames.count('g1') == 1
        assert 'total_flux' in pnames
        assert 'bulge_frac' in pnames
        assert 'vcirc' in pnames


# ==============================================================================
# Factory tests
# ==============================================================================


class TestFactory:
    """Verify factory registration."""

    def test_build_bulge_disk(self):
        model = build_intensity_model('bulge_disk')
        assert isinstance(model, BulgeDiskModel)

    def test_name_property(self):
        model = BulgeDiskModel()
        assert model.name == 'bulge_disk'

    def test_composite_name(self):
        model = CompositeIntensityModel(
            components=[
                ComponentSpec(InclinedExponentialModel(), prefix='disk'),
                ComponentSpec(
                    InclinedSersicModel(),
                    prefix='bulge',
                    fixed_params={'n_sersic': 4.0},
                ),
            ],
        )
        assert 'disk:inclined_exp' in model.name
        assert 'bulge:inclined_sersic' in model.name


# ==============================================================================
# Helpers for advanced tests
# ==============================================================================


# shared true pars for likelihood/optimizer tests (shared centroids, 12 params)
_TRUE_PARS_SHARED = {
    'cosi': 0.7,
    'theta_int': 0.785,
    'g1': 0.0,
    'g2': 0.0,
    'int_x0': 0.0,
    'int_y0': 0.0,
    'total_flux': 1.0,
    'bulge_frac': 0.25,
    'disk_rscale': 2.0,
    'disk_h_over_r': 0.1,
    'bulge_hlr': 0.8,
    'bulge_h_over_hlr': 0.3,
}

# image pars: finer pixels to resolve bulge
_IMAGE_PARS = ImagePars(shape=(64, 64), pixel_scale=0.15, indexing='xy')

# PSF for composite recovery tests. Gaussian FWHM=0.15 arcsec ≈ pixel scale
# and ≈ Roman-band diffraction limit. With bulge half-light radius 0.8 arcsec
# the bulge stays well-resolved (FWHM_psf << hlr_bulge); meanwhile the n=4
# cusp is smoothed enough that the Miller-Pasha emulator core-pixel error
# (3-11% per emulator note) gets convolved well below the per-pixel noise
# floor at the SNRs we test. Recovery tests omit the PSF prior to PR #41
# only when explicitly checking sub-pixel response in isolation.
_TEST_PSF = gs.Gaussian(fwhm=0.15)


def _generate_galsim_composite(pars, image_pars, psf=None):
    """Generate composite disk+bulge image via GalSim (independent of JAX model).

    Uses galsim.Add(InclinedExponential, InclinedSersic) as ground truth.

    When ``psf`` is None, draws with ``method='no_pixel'`` to expose the
    bare profile (used by the no-PSF regression test). When ``psf`` is
    provided, uses GalSim's default rendering path so the pixel-response
    top-hat is included in addition to the PSF convolution.
    """
    cosi = pars['cosi']
    inclination = np.arccos(cosi) * gs.radians
    sini = np.sqrt(1.0 - cosi**2)
    total_flux = pars['total_flux']
    bf = pars['bulge_frac']

    # disk: n=1 exponential
    disk = gs.InclinedExponential(
        inclination=inclination,
        scale_radius=pars['disk_rscale'],
        scale_h_over_r=pars['disk_h_over_r'],
        flux=total_flux * (1 - bf),
    )

    # bulge: n=4 sersic
    # NOTE: GalSim reinterprets scale_h_over_r as h_z/scale_radius (NOT
    # h_z/half_light_radius) when half_light_radius is supplied — see
    # galsim.InclinedSersic docstring. Pass scale_height (physical h_z in
    # arcsec) directly to bypass that reinterpretation. kl_pipe's
    # bulge_h_over_hlr is h_z/half_light_radius, so physical h_z =
    # bulge_h_over_hlr * bulge_hlr.
    bulge = gs.InclinedSersic(
        n=4.0,
        inclination=inclination,
        half_light_radius=pars['bulge_hlr'],
        scale_height=pars['bulge_h_over_hlr'] * pars['bulge_hlr'],
        flux=total_flux * bf,
    )

    composite = gs.Add(disk, bulge)

    # apply transforms: rotate, shear, shift
    theta_int = pars.get('theta_int', 0.0)
    g1 = pars.get('g1', 0.0)
    g2 = pars.get('g2', 0.0)
    x0 = pars.get('int_x0', pars.get('disk_x0', 0.0))
    y0 = pars.get('int_y0', pars.get('disk_y0', 0.0))

    composite = composite.rotate(theta_int * gs.radians)
    if abs(g1) > 0 or abs(g2) > 0:
        mu = 1.0 / ((1 - g1**2 - g2**2))
        composite = composite.lens(g1=g1, g2=g2, mu=mu)
    if abs(x0) > 0 or abs(y0) > 0:
        composite = composite.shift(x0, y0)

    if psf is not None:
        composite = gs.Convolve(composite, psf)

    # draw — when a PSF is convolved in, use GalSim's default method (auto)
    # so the pixel-response top-hat is included; otherwise stay on
    # 'no_pixel' to keep the no-PSF regression test sampling the bare profile.
    Nx = image_pars.Nx
    Ny = image_pars.Ny
    gs_image = gs.Image(Nx, Ny, scale=image_pars.pixel_scale)
    if psf is None:
        composite.drawImage(image=gs_image, method='no_pixel')
    else:
        composite.drawImage(image=gs_image)  # default method (auto)
    return gs_image.array


# ==============================================================================
# GalSim comparison test
# ==============================================================================


class TestGalSimComparison:
    """Compare composite rendering against GalSim ground truth."""

    def test_composite_vs_galsim_no_psf(self, output_dir):
        """Render vs galsim.Add(InclinedExponential, InclinedSersic)."""
        pars = dict(_TRUE_PARS_SHARED)
        model = BulgeDiskModel(shared_centroids=True)
        theta = model.pars2theta(pars)

        kl_img = np.array(
            model._render_kspace(
                theta, _IMAGE_PARS.Nrow, _IMAGE_PARS.Ncol, _IMAGE_PARS.pixel_scale
            )
        )

        gs_img = _generate_galsim_composite(pars, _IMAGE_PARS)

        residual = kl_img - gs_img
        rms = np.sqrt(np.mean(residual**2))
        max_val = np.abs(gs_img).max()
        rms_frac = rms / max_val

        # diagnostic plot
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        vmin, vmax = gs_img.min(), gs_img.max()
        axes[0].imshow(gs_img, origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('GalSim Add(disk,bulge)')
        axes[1].imshow(kl_img, origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('kl_pipe BulgeDiskModel')
        im = axes[2].imshow(residual, origin='lower', cmap='RdBu_r')
        axes[2].set_title(f'Residual (RMS frac={rms_frac:.4f})')
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        plt.savefig(output_dir / 'composite_vs_galsim.png', dpi=100)
        plt.close()

        assert rms_frac < 0.02, f'RMS fractional error {rms_frac:.4f} > 2%'


# ==============================================================================
# Synthetic data helper for slice / optimizer recovery tests
#
# The recovery tests themselves live in test_likelihood_slices.py and
# test_optimizer_recovery.py — they import this helper plus the constants
# (_TRUE_PARS_SHARED, _IMAGE_PARS, _TEST_PSF) and renderer
# (_generate_galsim_composite) defined above.
# ==============================================================================


def _generate_composite_synthetic(pars, image_pars, snr, seed=42, psf=None):
    """Generate synthetic composite data via GalSim (independent of JAX model).

    Uses galsim.Add(InclinedExponential, InclinedSersic) as ground truth.
    When ``psf`` is provided, the GalSim render also applies the pixel-response
    top-hat (default ``drawImage`` method) — the recommended setup for
    composite recovery tests, since a bare-cusp render exposes the n=4
    emulator's core-pixel approximation directly to the per-pixel data.

    Noise convention: matched-filter SNR (sigma = ||I||_2 / target_snr),
    matching the other intensity slice tests via
    ``add_intensity_noise(include_poisson=False)``.
    """
    data_true = _generate_galsim_composite(pars, image_pars, psf=psf)
    data_noisy, variance = add_intensity_noise(
        data_true, target_snr=snr, include_poisson=False, seed=seed
    )
    return data_true, data_noisy, variance


# BulgeDisk recovery tests live in:
#   tests/test_likelihood_slices.py::test_recover_bulge_disk
#   tests/test_optimizer_recovery.py::test_optimize_bulge_disk
# They import _TRUE_PARS_SHARED, _IMAGE_PARS, _TEST_PSF, and
# _generate_composite_synthetic from this file.
