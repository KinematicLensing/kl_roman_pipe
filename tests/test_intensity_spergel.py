"""
Unit tests for Spergel and DeVaucouleurs intensity models.

Tests include:
- Model instantiation, parameter conversion, factory
- nu=0.5 cross-checks against InclinedExponentialModel
- GalSim regression (face-on, inclined, with PSF)
- Flux conservation, gradient through nu, normalization integral
- KLModel composition, PSF path consistency
- Spergel vs Sersic mismatch diagnostics
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from kl_pipe.intensity import (
    InclinedExponentialModel,
    InclinedSpergelModel,
    InclinedDeVaucouleursModel,
    build_intensity_model,
)
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir, build_map_grid_from_image_pars


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def output_dir():
    """Output directory for backward-compatible test plots."""
    out_dir = get_test_dir() / "out" / "intensity"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def spergel_output_dir():
    """Dedicated output directory for Spergel diagnostic plots."""
    out_dir = get_test_dir() / "out" / "spergel"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def basic_meta_pars():
    """Basic metadata for model instantiation."""
    return {'test_param': 'test_value'}


@pytest.fixture(scope='module')
def galsim_image_pars():
    """ImagePars for GalSim regression tests.

    128x128 grid ensures profile is < 0.01% of peak at boundary,
    eliminating FFT aliasing artifacts in k-space rendering.
    """
    return ImagePars(shape=(128, 128), pixel_scale=0.2, indexing='ij')


@pytest.fixture
def spergel_theta():
    """Standard theta for InclinedSpergelModel (10 params)."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, nu, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.1, 0.5, 0.0, 0.0])


@pytest.fixture
def devaucouleurs_theta():
    """Standard theta for InclinedDeVaucouleursModel (9 params)."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.1, 0.0, 0.0])


# --- Unit Tests (Step 4) ---


def test_spergel_model_instantiation(basic_meta_pars):
    """Test InclinedSpergelModel can be instantiated."""
    model = InclinedSpergelModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'inclined_spergel'
    assert len(model.PARAMETER_NAMES) == 10


def test_spergel_parameter_names():
    """Test that nu is at index 7 and all expected params exist."""
    model = InclinedSpergelModel()
    assert model.PARAMETER_NAMES[7] == 'nu'
    for name in ('flux', 'int_rscale', 'cosi', 'theta_int', 'nu', 'int_x0', 'int_y0'):
        assert name in model.PARAMETER_NAMES


def test_spergel_theta_roundtrip():
    """Test theta -> pars -> theta roundtrip for Spergel model."""
    pars = {
        'cosi': 0.7,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
        'flux': 1.0,
        'int_rscale': 3.0,
        'int_h_over_r': 0.1,
        'nu': 0.5,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    theta = InclinedSpergelModel.pars2theta(pars)
    pars_back = InclinedSpergelModel.theta2pars(theta)
    theta_back = InclinedSpergelModel.pars2theta(pars_back)
    assert jnp.allclose(theta, theta_back)
    assert len(theta) == 10


def test_devaucouleurs_model_instantiation(basic_meta_pars):
    """Test InclinedDeVaucouleursModel: 9 params, no nu."""
    model = InclinedDeVaucouleursModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'de_vaucouleurs'
    assert len(model.PARAMETER_NAMES) == 9
    assert 'nu' not in model.PARAMETER_NAMES


def test_devaucouleurs_matches_spergel_fixed_nu(galsim_image_pars):
    """DeVaucouleurs render_image must be bit-identical to Spergel(nu=-0.6)."""
    spergel = InclinedSpergelModel()
    devac = InclinedDeVaucouleursModel()

    cosi, theta_int, g1, g2 = 0.7, 0.3, 0.02, -0.01
    flux, rscale, h_over_r, nu = 1.0, 2.0, 0.1, -0.6

    theta_sp = jnp.array(
        [cosi, theta_int, g1, g2, flux, rscale, h_over_r, nu, 0.0, 0.0]
    )
    theta_dv = jnp.array([cosi, theta_int, g1, g2, flux, rscale, h_over_r, 0.0, 0.0])

    img_sp = spergel.render_image(theta_sp, image_pars=galsim_image_pars)
    img_dv = devac.render_image(theta_dv, image_pars=galsim_image_pars)

    max_diff = float(jnp.max(jnp.abs(img_sp - img_dv)))
    assert max_diff < 1e-10, f"DeVauc vs Spergel(nu=-0.6): max diff = {max_diff:.2e}"


@pytest.mark.parametrize(
    "nu,cosi,g1,g2",
    [
        (0.5, 1.0, 0.0, 0.0),
        (-0.6, 1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (0.5, 0.7, 0.0, 0.0),
        (-0.6, 0.7, 0.05, -0.03),
        (2.0, 0.3, 0.03, -0.02),
    ],
)
def test_spergel_flux_conservation(nu, cosi, g1, g2, galsim_image_pars):
    """sum(image) * pixel_scale² must match input flux within 0.1%.

    Uses rscale=1.0 so FOV/rscale=25.6 on the 128x128 grid, keeping
    profile tails well within the grid boundary.
    """
    model = InclinedSpergelModel()
    flux = 1.0
    theta = jnp.array([cosi, 0.3, g1, g2, flux, 1.0, 0.1, nu, 0.0, 0.0])
    image = model.render_image(theta, image_pars=galsim_image_pars)
    ps = galsim_image_pars.pixel_scale
    measured_flux = float(jnp.sum(image) * ps**2)
    rel_err = abs(measured_flux - flux) / flux
    assert rel_err < 1e-3, (
        f"Flux: measured={measured_flux:.6f}, expected={flux}, "
        f"rel_err={rel_err:.2e} (nu={nu}, cosi={cosi})"
    )


@pytest.mark.parametrize(
    "nu,cosi,int_h_over_r,tol",
    [
        (0.5, 0.7, 0.1, 5e-3),
        (2.0, 0.7, 0.1, 5e-3),
        (0.5, 1.0, 0.1, 4e-3),
    ],
)
def test_spergel_render_image_vs_call_consistency(
    nu, cosi, int_h_over_r, tol, galsim_image_pars
):
    """K-space render_image vs real-space __call__ consistency.

    Uses nonzero centroid offset to avoid worst-case grid alignment where
    the cusp sits exactly between pixel centers on even-sized grids.
    Matches the existing exponential render_vs_call test convention.
    """
    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, np.pi / 6, 0.02, -0.01, 1.0, 2.0, int_h_over_r, nu, 0.3, -0.2]
    )

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    call_image = np.array(model(theta, 'obs', X, Y))
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(call_image))
    max_frac = np.max(np.abs(render - call_image)) / peak

    assert max_frac < tol, (
        f"render vs call: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, cosi={cosi}, h/r={int_h_over_r}, tol={tol})"
    )


def test_spergel_render_vs_call_negative_nu_diagnostic(galsim_image_pars, output_dir):
    """Diagnostic: render_image vs __call__ for nu=-0.6 (no assertion).

    For nu < 0 the Spergel volume density diverges as R^{2nu} at R=0.
    LOSes near the center have integrand ~ |ell|^{2nu} with 2nu < -1,
    making the 1D LOS integral divergent. The k-space render_image
    band-limits the profile (finite everywhere), while __call__ point-
    samples it (divergent at center). They disagree near the center by
    construction; this test records the disagreement quantitatively.
    """
    model = InclinedSpergelModel()
    nu = -0.6
    theta = jnp.array([0.7, np.pi / 6, 0.02, -0.01, 1.0, 2.0, 0.1, nu, 0.0, 0.0])

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    call_image = np.array(model(theta, 'obs', X, Y))
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(call_image))
    residual = np.abs(render - call_image)
    max_frac = np.max(residual) / peak
    p95 = np.percentile(residual / peak, 95)
    p99 = np.percentile(residual / peak, 99)

    print(
        f"\nSpergel nu={nu} render_vs_call diagnostic:\n"
        f"  max|resid|/peak = {max_frac:.4f}\n"
        f"  p95 = {p95:.4f}, p99 = {p99:.4f}\n"
        f"  (error localized to central cusp; 95% of pixels agree within {p95:.1%})"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axes[0, 0].imshow(call_image, origin='lower')
    axes[0, 0].set_title('__call__ (LOS quadrature)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(render, origin='lower')
    axes[0, 1].set_title('render_image (k-space FFT)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(residual, origin='lower', cmap='hot')
    axes[1, 0].set_title('|residual|')
    plt.colorbar(im2, ax=axes[1, 0])

    rel = residual / peak
    im3 = axes[1, 1].imshow(rel, origin='lower', cmap='hot')
    axes[1, 1].set_title(f'|residual|/peak (max={max_frac:.2e})')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(
        f'Spergel nu={nu} render vs call — DIAGNOSTIC (no assertion)\n'
        f'max={max_frac:.2e}, p95={p95:.2e}, p99={p99:.2e}',
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'spergel_nu{nu}_render_vs_call_diagnostic.png', dpi=150)
    plt.close()


def test_spergel_non_square_grid():
    """Output shape = (Nrow, Ncol) on non-square grid."""
    model = InclinedSpergelModel()
    ip = ImagePars(shape=(80, 120), pixel_scale=0.2, indexing='ij')
    theta = jnp.array([0.7, 0.3, 0.0, 0.0, 1.0, 2.0, 0.1, 0.5, 0.0, 0.0])
    image = model.render_image(theta, image_pars=ip)
    assert image.shape == (80, 120), f"Expected (80, 120), got {image.shape}"


def test_factory_spergel():
    """build_intensity_model('spergel') returns InclinedSpergelModel."""
    model = build_intensity_model('spergel')
    assert isinstance(model, InclinedSpergelModel)
    model2 = build_intensity_model('inclined_spergel')
    assert isinstance(model2, InclinedSpergelModel)


def test_factory_de_vaucouleurs():
    """build_intensity_model('de_vaucouleurs') returns InclinedDeVaucouleursModel."""
    model = build_intensity_model('de_vaucouleurs')
    assert isinstance(model, InclinedDeVaucouleursModel)


def test_spergel_gradient_through_nu():
    """jax.grad of k-space render w.r.t. nu must match finite differences."""
    model = InclinedSpergelModel()
    ip = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
    theta = jnp.array([0.7, 0.3, 0.0, 0.0, 1.0, 2.0, 0.1, 0.5, 0.0, 0.0])

    def scalar_fn(th):
        return jnp.sum(model._render_kspace(th, ip.Nrow, ip.Ncol, ip.pixel_scale))

    grad_val = jax.grad(scalar_fn)(theta)
    nu_grad = float(grad_val[7])

    eps = 1e-5
    f_plus = scalar_fn(theta.at[7].set(theta[7] + eps))
    f_minus = scalar_fn(theta.at[7].set(theta[7] - eps))
    fd_grad = float((f_plus - f_minus) / (2 * eps))

    rel_err = abs(nu_grad - fd_grad) / max(abs(fd_grad), 1e-10)
    assert (
        rel_err < 1e-3
    ), f"nu gradient: jax={nu_grad:.6e}, fd={fd_grad:.6e}, rel_err={rel_err:.2e}"


@pytest.mark.parametrize("nu", [-0.6, 0.0, 0.5, 1.0, 2.0])
def test_spergel_normalization_integral(nu):
    """Numerical radial integral of face-on profile must equal flux."""
    from scipy.integrate import quad
    from scipy.special import kve as scipy_kve_np, gamma as scipy_gamma

    flux = 1.0
    rscale = 2.0
    norm_factor = 2.0 * np.pi * rscale**2 * (2.0**nu) * scipy_gamma(nu + 1.0)
    I_0 = flux / norm_factor

    def integrand(r):
        x = r / rscale
        if x < 1e-30:
            return 0.0
        profile = (x**nu) * scipy_kve_np(nu, x) * np.exp(-x)
        return 2.0 * np.pi * r * I_0 * profile

    measured_flux, _ = quad(integrand, 0, 30.0 * rscale, limit=200)
    rel_err = abs(measured_flux - flux) / flux
    assert rel_err < 5e-3, (
        f"Normalization: measured={measured_flux:.6f}, expected={flux}, "
        f"rel_err={rel_err:.2e} (nu={nu})"
    )


def test_spergel_psf_path_consistency(galsim_image_pars):
    """Fused k-space PSF path must match fallback real-space PSF path."""
    import galsim as gs

    model = InclinedSpergelModel()
    psf_obj = gs.Gaussian(fwhm=0.625)
    theta = jnp.array([0.7, 0.3, 0.02, -0.01, 1.0, 2.0, 0.1, 0.5, 0.0, 0.0])

    # fused k-space path (int_model triggers kspace_psf_fft)
    obs_kspace = build_image_obs(
        galsim_image_pars, psf=psf_obj, oversample=5, int_model=model
    )
    img_kspace = np.array(model.render_image(theta, obs=obs_kspace))

    # fallback real-space path (no int_model -> no kspace_psf_fft)
    obs_realspace = build_image_obs(galsim_image_pars, psf=psf_obj, oversample=5)
    img_realspace = np.array(model.render_image(theta, obs=obs_realspace))

    peak = np.max(np.abs(img_kspace))
    max_frac = np.max(np.abs(img_kspace - img_realspace)) / peak

    assert max_frac < 1e-3, f"PSF path consistency: max|resid|/peak = {max_frac:.2e}"


def test_spergel_klmodel_composition():
    """Spergel + CenteredVelocityModel in KLModel: param slicing correct."""
    from kl_pipe.model import KLModel
    from kl_pipe.velocity import CenteredVelocityModel

    vel = CenteredVelocityModel()
    int_model = InclinedSpergelModel()

    kl = KLModel(vel, int_model, shared_pars={'cosi', 'theta_int', 'g1', 'g2'})

    # vel(7) + int(10) - shared(4) = 13 composite params
    assert (
        len(kl.PARAMETER_NAMES) == 13
    ), f"Expected 13, got {len(kl.PARAMETER_NAMES)}: {kl.PARAMETER_NAMES}"
    assert kl.PARAMETER_NAMES.count('cosi') == 1
    assert kl.PARAMETER_NAMES.count('theta_int') == 1
    assert 'nu' in kl.PARAMETER_NAMES

    theta_kl = jnp.arange(13, dtype=float)
    theta_vel = kl.get_velocity_pars(theta_kl)
    theta_int = kl.get_intensity_pars(theta_kl)
    assert len(theta_vel) == 7
    assert len(theta_int) == 10


# --- nu=0.5 Cross-Check (Step 2) ---


@pytest.mark.parametrize(
    "cosi,theta_int,g1,g2",
    [
        (1.0, 0.0, 0.0, 0.0),
        (0.7, 0.0, 0.0, 0.0),
        (0.7, np.pi / 4, 0.0, 0.0),
        (0.7, 0.0, 0.05, -0.03),
        (0.3, np.pi / 4, 0.03, -0.02),
    ],
)
def test_spergel_nu05_matches_exponential_render_image(
    cosi, theta_int, g1, g2, galsim_image_pars
):
    """Spergel(nu=0.5) render_image must match Exponential within 1e-10.

    FTs are algebraically identical: (1+k²)^{-1.5} for both.
    """
    exp_model = InclinedExponentialModel()
    sp_model = InclinedSpergelModel()

    flux, rscale, h_over_r = 1.0, 2.0, 0.1
    theta_exp = jnp.array([cosi, theta_int, g1, g2, flux, rscale, h_over_r, 0.0, 0.0])
    theta_sp = jnp.array(
        [cosi, theta_int, g1, g2, flux, rscale, h_over_r, 0.5, 0.0, 0.0]
    )

    img_exp = exp_model.render_image(theta_exp, image_pars=galsim_image_pars)
    img_sp = sp_model.render_image(theta_sp, image_pars=galsim_image_pars)

    peak = float(jnp.max(jnp.abs(img_exp)))
    max_frac = float(jnp.max(jnp.abs(img_exp - img_sp)) / peak)

    assert max_frac < 1e-10, (
        f"Spergel(nu=0.5) vs Exp render_image: max|resid|/peak = {max_frac:.2e} "
        f"(cosi={cosi}, theta_int={theta_int:.2f}, g=({g1},{g2}))"
    )


def test_spergel_nu05_matches_exponential_call(galsim_image_pars):
    """Spergel(nu=0.5) __call__ must match Exponential within 1e-3.

    K_{0.5}(x) = sqrt(pi/2x)*exp(-x) is exact, but kve callback
    and GL quadrature introduce small numerical differences.
    """
    exp_model = InclinedExponentialModel()
    sp_model = InclinedSpergelModel()

    cosi, theta_int = 0.7, np.pi / 6
    flux, rscale, h_over_r = 1.0, 2.0, 0.1
    theta_exp = jnp.array([cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.0, 0.0])
    theta_sp = jnp.array(
        [cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.5, 0.0, 0.0]
    )

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    img_exp = np.array(exp_model(theta_exp, 'obs', X, Y))
    img_sp = np.array(sp_model(theta_sp, 'obs', X, Y))

    peak = np.max(np.abs(img_exp))
    max_frac = np.max(np.abs(img_exp - img_sp)) / peak

    assert (
        max_frac < 1e-3
    ), f"Spergel(nu=0.5) vs Exp __call__: max|resid|/peak = {max_frac:.2e}"


def test_spergel_nu05_matches_exponential_disk_plane(galsim_image_pars):
    """Spergel(nu=0.5) evaluate_in_disk_plane must match Exponential within 1e-4.

    K_{0.5}(x) = sqrt(pi/2x)*exp(-x) is exact.
    """
    exp_model = InclinedExponentialModel()
    sp_model = InclinedSpergelModel()

    flux, rscale = 1.0, 2.0
    theta_exp = jnp.array([0.7, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.0, 0.0])
    theta_sp = jnp.array([0.7, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.5, 0.0, 0.0])

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    disk_exp = np.array(exp_model.evaluate_in_disk_plane(theta_exp, X, Y))
    disk_sp = np.array(sp_model.evaluate_in_disk_plane(theta_sp, X, Y))

    peak = np.max(np.abs(disk_exp))
    max_frac = np.max(np.abs(disk_exp - disk_sp)) / peak

    assert (
        max_frac < 1e-4
    ), f"Spergel(nu=0.5) vs Exp disk plane: max|resid|/peak = {max_frac:.2e}"


def test_spergel_scipy_vs_render_image_consistency(galsim_image_pars):
    """Spergel scipy synthetic backend (numpy) vs render_image (JAX) consistency."""
    from kl_pipe.synthetic import _generate_spergel_scipy

    cosi = 0.7
    flux = 1.0
    int_rscale = 2.0
    int_h_over_r = 0.1
    nu = 0.5
    theta_int = np.pi / 6
    g1 = 0.02
    g2 = -0.01
    int_x0 = 0.3
    int_y0 = -0.2

    scipy_image = _generate_spergel_scipy(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        nu=nu,
        cosi=cosi,
        theta_int=theta_int,
        g1=g1,
        g2=g2,
        int_x0=int_x0,
        int_y0=int_y0,
        int_h_over_r=int_h_over_r,
    )

    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, nu, int_x0, int_y0]
    )
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(render))
    max_frac = np.max(np.abs(scipy_image - render)) / peak

    assert (
        max_frac < 2e-3
    ), f"Spergel scipy vs render_image: max|resid|/peak = {max_frac:.2e}"


# --- GalSim Regression Tests for Spergel (Step 3) ---


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0, 3.5])
def test_galsim_regression_spergel_faceon(nu, galsim_image_pars):
    """Face-on Spergel render_image vs galsim.Spergel (independent ground truth).

    Point-sampled comparison (no pixel response). Only nu >= 0.5 where
    the FT decays fast enough (k^{-(2+2nu)}) for aliasing to be negligible.
    See issue #38 for future migration to pixel-integrated comparisons.
    """
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, nu, 0.0, 0.0])
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    gs_profile = gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp)
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='no_pixel',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    # 5e-3: GalSim Spergel uses real-space evaluation (is_analytic_x=True)
    # while our code uses k-space IFFT. The difference is the band-limiting
    # error at the cusp. See issue #38 for pixel-integrated comparison.
    assert (
        max_frac < 5e-3
    ), f"Face-on Spergel vs GalSim: max|resid|/peak = {max_frac:.2e} (nu={nu})"


@pytest.mark.parametrize("nu", [-0.6, -0.3, 0.0])
def test_galsim_regression_spergel_faceon_cuspy(nu, galsim_image_pars):
    """Face-on Spergel vs GalSim for nu < 0.5 (requires PSF + pixel integration).

    For nu < 0.5 the Spergel FT decays slowly (k^{-(2+2nu)}), leaving
    significant power at Nyquist. Point-sampled comparison fails because
    our IFFT and GalSim's rendering handle the aliased high-k content
    differently. A small PSF suppresses high-k power; method='auto' +
    oversample=5 ensures both sides pixel-integrate for a fair comparison.
    """
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(folding_threshold=1e-5, maxk_threshold=1e-5, kvalue_accuracy=1e-7)
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, nu, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars, psf=psf, oversample=5, int_model=model, gsparams=gsp
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    gs_profile = gs.Convolve(
        gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp), psf
    )
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert (
        max_frac < 3e-3
    ), f"Face-on Spergel+PSF vs GalSim: max|resid|/peak = {max_frac:.2e} (nu={nu})"


@pytest.mark.parametrize(
    "nu,g1,g2",
    [
        (0.5, 0.05, -0.03),
        (2.0, 0.03, 0.04),
        (0.5, -0.04, 0.02),
    ],
)
def test_galsim_regression_spergel_faceon_with_shear(nu, g1, g2, galsim_image_pars):
    """Face-on sheared Spergel render_image vs galsim.Spergel."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, g1, g2, flux, rscale, 0.1, nu, 0.0, 0.0])
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    mu = 1.0 / (1.0 - (g1**2 + g2**2))
    gs_profile = gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp)
    gs_profile = gs_profile.lens(g1=g1, g2=g2, mu=mu)
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='no_pixel',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    # 5e-3: real-space (GalSim) vs k-space (ours) band-limiting difference
    assert max_frac < 5e-3, (
        f"Sheared Spergel vs GalSim: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, g1={g1}, g2={g2})"
    )


@pytest.mark.parametrize(
    "nu,g1,g2",
    [
        (-0.6, 0.05, -0.03),
    ],
)
def test_galsim_regression_spergel_faceon_with_shear_negative_nu(
    nu, g1, g2, galsim_image_pars
):
    """Face-on sheared Spergel vs GalSim for nu < 0 (PSF + pixel integration)."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(folding_threshold=1e-5, maxk_threshold=1e-5, kvalue_accuracy=1e-7)
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, g1, g2, flux, rscale, 0.1, nu, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars, psf=psf, oversample=5, int_model=model, gsparams=gsp
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    mu = 1.0 / (1.0 - (g1**2 + g2**2))
    gs_profile = gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp)
    gs_profile = gs_profile.lens(g1=g1, g2=g2, mu=mu)
    gs_profile = gs.Convolve(gs_profile, psf)
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert max_frac < 3e-3, (
        f"Sheared Spergel+PSF vs GalSim: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, g1={g1}, g2={g2})"
    )


def test_galsim_regression_devaucouleurs_faceon(galsim_image_pars):
    """Face-on DeVaucouleurs vs GalSim Spergel(nu=-0.6) (PSF + pixel integration)."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(folding_threshold=1e-5, maxk_threshold=1e-5, kvalue_accuracy=1e-7)

    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedDeVaucouleursModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars, psf=psf, oversample=5, int_model=model, gsparams=gsp
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    gs_profile = gs.Convolve(
        gs.Spergel(nu=-0.6, scale_radius=rscale, flux=flux, gsparams=gsp), psf
    )
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert (
        max_frac < 3e-3
    ), f"DeVaucouleurs+PSF vs GalSim Spergel(nu=-0.6): max|resid|/peak = {max_frac:.2e}"


@pytest.mark.parametrize(
    "cosi,theta_int",
    [
        (1.0, 0.0),
        (0.7, 0.0),
        (0.7, np.pi / 4),
        (0.4, 0.0),
        (0.4, np.pi / 4),
    ],
)
def test_galsim_regression_spergel_inclined_nu05(cosi, theta_int, galsim_image_pars):
    """Inclined Spergel(nu=0.5) vs galsim.InclinedExponential (exact match)."""
    from kl_pipe.synthetic import _generate_sersic_galsim
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    h_over_r = 0.1
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.5, 0.0, 0.0]
    )
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=h_over_r,
        gsparams=gsp,
        method='no_pixel',
    )
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert max_frac < 1e-3, (
        f"Inclined Spergel(nu=0.5) vs GalSim InclinedExp: max|resid|/peak = {max_frac:.2e} "
        f"(cosi={cosi}, theta_int={theta_int:.2f})"
    )


def test_spergel_nu05_vs_inclined_sersic_n1(galsim_image_pars):
    """Spergel(nu=0.5) must match InclinedSersic(n=1) — exact equivalence."""
    from kl_pipe.synthetic import _generate_sersic_galsim
    import galsim as gs

    cosi = 0.7
    flux = 1.0
    rscale = 2.0
    h_over_r = 0.1
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, rscale, h_over_r, 0.5, 0.0, 0.0])
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=0.0,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=h_over_r,
        gsparams=gsp,
        method='no_pixel',
    )
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert (
        max_frac < 1e-3
    ), f"Spergel(nu=0.5) vs InclinedSersic(n=1): max|resid|/peak = {max_frac:.2e}"


def test_spergel_vs_inclined_sersic_devac_mismatch(galsim_image_pars):
    """Quantify Spergel(nu=-0.6) vs InclinedSersic(n=4) mismatch (informational).

    Profiles matched at half_light_radius=2.0 for fair shape comparison.
    Uses PSF + method='auto' + coarser grid because InclinedSersic(n=4)
    needs enormous FFT grids at fine pixel scales.
    """
    import galsim as gs

    cosi = 0.7
    inc = gs.Angle(np.arccos(cosi), gs.radians)
    flux = 1.0
    hlr = 2.0
    h_over_r = 0.1
    fwhm = 0.5

    gsp = gs.GSParams(folding_threshold=1e-3, maxk_threshold=1e-3, kvalue_accuracy=1e-5)
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    # GalSim InclinedSersic(n=4) matched at half_light_radius
    gs_prof = gs.InclinedSersic(
        n=4.0,
        inclination=inc,
        half_light_radius=hlr,
        scale_h_over_r=h_over_r,
        flux=flux,
        gsparams=gsp,
    )
    gs_prof = gs.Convolve(gs_prof, psf)

    # coarser grid for tractable GalSim FFT
    ip = ImagePars(shape=(64, 64), pixel_scale=0.5, indexing='ij')
    gs_im = gs_prof.drawImage(
        nx=ip.Ncol,
        ny=ip.Nrow,
        scale=ip.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / ip.pixel_scale**2

    # our Spergel(nu=-0.6) matched at same half_light_radius
    spergel_rscale = gs.Spergel(nu=-0.6, half_light_radius=hlr).scale_radius
    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, 0.0, 0.0, 0.0, flux, spergel_rscale, h_over_r, -0.6, 0.0, 0.0]
    )
    obs = build_image_obs(ip, psf=psf, oversample=5, int_model=model, gsparams=gsp)
    our_image = np.array(model.render_image(theta, obs=obs))

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak
    rms_frac = np.sqrt(np.mean(((our_image - gs_sb) / peak) ** 2))

    # informational — Spergel != Sersic, so we just record the mismatch
    print(
        f"\nSpergel(nu=-0.6) vs InclinedSersic(n=4) at hlr={hlr}:\n"
        f"  max|resid|/peak = {max_frac:.4f}\n"
        f"  rms|resid|/peak = {rms_frac:.4f}\n"
        f"  spergel_rscale = {spergel_rscale:.4f}"
    )


@pytest.fixture
def spergel_output_dir():
    """Dedicated output directory for Spergel diagnostic plots."""
    out_dir = get_test_dir() / "out" / "spergel"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _render_spergel_vs_sersic_panel(
    n_values,
    nu_values,
    cosi_values,
    label,
    spergel_output_dir,
):
    """Render a 4-row x 3-col diagnostic comparing Spergel vs InclinedSersic.

    Returns dict of (n, cosi) -> max_frac for summary statistics.
    """
    import galsim as gs

    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    npix = 64
    ps = 0.5

    gsp = gs.GSParams(
        folding_threshold=1e-3, maxk_threshold=1e-3, maximum_fft_size=16384
    )
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSpergelModel()

    fig, axes = plt.subplots(
        len(cosi_values) * 2,
        len(n_values),
        figsize=(4 * len(n_values), 2.5 * len(cosi_values) * 2),
        gridspec_kw={'height_ratios': [3, 1] * len(cosi_values)},
    )

    stats = {}
    for j, (n, nu) in enumerate(zip(n_values, nu_values)):
        spergel_rscale = gs.Spergel(nu=nu, half_light_radius=hlr).scale_radius

        for i, cosi in enumerate(cosi_values):
            ax_main = axes[i * 2, j]
            ax_resid = axes[i * 2 + 1, j]

            inc = gs.Angle(np.arccos(cosi), gs.radians)

            # GalSim InclinedSersic (3D)
            gs_prof = gs.InclinedSersic(
                n=n,
                inclination=inc,
                half_light_radius=hlr,
                scale_h_over_r=h_over_r,
                flux=flux,
                gsparams=gsp,
            )
            gs_im = gs_prof.drawImage(nx=npix, ny=npix, scale=ps, method='no_pixel')
            gs_sb = gs_im.array / ps**2

            # our InclinedSpergelModel (3D with sech²)
            theta = jnp.array(
                [
                    cosi,
                    0.0,
                    0.0,
                    0.0,
                    flux,
                    spergel_rscale,
                    h_over_r,
                    nu,
                    0.0,
                    0.0,
                ]
            )
            our_sb = np.array(model.render_image(theta, image_pars=ip))

            # 1D major-axis cut
            center = npix // 2
            r_arcsec = (np.arange(npix) - center) * ps
            sersic_cut = gs_sb[center, :]
            spergel_cut = our_sb[center, :]
            peak = np.max(np.abs(sersic_cut))
            residual = spergel_cut - sersic_cut

            # 2D stats
            peak_2d = np.max(np.abs(gs_sb))
            max_frac = np.max(np.abs(our_sb - gs_sb)) / peak_2d
            rms_frac = np.sqrt(np.mean(((our_sb - gs_sb) / peak_2d) ** 2))
            stats[(n, cosi)] = {'max': max_frac, 'rms': rms_frac}

            # main panel
            ax_main.semilogy(
                r_arcsec, np.maximum(sersic_cut, 1e-10), 'b-', label='Sersic'
            )
            ax_main.semilogy(
                r_arcsec, np.maximum(spergel_cut, 1e-10), 'r--', label='Spergel'
            )
            ax_main.axvline(hlr, color='grey', ls=':', alpha=0.5, label='$R_e$')
            ax_main.axvline(-hlr, color='grey', ls=':', alpha=0.5)
            ax_main.axvline(
                spergel_rscale, color='orange', ls=':', alpha=0.5, label='$c$'
            )
            ax_main.axvline(-spergel_rscale, color='orange', ls=':', alpha=0.5)
            ax_main.set_ylim(peak * 1e-4, peak * 2)
            ax_main.set_ylabel('SB')
            if i == 0:
                ax_main.set_title(f'n={n:.0f}, nu={nu:+.2f}')
            if i == 0 and j == 0:
                ax_main.legend(fontsize=7)
            if j == 0:
                ax_main.text(
                    -0.25,
                    0.5,
                    f'cosi={cosi}',
                    transform=ax_main.transAxes,
                    ha='center',
                    va='center',
                    rotation=90,
                    fontsize=10,
                )

            # residual panel
            ax_resid.plot(r_arcsec, residual / peak, 'k-', lw=0.8)
            ax_resid.axhline(0, color='grey', ls='-', alpha=0.3)
            ax_resid.set_ylim(-0.15, 0.15)
            ax_resid.set_ylabel('frac')
            if i == len(cosi_values) - 1:
                ax_resid.set_xlabel('arcsec')
            ax_resid.text(
                0.97,
                0.85,
                f'max={max_frac:.1%}  rms={rms_frac:.1%}',
                transform=ax_resid.transAxes,
                ha='right',
                va='top',
                fontsize=7,
            )

    # summary statistic across all panels
    all_max = [s['max'] for s in stats.values()]
    all_rms = [s['rms'] for s in stats.values()]
    mean_max = np.mean(all_max)
    mean_rms = np.mean(all_rms)

    fig.suptitle(
        f'Spergel vs InclinedSersic — {label}\n'
        f'(hlr-matched, no PSF | mean max={mean_max:.1%}, mean rms={mean_rms:.1%})',
        fontsize=12,
    )
    plt.tight_layout()
    fname = f'spergel_vs_sersic_{label.lower().replace(" ", "_")}.png'
    plt.savefig(spergel_output_dir / fname, dpi=150)
    plt.close()

    return stats


def test_spergel_vs_sersic_inclination_diagnostic(spergel_output_dir):
    """Diagnostic: Spergel vs InclinedSersic with two nu mappings.

    Generates two panel plots (canonical face-on mapping vs inclined mapping)
    and prints summary statistics to compare which mapping works better
    across inclinations. No assertion.
    """
    n_values = [1.0, 2.0, 4.0]
    cosi_values = [1.0, 0.75, 0.25, 0.1]

    # mapping 1: canonical face-on (from flux-weighted L2)
    nu_faceon = [0.5, -0.17, -0.48]
    stats_faceon = _render_spergel_vs_sersic_panel(
        n_values,
        nu_faceon,
        cosi_values,
        'face-on mapping',
        spergel_output_dir,
    )

    # mapping 2: inclined (placeholder values until script finishes)
    # these will be updated with the inclined table results
    nu_inclined = [0.5, -0.53, -0.69]
    stats_inclined = _render_spergel_vs_sersic_panel(
        n_values,
        nu_inclined,
        cosi_values,
        'inclined mapping',
        spergel_output_dir,
    )

    # summary comparison
    print('\nSpergel vs Sersic mapping comparison:')
    print(
        f'  {"n":>3s} {"cosi":>5s}  {"faceon max":>10s} {"incl max":>10s}  '
        f'{"faceon rms":>10s} {"incl rms":>10s}'
    )
    for n in n_values:
        for cosi in cosi_values:
            sf = stats_faceon[(n, cosi)]
            si = stats_inclined[(n, cosi)]
            winner_max = '<' if sf['max'] < si['max'] else '>'
            print(
                f'  {n:3.0f} {cosi:5.2f}  {sf["max"]:10.2%} '
                f'{winner_max} {si["max"]:8.2%}  '
                f'{sf["rms"]:10.2%}   {si["rms"]:8.2%}'
            )

    # aggregate
    fo_max = np.mean([s['max'] for s in stats_faceon.values()])
    in_max = np.mean([s['max'] for s in stats_inclined.values()])
    fo_rms = np.mean([s['rms'] for s in stats_faceon.values()])
    in_rms = np.mean([s['rms'] for s in stats_inclined.values()])
    print(f'\n  Mean max: face-on={fo_max:.2%}  inclined={in_max:.2%}')
    print(f'  Mean rms: face-on={fo_rms:.2%}  inclined={in_rms:.2%}')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
