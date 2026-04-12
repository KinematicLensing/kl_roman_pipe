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
    sersic_to_spergel,
    spergel_to_sersic,
)
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir, build_map_grid_from_image_pars

# maximum GB for a single GalSim FFT allocation in tests
_MAX_FFT_GB = 8.0


def _galsim_fft_safe(profile, pixel_scale, max_gb=_MAX_FFT_GB):
    """Check if a GalSim profile can render without exceeding FFT memory budget.

    Returns (safe, fft_gb) tuple. GalSim InclinedSersic(n=4) segfaults
    at the C level when the FFT is too large — try/except cannot catch it.
    """
    fft_n = int(2 * profile.maxk / profile.stepk)
    fft_gb = fft_n**2 * 16 / 1e9  # complex128
    return fft_gb <= max_gb, fft_gb


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
    render = np.array(
        model.render_image(theta, image_pars=galsim_image_pars, pixel_response=None)
    )

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
        galsim_image_pars,
        psf=psf_obj,
        oversample=5,
        int_model=model,
        pixel_response=None,
    )
    img_kspace = np.array(model.render_image(theta, obs=obs_kspace))

    # fallback real-space path (no int_model -> no kspace_psf_fft)
    obs_realspace = build_image_obs(
        galsim_image_pars,
        psf=psf_obj,
        oversample=5,
        pixel_response=None,
    )
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


# --- nu <-> n mapping ---


@pytest.mark.parametrize(
    "n,expected_nu,tol",
    [
        (1.0, 0.5, 1e-6),  # exact
        (2.0, -0.17, 0.02),
        (4.0, -0.48, 0.02),
    ],
)
def test_sersic_to_spergel(n, expected_nu, tol):
    """sersic_to_spergel returns expected values."""
    nu = sersic_to_spergel(n)
    assert (
        abs(nu - expected_nu) < tol
    ), f"sersic_to_spergel({n}) = {nu:.4f}, expected ~{expected_nu}"


@pytest.mark.parametrize("n_input", [0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
def test_nu_n_roundtrip_faceon(n_input):
    """n -> nu -> n roundtrip within 1% for face-on table."""
    nu = sersic_to_spergel(n_input, inclined=False)
    n_back = spergel_to_sersic(nu, inclined=False)
    rel_err = abs(n_back - n_input) / n_input
    assert (
        rel_err < 0.01
    ), f"Roundtrip: n={n_input} -> nu={nu:.4f} -> n={n_back:.4f} ({rel_err:.1%})"


@pytest.mark.parametrize("n_input", [1.0, 1.5, 2.0, 3.0, 4.0])
def test_nu_n_roundtrip_inclined(n_input):
    """n -> nu -> n roundtrip within 1% for inclined table."""
    nu = sersic_to_spergel(n_input, inclined=True)
    n_back = spergel_to_sersic(nu, inclined=True)
    rel_err = abs(n_back - n_input) / n_input
    assert (
        rel_err < 0.01
    ), f"Roundtrip: n={n_input} -> nu={nu:.4f} -> n={n_back:.4f} ({rel_err:.1%})"


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

    # pixel_response=None on both sides: point-sampled method comparison
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
        pixel_response=None,
    )

    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, nu, int_x0, int_y0]
    )
    render = np.array(
        model.render_image(theta, image_pars=galsim_image_pars, pixel_response=None)
    )

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
    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )

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
    gsp = gs.GSParams(
        folding_threshold=1e-5,
        maxk_threshold=1e-5,
        kvalue_accuracy=1e-7,
        maximum_fft_size=32768,
    )
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, nu, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars,
        psf=psf,
        oversample=5,
        int_model=model,
        gsparams=gsp,
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

    # cuspy profiles (nu < 0.5): k-space sinc pixel response vs GalSim real-space
    # pixel integration diverge because FT decays as k^{-(2+2nu)}, leaving large
    # power at Nyquist that the two methods handle differently.
    # nu=-0.6: ~5.5%, nu=-0.3: ~3.5%, nu=0.0: ~2%
    tol = 6e-2
    assert (
        max_frac < tol
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
    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )

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
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    # 5e-3: sinc pixel response (ours) vs GalSim pixel integration can differ
    # due to DFT aliasing at profile cusps
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
    gsp = gs.GSParams(
        folding_threshold=1e-5,
        maxk_threshold=1e-5,
        kvalue_accuracy=1e-7,
        maximum_fft_size=32768,
    )
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, g1, g2, flux, rscale, 0.1, nu, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars,
        psf=psf,
        oversample=5,
        int_model=model,
        gsparams=gsp,
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

    # cuspy nu < 0: sinc vs GalSim pixel integration diverge at high-k (see
    # test_galsim_regression_spergel_faceon_cuspy for explanation)
    assert max_frac < 6e-2, (
        f"Sheared Spergel+PSF vs GalSim: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, g1={g1}, g2={g2})"
    )


def test_galsim_regression_devaucouleurs_faceon(galsim_image_pars):
    """Face-on DeVaucouleurs vs GalSim Spergel(nu=-0.6) (PSF + pixel integration)."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(
        folding_threshold=1e-5,
        maxk_threshold=1e-5,
        kvalue_accuracy=1e-7,
        maximum_fft_size=32768,
    )

    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedDeVaucouleursModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars,
        psf=psf,
        oversample=5,
        int_model=model,
        gsparams=gsp,
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

    # nu=-0.6: sinc vs GalSim pixel integration diverge at high-k (see
    # test_galsim_regression_spergel_faceon_cuspy for explanation)
    assert (
        max_frac < 6e-2
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
    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )

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
        method='auto',
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
    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )

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
        method='auto',
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

    gsp = gs.GSParams(
        folding_threshold=1e-3,
        maxk_threshold=1e-3,
        kvalue_accuracy=1e-5,
        maximum_fft_size=32768,
    )
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    # GalSim InclinedSersic(n=4) matched at half_light_radius
    gs_sersic = gs.InclinedSersic(
        n=4.0,
        inclination=inc,
        half_light_radius=hlr,
        scale_h_over_r=h_over_r,
        flux=flux,
        gsparams=gsp,
    )

    # coarser grid for tractable GalSim FFT
    ip = ImagePars(shape=(64, 64), pixel_scale=0.5, indexing='ij')

    # pre-screen to avoid C-level segfault
    safe, fft_gb = _galsim_fft_safe(gs_sersic, ip.pixel_scale)
    if not safe:
        pytest.skip(
            f'GalSim InclinedSersic(n=4) FFT needs {fft_gb:.1f} GB '
            f'(limit {_MAX_FFT_GB} GB)'
        )

    gs_prof = gs.Convolve(gs_sersic, psf)
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


def _render_spergel_vs_sersic_panel(
    n_values,
    nu_values,
    cosi_values,
    label,
    spergel_output_dir,
    col_labels=None,
    psf_fwhm=None,
):
    """Render diagnostic comparing Spergel vs InclinedSersic across inclinations.

    X-axis in units of R_e (half-light radius). Radial profiles are
    azimuthally averaged using log-spaced bins to emphasize the core.

    Parameters
    ----------
    psf_fwhm : float, optional
        Gaussian PSF FWHM in arcsec. None = no PSF (point-sampled).

    Returns dict of (n, cosi, col_idx) -> {max, rms} for summary statistics.
    """
    import galsim as gs

    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    npix = 128
    ps = 0.11  # Roman pixel scale

    gsp = gs.GSParams(
        folding_threshold=1e-3,
        maxk_threshold=1e-3,
        maximum_fft_size=32768,
    )
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSpergelModel()

    use_psf = psf_fwhm is not None
    psf_obj = gs.Gaussian(fwhm=psf_fwhm, gsparams=gsp) if use_psf else None
    draw_method = 'auto' if use_psf else 'no_pixel'
    oversample = 5 if use_psf else 1

    # log-spaced radial bins in units of R_e
    r_bin_edges = np.concatenate([[0], np.logspace(-1.5, np.log10(5.0), 40)])
    r_mid = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

    # pixel radius grid (reusable across panels)
    center = npix // 2
    y_idx, x_idx = np.mgrid[:npix, :npix]
    r_re = np.sqrt(((x_idx - center) * ps) ** 2 + ((y_idx - center) * ps) ** 2) / hlr

    fig, axes = plt.subplots(
        len(cosi_values) * 2,
        len(n_values),
        figsize=(4.5 * len(n_values), 2.2 * len(cosi_values) * 2),
        gridspec_kw={'height_ratios': [3, 1] * len(cosi_values)},
    )

    stats = {}
    for j, (n, nu) in enumerate(zip(n_values, nu_values)):
        spergel_c = gs.Spergel(nu=nu, half_light_radius=hlr).scale_radius
        sersic_rs = gs.InclinedSersic(
            n=n,
            inclination=0 * gs.radians,
            half_light_radius=hlr,
            scale_h_over_r=h_over_r,
            flux=flux,
            gsparams=gsp,
        ).scale_radius

        for i, cosi in enumerate(cosi_values):
            ax_main = axes[i * 2, j]
            ax_resid = axes[i * 2 + 1, j]

            inc = gs.Angle(np.arccos(cosi), gs.radians)

            # GalSim InclinedSersic (3D), k-space FFT
            gs_prof = gs.InclinedSersic(
                n=n,
                inclination=inc,
                half_light_radius=hlr,
                scale_h_over_r=h_over_r,
                flux=flux,
                gsparams=gsp,
            )

            # pre-screen FFT size to avoid C-level segfault
            safe, fft_gb = _galsim_fft_safe(gs_prof, ps)
            if not safe:
                print(
                    f'  SKIP n={n}, cosi={cosi}: GalSim FFT needs '
                    f'{fft_gb:.1f} GB (limit {_MAX_FFT_GB} GB)'
                )
                ax_main.text(
                    0.5,
                    0.5,
                    f'FFT too large\n({fft_gb:.1f} GB)',
                    transform=ax_main.transAxes,
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='red',
                )
                ax_resid.text(
                    0.5,
                    0.5,
                    'skipped',
                    transform=ax_resid.transAxes,
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='red',
                )
                stats[(n, cosi, j)] = {'max': np.nan, 'rms': np.nan}
                continue

            gs_draw = gs.Convolve(gs_prof, psf_obj) if use_psf else gs_prof
            gs_im = gs_draw.drawImage(nx=npix, ny=npix, scale=ps, method=draw_method)
            gs_sb = gs_im.array / ps**2

            # our InclinedSpergelModel (3D with sech²)
            theta = jnp.array(
                [cosi, 0.0, 0.0, 0.0, flux, spergel_c, h_over_r, nu, 0.0, 0.0]
            )
            if use_psf:
                obs = build_image_obs(
                    ip,
                    psf=psf_obj,
                    oversample=oversample,
                    int_model=model,
                    gsparams=gsp,
                )
                our_sb = np.array(model.render_image(theta, obs=obs))
            else:
                our_sb = np.array(model.render_image(theta, image_pars=ip))

            # azimuthally averaged radial profiles in R_e units
            sersic_prof = np.full(len(r_mid), np.nan)
            spergel_prof = np.full(len(r_mid), np.nan)
            for k in range(len(r_mid)):
                mask = (r_re >= r_bin_edges[k]) & (r_re < r_bin_edges[k + 1])
                if np.any(mask):
                    sersic_prof[k] = np.mean(gs_sb[mask])
                    spergel_prof[k] = np.mean(our_sb[mask])

            valid = np.isfinite(sersic_prof) & (sersic_prof > 0)
            peak = np.nanmax(sersic_prof[valid])
            residual = np.where(valid, (spergel_prof - sersic_prof) / peak, np.nan)

            # 2D stats
            peak_2d = np.max(np.abs(gs_sb))
            max_frac = np.max(np.abs(our_sb - gs_sb)) / peak_2d
            rms_frac = np.sqrt(np.mean(((our_sb - gs_sb) / peak_2d) ** 2))
            stats[(n, cosi, j)] = {'max': max_frac, 'rms': rms_frac}

            # main panel: log-log radial profile
            ax_main.semilogy(
                r_mid[valid],
                sersic_prof[valid],
                'b-',
                lw=1.5,
                label='Sersic' if i == 0 else None,
            )
            ax_main.semilogy(
                r_mid[valid],
                spergel_prof[valid],
                'r--',
                lw=1.5,
                label='Spergel' if i == 0 else None,
            )
            ax_main.axvline(
                1.0,
                color='grey',
                ls='-',
                alpha=0.4,
                lw=1.5,
                label='$R_e$ (shared)' if (i == 0 and j == 0) else None,
            )
            ax_main.axvline(
                sersic_rs / hlr,
                color='blue',
                ls=':',
                alpha=0.5,
                lw=1,
                label=f'Sersic $r_s$={sersic_rs:.2f}"' if i == 0 else None,
            )
            ax_main.axvline(
                spergel_c / hlr,
                color='red',
                ls=':',
                alpha=0.5,
                lw=1,
                label=f'Spergel $c$={spergel_c:.2f}"' if i == 0 else None,
            )
            ax_main.set_xlim(0.02, 5.0)
            ax_main.set_xscale('log')
            ax_main.set_ylim(peak * 1e-4, peak * 3)
            ax_main.set_ylabel('SB')
            if i == 0:
                if col_labels is not None:
                    ax_main.set_title(col_labels[j], fontsize=9)
                else:
                    ax_main.set_title(f'n={n:.0f}, nu={nu:+.2f}')
                ax_main.legend(fontsize=6, loc='upper right')
            if j == 0:
                ax_main.text(
                    -0.22,
                    0.5,
                    f'cosi={cosi}',
                    transform=ax_main.transAxes,
                    ha='center',
                    va='center',
                    rotation=90,
                    fontsize=10,
                )

            # residual panel
            ax_resid.plot(r_mid[valid], residual[valid], 'k-', lw=0.8)
            ax_resid.axhline(0, color='grey', ls='-', alpha=0.3)
            ax_resid.set_ylim(-0.15, 0.15)
            ax_resid.set_xlim(0.02, 5.0)
            ax_resid.set_xscale('log')
            ax_resid.set_ylabel('frac resid')
            if i == len(cosi_values) - 1:
                ax_resid.set_xlabel('$r / R_e$')
            ax_resid.text(
                0.97,
                0.85,
                f'max={max_frac:.1%}  rms={rms_frac:.1%}',
                transform=ax_resid.transAxes,
                ha='right',
                va='top',
                fontsize=7,
            )

    # summary (exclude NaN from skipped panels)
    all_max = [s['max'] for s in stats.values() if np.isfinite(s.get('max', np.nan))]
    all_rms = [s['rms'] for s in stats.values() if np.isfinite(s.get('rms', np.nan))]
    mean_max = np.mean(all_max) if all_max else np.nan
    mean_rms = np.mean(all_rms) if all_rms else np.nan

    fig.suptitle(
        f'Spergel vs InclinedSersic — {label}\n'
        f'(hlr-matched at {hlr}", {"PSF FWHM=" + str(psf_fwhm) + chr(34) if use_psf else "no PSF"}, {ps}"/pix '
        f'| mean max={mean_max:.1%}, mean rms={mean_rms:.1%})',
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    fname = f'spergel_vs_sersic_{label.lower().replace(" ", "_")}.png'
    plt.savefig(spergel_output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()

    return stats


def test_spergel_vs_sersic_inclination_diagnostic(spergel_output_dir):
    """Diagnostic: Spergel vs InclinedSersic in a single 4-column plot.

    Columns: n=1, n=2, n=4 (L2-optimal nu=-0.48), n=4 (convention nu=-0.6)
    Rows: one pair (profile + residual) per inclination.
    L2-optimal = flux-weighted L2 profile matching (compute_nu_n_mapping.py).
    No assertion — diagnostic only.
    """
    # 4 columns: n=1, n=2, n=4 at two different nu values
    n_values = [1.0, 2.0, 4.0, 4.0]
    nu_l2_n4 = float(sersic_to_spergel(4.0, inclined=False))
    nu_values = [
        0.5,  # n=1 (exact)
        float(sersic_to_spergel(2.0, inclined=False)),  # n=2 (L2-optimal)
        nu_l2_n4,  # n=4 (L2-optimal)
        -0.6,  # n=4 (convention)
    ]
    col_labels = [
        f'n=1, nu=+0.50',
        f'n=2, nu={nu_values[1]:+.2f}',
        f'n=4, nu={nu_l2_n4:+.2f} (L2-opt)',
        f'n=4, nu=−0.60 (convention)',
    ]
    # cosi=0.1 excluded: InclinedSersic(n=4) needs ~45 GB FFT
    cosi_values = [1.0, 0.75, 0.5, 0.25]

    def _print_summary(tag, stats):
        print(f'\nSpergel vs InclinedSersic — {tag}:')
        print(f'  {"col":>25s} {"cosi":>5s}  {"max":>8s}  {"rms":>8s}')
        for j, lab in enumerate(col_labels):
            n = n_values[j]
            for cosi in cosi_values:
                s = stats.get((n, cosi, j))
                if s is None or np.isnan(s.get('max', np.nan)):
                    print(f'  {lab:>25s} {cosi:5.2f}  (skipped)')
                    continue
                print(f'  {lab:>25s} {cosi:5.2f}  {s["max"]:8.2%}  {s["rms"]:8.2%}')

    # no PSF
    stats_nopsf = _render_spergel_vs_sersic_panel(
        n_values,
        nu_values,
        cosi_values,
        'L2-opt vs convention (no PSF)',
        spergel_output_dir,
        col_labels=col_labels,
    )
    _print_summary('no PSF', stats_nopsf)

    # with PSF
    stats_psf = _render_spergel_vs_sersic_panel(
        n_values,
        nu_values,
        cosi_values,
        'L2-opt vs convention (PSF 0.15)',
        spergel_output_dir,
        col_labels=col_labels,
        psf_fwhm=0.15,
    )
    _print_summary('PSF FWHM=0.15"', stats_psf)


# ==============================================================================
# 2D image comparison diagnostics
# ==============================================================================


def _n4_2d_diagnostic(spergel_output_dir, psf_fwhm=None, cosi=1.0):
    """2D image comparison: Sersic(n=4) vs Spergel at nu=-0.6 and nu=-0.48.

    Face-on layout (4 rows):
      Row 1: GalSim Sersic(n=4) | GalSim Spergel(-0.6) | GalSim Spergel(-0.48)
      Row 2: GS Spergel residuals from Sersic
      Row 3: GalSim Sersic(n=4) | Our Spergel(-0.6) | Our Spergel(-0.48)
      Row 4: Our Spergel residuals from Sersic

    Inclined layout (2 rows — no GalSim inclined Spergel exists):
      Row 1: GalSim InclinedSersic(n=4) | Our Spergel(-0.6) | Our Spergel(-0.48)
      Row 2: Our Spergel residuals from InclinedSersic
    """
    import galsim as gs

    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    npix = 128
    ps = 0.11
    nu_values = [-0.6, -0.48]

    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )

    use_psf = psf_fwhm is not None
    psf_obj = gs.Gaussian(fwhm=psf_fwhm, gsparams=gsp) if use_psf else None
    draw_method = 'auto' if use_psf else 'no_pixel'
    oversample = 5 if use_psf else 1
    is_faceon = abs(cosi - 1.0) < 0.01

    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSpergelModel()
    extent = np.array([-npix / 2, npix / 2, -npix / 2, npix / 2]) * ps

    # GalSim Sersic(n=4) baseline
    gsp_sersic = gs.GSParams(
        folding_threshold=1e-3,
        maxk_threshold=1e-3,
        maximum_fft_size=32768,
    )
    if is_faceon:
        gs_sersic_base = gs.Sersic(n=4, half_light_radius=hlr, flux=flux, gsparams=gsp)
    else:
        inc = gs.Angle(np.arccos(cosi), gs.radians)
        gs_sersic_base = gs.InclinedSersic(
            n=4,
            inclination=inc,
            half_light_radius=hlr,
            scale_h_over_r=h_over_r,
            flux=flux,
            gsparams=gsp_sersic,
        )
        safe, fft_gb = _galsim_fft_safe(gs_sersic_base, ps)
        if not safe:
            print(f'  SKIP: InclinedSersic(n=4) cosi={cosi} needs {fft_gb:.1f} GB')
            return

    gs_sersic_draw = gs.Convolve(gs_sersic_base, psf_obj) if use_psf else gs_sersic_base
    sersic_sb = (
        gs_sersic_draw.drawImage(
            nx=npix,
            ny=npix,
            scale=ps,
            method=draw_method,
        ).array
        / ps**2
    )

    # our Spergel renders (always available)
    our_spergels = {}
    for nu in nu_values:
        rscale = gs.Spergel(nu=nu, half_light_radius=hlr).scale_radius
        theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, rscale, h_over_r, nu, 0.0, 0.0])
        if use_psf:
            obs = build_image_obs(
                ip,
                psf=psf_obj,
                oversample=oversample,
                int_model=model,
                gsparams=gsp,
            )
            our_spergels[nu] = np.array(model.render_image(theta, obs=obs))
        else:
            our_spergels[nu] = np.array(model.render_image(theta, image_pars=ip))

    # GalSim Spergel renders (only for face-on — no inclined GalSim Spergel)
    gs_spergels = {}
    if is_faceon:
        for nu in nu_values:
            gs_sp = gs.Spergel(nu=nu, half_light_radius=hlr, flux=flux, gsparams=gsp)
            gs_sp_draw = gs.Convolve(gs_sp, psf_obj) if use_psf else gs_sp
            gs_spergels[nu] = (
                gs_sp_draw.drawImage(
                    nx=npix,
                    ny=npix,
                    scale=ps,
                    method=draw_method,
                ).array
                / ps**2
            )

    peak = np.max(np.abs(sersic_sb))

    def _plot_img(ax, img, title, vmin, vmax, cmap='viridis', is_resid=False):
        if is_resid:
            im = ax.imshow(
                img, origin='lower', extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax
            )
        else:
            im = ax.imshow(
                img,
                origin='lower',
                extent=extent,
                cmap=cmap,
                norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        plt.colorbar(im, ax=ax, shrink=0.8)
        if is_resid and peak > 0:
            mf = np.max(np.abs(img)) / peak
            rf = np.sqrt(np.mean(img**2)) / peak
            ax.text(
                0.03,
                0.97,
                f'max={mf:.2%}\nrms={rf:.3%}',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=7,
                family='monospace',
                color='white',
                bbox=dict(facecolor='black', alpha=0.6),
            )

    vmax_img = np.max(sersic_sb)
    vmin_img = vmax_img * 1e-4

    # compute shared residual scale
    all_resid = [np.max(np.abs(our_spergels[nu] - sersic_sb)) for nu in nu_values]
    if gs_spergels:
        all_resid += [np.max(np.abs(gs_spergels[nu] - sersic_sb)) for nu in nu_values]
    vmax_resid = max(all_resid) * 0.8

    psf_str = f'PSF FWHM={psf_fwhm}"' if use_psf else 'no PSF'
    cosi_str = f'cosi={cosi}' if not is_faceon else 'face-on'
    sersic_label = 'InclinedSersic(n=4)' if not is_faceon else 'Sersic(n=4)'

    if is_faceon:
        # 4-row layout: GalSim Spergel block + Our Spergel block
        fig, axes = plt.subplots(
            4, 3, figsize=(15, 18), gridspec_kw={'height_ratios': [1, 0.8, 1, 0.8]}
        )

        _plot_img(axes[0, 0], sersic_sb, f'GalSim {sersic_label}', vmin_img, vmax_img)
        _plot_img(
            axes[0, 1], gs_spergels[-0.6], 'GalSim Spergel(ν=−0.6)', vmin_img, vmax_img
        )
        _plot_img(
            axes[0, 2],
            gs_spergels[-0.48],
            'GalSim Spergel(ν=−0.48)',
            vmin_img,
            vmax_img,
        )

        axes[1, 0].set_visible(False)
        _plot_img(
            axes[1, 1],
            gs_spergels[-0.6] - sersic_sb,
            'GS Spergel(−0.6) − Sersic',
            0,
            vmax_resid,
            'RdBu_r',
            True,
        )
        _plot_img(
            axes[1, 2],
            gs_spergels[-0.48] - sersic_sb,
            'GS Spergel(−0.48) − Sersic',
            0,
            vmax_resid,
            'RdBu_r',
            True,
        )

        _plot_img(axes[2, 0], sersic_sb, f'GalSim {sersic_label}', vmin_img, vmax_img)
        _plot_img(
            axes[2, 1], our_spergels[-0.6], 'Our Spergel(ν=−0.6)', vmin_img, vmax_img
        )
        _plot_img(
            axes[2, 2], our_spergels[-0.48], 'Our Spergel(ν=−0.48)', vmin_img, vmax_img
        )

        axes[3, 0].set_visible(False)
        _plot_img(
            axes[3, 1],
            our_spergels[-0.6] - sersic_sb,
            'Our Spergel(−0.6) − Sersic',
            0,
            vmax_resid,
            'RdBu_r',
            True,
        )
        _plot_img(
            axes[3, 2],
            our_spergels[-0.48] - sersic_sb,
            'Our Spergel(−0.48) − Sersic',
            0,
            vmax_resid,
            'RdBu_r',
            True,
        )
    else:
        # 2-row layout: our Spergel vs InclinedSersic only
        fig, axes = plt.subplots(
            2, 3, figsize=(15, 9), gridspec_kw={'height_ratios': [1, 0.8]}
        )

        _plot_img(axes[0, 0], sersic_sb, f'GalSim {sersic_label}', vmin_img, vmax_img)
        _plot_img(
            axes[0, 1], our_spergels[-0.6], 'Our Spergel(ν=−0.6)', vmin_img, vmax_img
        )
        _plot_img(
            axes[0, 2], our_spergels[-0.48], 'Our Spergel(ν=−0.48)', vmin_img, vmax_img
        )

        axes[1, 0].set_visible(False)
        _plot_img(
            axes[1, 1],
            our_spergels[-0.6] - sersic_sb,
            'Our Spergel(−0.6) − Sersic',
            0,
            vmax_resid,
            'RdBu_r',
            True,
        )
        _plot_img(
            axes[1, 2],
            our_spergels[-0.48] - sersic_sb,
            'Our Spergel(−0.48) − Sersic',
            0,
            vmax_resid,
            'RdBu_r',
            True,
        )

    fig.suptitle(
        f'2D: {sersic_label} vs Spergel — {psf_str}, {cosi_str}\n'
        f'hlr={hlr}", {ps}"/pix, {npix}×{npix}',
        fontsize=12,
    )
    plt.tight_layout()
    suffix_parts = ['2d']
    suffix_parts.append('nopsf' if not use_psf else f'psf_{psf_fwhm}')
    if not is_faceon:
        suffix_parts.append(f'cosi{cosi}')
    fname = f'n4_{"_".join(suffix_parts)}.png'
    plt.savefig(spergel_output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {spergel_output_dir / fname}')


def test_n4_2d_diagnostic(spergel_output_dir):
    """2D image comparison: Sersic(n=4) vs Spergel.

    Face-on: GalSim Spergel + our Spergel vs Sersic (no PSF and with PSF).
    Inclined: our Spergel vs InclinedSersic at cosi=0.25, 0.5, 0.75 (with PSF).
    """
    # face-on, no PSF
    _n4_2d_diagnostic(spergel_output_dir, psf_fwhm=None, cosi=1.0)
    # face-on, with PSF
    _n4_2d_diagnostic(spergel_output_dir, psf_fwhm=0.15, cosi=1.0)
    # inclined, with PSF
    for cosi in [0.25, 0.5, 0.75]:
        _n4_2d_diagnostic(spergel_output_dir, psf_fwhm=0.15, cosi=cosi)


# ==============================================================================
# Rendering accuracy diagnostics (Tier 1-2)
# ==============================================================================


def _radial_profile(image, r_re, n_bins=100):
    """Azimuthally averaged radial profile with log-spaced bins in R_e."""
    edges = np.concatenate([[0], np.logspace(-1.5, np.log10(6.0), n_bins)])
    mid = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(len(mid), np.nan)
    for k in range(len(mid)):
        mask = (r_re >= edges[k]) & (r_re < edges[k + 1])
        if np.any(mask):
            prof[k] = np.mean(image[mask])
    return mid, prof


def _annular_stats(test, ref, r_re, bands):
    """Residual stats by annular region in units of R_e."""
    peak = np.max(np.abs(ref))
    results = {}
    for label, r_lo, r_hi in bands:
        mask = (r_re >= r_lo) & (r_re < r_hi)
        if not np.any(mask) or peak == 0:
            results[label] = {'max_frac': np.nan, 'rms_frac': np.nan}
            continue
        diff = test[mask] - ref[mask]
        results[label] = {
            'max_frac': np.max(np.abs(diff)) / peak,
            'rms_frac': np.sqrt(np.mean(diff**2)) / peak,
        }
    return results


_BANDS = [
    ('core', 0.0, 1.0),
    ('mid', 1.0, 3.0),
    ('wings', 3.0, 6.0),
    ('total', 0.0, 6.0),
]


def _faceon_n4_diagnostic(
    spergel_output_dir, psf_obj=None, psf_label='no PSF', fwhm_str=''
):
    """Comprehensive face-on Sersic(n=4) vs Spergel diagnostic.

    2 cols (nu=-0.6, nu=-0.48) x 4 rows:
      Row 1-2: GalSim Spergel vs GalSim Sersic(n=4) [approximation quality]
      Row 3-4: Our Spergel vs GalSim Sersic(n=4) [total error]
    """
    import galsim as gs

    hlr = 2.0
    flux = 1.0
    npix = 128
    ps = 0.11
    h_over_r = 0.1
    nu_values = [-0.6, -0.48]
    nu_labels = [r'$\nu$=−0.6 (convention)', r'$\nu$=−0.48 (L2-optimal)']

    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )

    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSpergelModel()
    center = npix // 2
    y, x = np.mgrid[:npix, :npix]
    r_re = np.sqrt(((x - center) * ps) ** 2 + ((y - center) * ps) ** 2) / hlr

    use_psf = psf_obj is not None
    draw_method = 'auto' if use_psf else 'no_pixel'
    oversample = 5 if use_psf else 1

    # GalSim Sersic(n=4) baseline
    gs_sersic = gs.Sersic(n=4, half_light_radius=hlr, flux=flux, gsparams=gsp)
    gs_sersic_draw = gs.Convolve(gs_sersic, psf_obj) if use_psf else gs_sersic
    sersic_sb = (
        gs_sersic_draw.drawImage(
            nx=npix,
            ny=npix,
            scale=ps,
            method=draw_method,
        ).array
        / ps**2
    )
    r_mid, sersic_prof = _radial_profile(sersic_sb, r_re)
    valid = np.isfinite(sersic_prof) & (sersic_prof > 0)
    peak = np.nanmax(sersic_prof[valid])

    fig, axes = plt.subplots(
        4, 2, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1.2, 3, 1.2]}
    )

    for j, (nu, nu_lab) in enumerate(zip(nu_values, nu_labels)):
        # GalSim Spergel
        gs_sp = gs.Spergel(nu=nu, half_light_radius=hlr, flux=flux, gsparams=gsp)
        gs_sp_draw = gs.Convolve(gs_sp, psf_obj) if use_psf else gs_sp
        gs_sp_sb = (
            gs_sp_draw.drawImage(
                nx=npix,
                ny=npix,
                scale=ps,
                method=draw_method,
            ).array
            / ps**2
        )
        _, gs_sp_prof = _radial_profile(gs_sp_sb, r_re)
        gs_stats = _annular_stats(gs_sp_sb, sersic_sb, r_re, _BANDS)

        # Our Spergel
        rscale = gs.Spergel(nu=nu, half_light_radius=hlr).scale_radius
        theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, h_over_r, nu, 0.0, 0.0])
        if use_psf:
            obs = build_image_obs(
                ip,
                psf=psf_obj,
                oversample=oversample,
                int_model=model,
                gsparams=gsp,
            )
            our_sb = np.array(model.render_image(theta, obs=obs))
        else:
            our_sb = np.array(model.render_image(theta, image_pars=ip))
        _, our_prof = _radial_profile(our_sb, r_re)
        our_stats = _annular_stats(our_sb, sersic_sb, r_re, _BANDS)

        def _stat_text(stats):
            return '\n'.join(
                f'{lab}: max={s["max_frac"]:.3%}, rms={s["rms_frac"]:.3%}'
                for lab, s in stats.items()
            )

        def _annotate(ax, stats):
            ax.text(
                0.97,
                0.95,
                _stat_text(stats),
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=6,
                family='monospace',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'),
            )

        # Row 1: GalSim Spergel vs Sersic profiles
        ax = axes[0, j]
        ax.semilogy(
            r_mid[valid], sersic_prof[valid], 'b-', lw=1.5, label='GalSim Sersic(n=4)'
        )
        ax.semilogy(
            r_mid[valid],
            gs_sp_prof[valid],
            'r--',
            lw=1.5,
            label=f'GalSim Spergel({nu})',
        )
        ax.axvline(1.0, color='grey', ls=':', alpha=0.4)
        ax.axvline(3.0, color='grey', ls='--', alpha=0.3)
        ax.set_xlim(0.02, 6.0)
        ax.set_xscale('log')
        ax.set_ylim(peak * 1e-5, peak * 3)
        ax.set_ylabel('Surface brightness')
        ax.set_title(f'{nu_lab}\nGalSim Spergel vs Sersic(n=4)', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

        # Row 2: GalSim Spergel residual
        ax = axes[1, j]
        gs_resid = np.where(valid, (gs_sp_prof - sersic_prof) / peak, np.nan)
        ax.plot(r_mid[valid], gs_resid[valid], 'r-', lw=0.8)
        ax.axhline(0, color='grey', ls='-', alpha=0.3)
        ax.set_xlim(0.02, 6.0)
        ax.set_xscale('log')
        ylim = max(0.05, np.nanmax(np.abs(gs_resid[valid])) * 1.3)
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel('(GS Spergel − Sersic) / peak')
        _annotate(ax, gs_stats)

        # Row 3: Our Spergel vs Sersic profiles
        ax = axes[2, j]
        ax.semilogy(
            r_mid[valid], sersic_prof[valid], 'b-', lw=1.5, label='GalSim Sersic(n=4)'
        )
        ax.semilogy(
            r_mid[valid], our_prof[valid], 'm--', lw=1.5, label=f'Our Spergel({nu})'
        )
        ax.axvline(1.0, color='grey', ls=':', alpha=0.4)
        ax.axvline(3.0, color='grey', ls='--', alpha=0.3)
        ax.set_xlim(0.02, 6.0)
        ax.set_xscale('log')
        ax.set_ylim(peak * 1e-5, peak * 3)
        ax.set_ylabel('Surface brightness')
        ax.set_title(f'Our Spergel({nu}) vs Sersic(n=4)', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

        # Row 4: Our Spergel residual
        ax = axes[3, j]
        our_resid = np.where(valid, (our_prof - sersic_prof) / peak, np.nan)
        ax.plot(r_mid[valid], our_resid[valid], 'm-', lw=0.8)
        ax.axhline(0, color='grey', ls='-', alpha=0.3)
        ax.set_xlim(0.02, 6.0)
        ax.set_xscale('log')
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel('(Our Spergel − Sersic) / peak')
        ax.set_xlabel('$r / R_e$')
        _annotate(ax, our_stats)

        # print
        for src_label, st in [('GalSim Spergel', gs_stats), ('Our Spergel', our_stats)]:
            print(f'\n  {src_label} (nu={nu}), {psf_label}:')
            for band, s in st.items():
                print(
                    f'    {band:6s}: max={s["max_frac"]:.4%}, rms={s["rms_frac"]:.4%}'
                )

    fig.suptitle(
        f'Face-on Sersic(n=4) vs Spergel approximation — {psf_label}\n'
        f'hlr={hlr}", {ps}"/pix, {npix}×{npix} | '
        f'L2-optimal = flux-weighted L2 profile matching',
        fontsize=12,
    )
    plt.tight_layout()
    suffix = 'nopsf' if not use_psf else f'psf_{fwhm_str}'
    fname = f'faceon_n4_diagnostic_{suffix}.png'
    plt.savefig(spergel_output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


def test_faceon_n4_diagnostic_nopsf(spergel_output_dir):
    """Face-on Sersic(n=4) vs Spergel approximation diagnostic (no PSF)."""
    _faceon_n4_diagnostic(spergel_output_dir, psf_obj=None, psf_label='no PSF')


def test_faceon_n4_diagnostic_psf(spergel_output_dir):
    """Face-on Sersic(n=4) vs Spergel approximation diagnostic (with PSF)."""
    import galsim as gs

    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )
    fwhm = 0.15
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)
    _faceon_n4_diagnostic(
        spergel_output_dir,
        psf_obj=psf,
        psf_label=f'PSF FWHM={fwhm}"',
        fwhm_str=f'{fwhm}',
    )


def test_oversample_convergence(spergel_output_dir):
    """Oversample convergence: face-on vs GalSim + inclined self-convergence.

    Face-on: residuals against gs.Spergel(nu=-0.6) ground truth.
    Inclined: self-convergence against oversample=15 (no external reference;
    validates that inclined rendering converges without aliasing artifacts).
    """
    import galsim as gs

    nu = -0.6
    hlr = 2.0
    flux = 1.0
    npix = 128
    ps = 0.11
    h_over_r = 0.1
    fwhm = 0.15

    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )
    gs_sp = gs.Spergel(nu=nu, half_light_radius=hlr, flux=flux, gsparams=gsp)
    rscale = gs_sp.scale_radius
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSpergelModel()

    oversamples = [1, 3, 5, 7, 9, 15]
    ref_osamp = 15
    cosi_values = [1.0, 0.3, 0.5, 0.7]
    col_labels = [
        'Face-on vs GalSim',
        'cosi=0.3 (self-conv, ref=os15)',
        'cosi=0.5 (self-conv, ref=os15)',
        'cosi=0.7 (self-conv, ref=os15)',
    ]

    fig, axes = plt.subplots(1, len(cosi_values), figsize=(4.5 * len(cosi_values), 5))

    for ci, (cosi, col_lab) in enumerate(zip(cosi_values, col_labels)):
        theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, rscale, h_over_r, nu, 0.0, 0.0])
        ax = axes[ci]

        if ci == 0:
            gs_conv = gs.Convolve(gs_sp, psf)
            ref_sb = (
                gs_conv.drawImage(
                    nx=npix,
                    ny=npix,
                    scale=ps,
                    method='auto',
                ).array
                / ps**2
            )
            ref_label = 'GalSim'
        else:
            obs_ref = build_image_obs(
                ip, psf=psf, oversample=ref_osamp, int_model=model, gsparams=gsp
            )
            ref_sb = np.array(model.render_image(theta, obs=obs_ref))
            ref_label = f'oversample={ref_osamp}'

        peak = np.max(np.abs(ref_sb))
        max_fracs = []
        rms_fracs = []

        print(f'\n  cosi={cosi} (ref={ref_label}):')
        for osamp in oversamples:
            obs = build_image_obs(
                ip, psf=psf, oversample=osamp, int_model=model, gsparams=gsp
            )
            our_sb = np.array(model.render_image(theta, obs=obs))
            diff = our_sb - ref_sb
            mf = np.max(np.abs(diff)) / peak
            rf = np.sqrt(np.mean(diff**2)) / peak
            max_fracs.append(mf)
            rms_fracs.append(rf)
            print(f'    os={osamp:2d}: max={mf:.4%}, rms={rf:.4%}')

        ax.semilogy(oversamples, max_fracs, 'bo-', lw=2, ms=5, label='max |resid|/peak')
        ax.semilogy(oversamples, rms_fracs, 'rs--', lw=2, ms=5, label='rms resid/peak')
        ax.set_xlabel('Oversample factor')
        ax.set_title(col_lab, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(oversamples)
        if ci == 0:
            ax.set_ylabel('Fractional residual')

    fig.suptitle(
        f'Oversample convergence: Spergel(nu={nu}) + PSF (FWHM={fwhm}")\n'
        f'{ps}"/pix, {npix}×{npix} | face-on: vs GalSim; '
        f'inclined: self-convergence vs os={ref_osamp}',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(
        spergel_output_dir / 'oversample_convergence.png', dpi=150, bbox_inches='tight'
    )
    plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
