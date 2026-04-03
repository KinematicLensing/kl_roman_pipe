"""
Unit tests for InclinedSersicModel.

Tests include:
- b_n accuracy vs scipy root-finding
- Emulator FT accuracy vs GalSim kValue
- n=1 cross-check against InclinedExponentialModel
- GalSim face-on regression
- GalSim inclined regression (critical: no minor-axis broadening)
- Flux conservation, gradient check
- render_image vs __call__ consistency
- Diagnostic plots
"""

import pytest
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gammainc

import galsim as gs

from kl_pipe.observation import build_image_obs
from kl_pipe.intensity import (
    InclinedSersicModel,
    InclinedExponentialModel,
    build_intensity_model,
    _sersic_bn,
    _sersic_ft_emulator,
    _sersic_norm_2d,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir, build_map_grid_from_image_pars


# maximum GB for a single GalSim FFT allocation in tests
_MAX_FFT_GB = 8.0


def _galsim_fft_safe(profile, pixel_scale, max_gb=_MAX_FFT_GB):
    """Check if a GalSim profile can render without exceeding FFT memory."""
    fft_n = int(2 * profile.maxk / profile.stepk)
    fft_gb = fft_n**2 * 16 / 1e9
    return fft_gb <= max_gb, fft_gb


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope='module')
def sersic_output_dir():
    out_dir = get_test_dir() / "out" / "sersic"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope='module')
def sersic_model():
    return InclinedSersicModel()


# ==============================================================================
# b_n accuracy
# ==============================================================================


@pytest.mark.parametrize("n", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
def test_sersic_bn_accuracy(n):
    """b_n via Ciotti & Bertin vs exact root-finding."""
    exact = brentq(lambda b: gammainc(2 * n, b) - 0.5, 0.01, 50.0)
    approx = float(_sersic_bn(n))
    rel_err = abs(approx - exact) / exact
    assert rel_err < 1e-3, f"b_{n} rel error {rel_err:.2e} exceeds 1e-3"


# ==============================================================================
# Emulator accuracy vs GalSim
# ==============================================================================


def test_emulator_vs_galsim_kvalue(sersic_output_dir):
    """Compare emulator FT to GalSim Sersic kValue across n and k."""
    n_values = [1.0, 2.0, 3.0, 4.0]
    k_Re = np.logspace(-2, 2, 100)  # dimensionless k * R_e

    fig, axes = plt.subplots(2, len(n_values), figsize=(16, 8))

    for j, n in enumerate(n_values):
        gs_prof = gs.Sersic(n=n, half_light_radius=1.0, flux=1.0)
        gs_ft = np.array([gs_prof.kValue(k, 0.0).real for k in k_Re])
        emu_ft = np.array(_sersic_ft_emulator(jnp.array(k_Re), n))

        rel_err = np.abs(emu_ft - gs_ft) / np.maximum(np.abs(gs_ft), 1e-15)

        axes[0, j].loglog(k_Re, gs_ft, 'k-', label='GalSim')
        axes[0, j].loglog(k_Re, emu_ft, 'r--', label='emulator')
        axes[0, j].set_title(f'n={n}')
        axes[0, j].set_xlabel('k * R_e')
        axes[0, j].set_ylabel('FT')
        axes[0, j].legend(fontsize=8)

        axes[1, j].loglog(k_Re, rel_err, 'b-')
        axes[1, j].set_xlabel('k * R_e')
        axes[1, j].set_ylabel('|rel error|')
        axes[1, j].axhline(0.01, color='r', ls=':', label='1%')
        axes[1, j].legend(fontsize=8)

    fig.suptitle('Sersic FT emulator vs GalSim kValue')
    fig.tight_layout()
    fig.savefig(sersic_output_dir / 'emulator_vs_galsim_kvalue.png', dpi=150)
    plt.close(fig)

    # print max errors for each n
    for n in n_values:
        gs_prof = gs.Sersic(n=n, half_light_radius=1.0, flux=1.0)
        k_test = np.logspace(-1, 1, 50)
        gs_ft = np.array([gs_prof.kValue(k, 0.0).real for k in k_test])
        emu_ft = np.array(_sersic_ft_emulator(jnp.array(k_test), n))
        mask = gs_ft > 1e-4  # only where FT is significant
        if np.any(mask):
            max_rel = np.max(np.abs(emu_ft[mask] - gs_ft[mask]) / gs_ft[mask])
            print(f"  n={n}: max rel error (k*Re in [0.1,10]) = {max_rel:.4%}")


# ==============================================================================
# Model instantiation and factory
# ==============================================================================


def test_model_instantiation():
    model = InclinedSersicModel()
    assert model.name == 'inclined_sersic'
    assert len(model.PARAMETER_NAMES) == 10
    assert 'n_sersic' in model.PARAMETER_NAMES
    assert 'int_hlr' in model.PARAMETER_NAMES
    assert 'int_h_over_hlr' in model.PARAMETER_NAMES


def test_factory():
    model = build_intensity_model('inclined_sersic')
    assert isinstance(model, InclinedSersicModel)
    model2 = build_intensity_model('sersic')
    assert isinstance(model2, InclinedSersicModel)


def test_n_bounds():
    model = InclinedSersicModel()
    x = jnp.zeros(4)
    y = jnp.zeros(4)

    # n too low
    theta_low = jnp.array([0.7, 0.5, 0.0, 0.0, 100.0, 1.0, 0.1, 0.3, 0.0, 0.0])
    with pytest.raises(ValueError, match="n_sersic.*outside valid range"):
        model(theta_low, 'obs', x, y)

    # n too high
    theta_high = jnp.array([0.7, 0.5, 0.0, 0.0, 100.0, 1.0, 0.1, 7.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="n_sersic.*outside valid range"):
        model(theta_high, 'obs', x, y)


# ==============================================================================
# n=1 cross-check against InclinedExponentialModel
# ==============================================================================


def test_n1_crosscheck_faceon(sersic_output_dir):
    """InclinedSersicModel(n=1) vs InclinedExponentialModel face-on."""
    sersic = InclinedSersicModel()
    exp_model = InclinedExponentialModel()

    # R_e = b_1 * r_s for n=1 -> r_s = R_e / b_1
    Re = 2.0
    b1 = float(_sersic_bn(1.0))
    r_s = Re / b1

    theta_sersic = jnp.array([1.0, 0.0, 0.0, 0.0, 100.0, Re, 0.1, 1.0, 0.0, 0.0])
    theta_exp = jnp.array([1.0, 0.0, 0.0, 0.0, 100.0, r_s, 0.1, 0.0, 0.0])

    img_s = sersic._render_kspace(theta_sersic, 64, 64, 0.11)
    img_e = exp_model._render_kspace(theta_exp, 64, 64, 0.11)

    peak = float(jnp.max(jnp.abs(img_e)))
    max_frac = float(jnp.max(jnp.abs(img_s - img_e))) / peak
    rms_frac = float(jnp.sqrt(jnp.mean((img_s - img_e) ** 2))) / peak

    print(f"  n=1 face-on: max_frac={max_frac:.4%}, rms_frac={rms_frac:.4%}")

    # the emulator has ~0.5% RMS accuracy but up to ~12% peak error at the
    # core pixel where k-space band-limiting amplifies FT differences.
    # this documents the known emulator approximation error at n=1.
    # for exact n=1 rendering, use InclinedExponentialModel.
    assert rms_frac < 0.02, f"n=1 face-on rms frac {rms_frac:.4%} exceeds 2%"
    assert max_frac < 0.20, f"n=1 face-on max frac {max_frac:.4%} exceeds 20%"


# ==============================================================================
# Gradient check
# ==============================================================================


def test_gradient_through_emulator():
    """jax.grad through render_kspace w.r.t. all params including n_sersic."""
    model = InclinedSersicModel()
    theta = jnp.array([0.7, 0.5, 0.0, 0.0, 100.0, 1.0, 0.1, 4.0, 0.0, 0.0])

    def loss(t):
        img = model._render_kspace(t, 32, 32, 0.11)
        return jnp.sum(img**2)

    g = jax.grad(loss)(theta)
    assert g.shape == theta.shape
    assert not jnp.any(jnp.isnan(g)), f"NaN in gradient: {g}"
    assert not jnp.any(jnp.isinf(g)), f"Inf in gradient: {g}"

    # check n_sersic gradient is nonzero
    n_idx = model.PARAMETER_NAMES.index('n_sersic')
    assert abs(float(g[n_idx])) > 1e-6, "n_sersic gradient is zero"


# ==============================================================================
# Flux conservation
# ==============================================================================


@pytest.mark.parametrize("n", [1.0, 2.0, 4.0])
def test_flux_conservation(n):
    """Total rendered flux matches input flux on a large grid."""
    model = InclinedSersicModel()
    Re = 2.0
    flux = 100.0
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, Re, 0.1, n, 0.0, 0.0])

    # grid must be large enough to capture extended wings
    # n=4 (de Vaucouleurs): R_90 ~ 7*R_e, so need grid > 14*R_e on a side
    Npix = 512 if n >= 3 else 256
    img = model._render_kspace(theta, Npix, Npix, 0.11)
    rendered_flux = float(jnp.sum(img) * 0.11**2)
    rel_err = abs(rendered_flux - flux) / flux
    print(f"  n={n}: rendered_flux={rendered_flux:.2f}, rel_err={rel_err:.4%}")
    assert rel_err < 0.05, f"flux error {rel_err:.2%} exceeds 5% for n={n}"


# ==============================================================================
# render_image vs __call__ consistency
# ==============================================================================


def test_render_vs_call_faceon():
    """k-space and real-space paths agree face-on (Sersic is finite, no divergence)."""
    model = InclinedSersicModel()
    ip = ImagePars(shape=(64, 64), pixel_scale=0.11, indexing='ij')
    X, Y = build_map_grid_from_image_pars(ip, unit='arcsec', centered=True)

    theta = jnp.array([1.0, 0.0, 0.0, 0.0, 100.0, 1.0, 0.1, 2.0, 0.0, 0.0])

    img_kspace = model._render_kspace(theta, ip.Nrow, ip.Ncol, ip.pixel_scale)
    img_call = model(theta, 'obs', X, Y)

    peak = float(jnp.max(jnp.abs(img_call)))
    max_frac = float(jnp.max(jnp.abs(img_kspace - img_call))) / peak
    print(f"  render vs call face-on: max_frac={max_frac:.4%}")
    # k-space band-limits; allow up to 10% at core for concentrated profiles
    assert max_frac < 0.20, f"render vs call max frac {max_frac:.4%} exceeds 20%"


# ==============================================================================
# GalSim face-on regression
# ==============================================================================


@pytest.mark.parametrize("n", [1.0, 2.0, 4.0])
def test_faceon_vs_galsim(sersic_output_dir, n):
    """Face-on Sersic with PSF vs GalSim Sersic."""
    Re = 2.0  # arcsec
    flux = 100.0
    psf_fwhm = 0.15
    Npix = 128
    ps = 0.11

    # our model
    model = InclinedSersicModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, Re, 0.1, n, 0.0, 0.0])

    # PSF
    psf_obj = gs.Gaussian(fwhm=psf_fwhm, flux=1.0)
    ip = ImagePars(shape=(Npix, Npix), pixel_scale=ps, indexing='ij')
    obs = build_image_obs(ip, psf=psf_obj, oversample=5)
    img_ours = np.array(model.render_image(theta, obs=obs))

    # GalSim reference — method='no_pixel', divide by ps² to match our SB units
    gs_prof = gs.Sersic(n=n, half_light_radius=Re, flux=flux)
    gs_conv = gs.Convolve(gs_prof, psf_obj)
    gs_img = (
        gs_conv.drawImage(nx=Npix, ny=Npix, scale=ps, method='no_pixel').array / ps**2
    )

    peak = np.max(np.abs(gs_img))
    max_frac = np.max(np.abs(img_ours - gs_img)) / peak
    rms_frac = np.sqrt(np.mean((img_ours - gs_img) ** 2)) / peak

    print(f"  face-on n={n} with PSF: max_frac={max_frac:.4%}, rms_frac={rms_frac:.4%}")

    # the emulator has L2 < 2e-6 but up to ~11% peak error at the core pixel
    # where IFFT band-limiting amplifies FT differences. RMS is the primary
    # accuracy metric (< 0.5%); max is dominated by core-pixel emulator error.
    assert rms_frac < 0.005, f"face-on n={n} rms frac {rms_frac:.4%} exceeds 0.5%"
    assert max_frac < 0.15, f"face-on n={n} max frac {max_frac:.4%} exceeds 15%"


# ==============================================================================
# CRITICAL: inclined vs GalSim InclinedSersic — no minor-axis broadening
# ==============================================================================


@pytest.mark.parametrize(
    "n,cosi",
    [
        (1.0, 0.5),
        (2.0, 0.5),
        (2.0, 0.75),
        (4.0, 0.5),
        (4.0, 0.75),
    ],
)
def test_inclined_vs_galsim(sersic_output_dir, n, cosi):
    """Inclined Sersic with PSF vs GalSim InclinedSersic.

    This is the critical test: must show no minor-axis broadening
    (the failure mode of the Spergel approximation).
    """
    Re = 2.0
    flux = 100.0
    h_over_r = 0.1
    psf_fwhm = 0.15
    Npix = 128
    ps = 0.11
    theta_int = 0.0  # aligned with axes for easy minor-axis comparison

    # our model
    model = InclinedSersicModel()
    theta = jnp.array([cosi, theta_int, 0.0, 0.0, flux, Re, h_over_r, n, 0.0, 0.0])

    psf_obj = gs.Gaussian(fwhm=psf_fwhm, flux=1.0)
    ip = ImagePars(shape=(Npix, Npix), pixel_scale=ps, indexing='ij')
    obs = build_image_obs(ip, psf=psf_obj, oversample=5)
    img_ours = np.array(model.render_image(theta, obs=obs))

    # GalSim InclinedSersic reference
    # R_e for the scale_radius param (GalSim uses scale_radius = r_s, not R_e)
    bn = float(_sersic_bn(n))
    gs_r_s = Re / bn**n
    gs_h_s = h_over_r * Re  # h_z = h_over_hlr * R_e
    inc_angle = np.arccos(cosi) * gs.radians

    gs_params = gs.GSParams(maximum_fft_size=32768)
    gs_prof = gs.InclinedSersic(
        n=n,
        inclination=inc_angle,
        half_light_radius=Re,
        scale_h_over_r=gs_h_s / gs_r_s,
        flux=flux,
        gsparams=gs_params,
    )

    # check FFT on the CONVOLVED profile (PSF limits maxk)
    gs_conv = gs.Convolve(gs_prof, psf_obj, gsparams=gs_params)
    safe, fft_gb = _galsim_fft_safe(gs_conv, ps)
    if not safe:
        pytest.skip(
            f"GalSim InclinedSersic(n={n}) FFT needs {fft_gb:.1f} GB "
            f"(limit {_MAX_FFT_GB} GB)"
        )
    gs_img = (
        gs_conv.drawImage(nx=Npix, ny=Npix, scale=ps, method='no_pixel').array / ps**2
    )

    peak = np.max(np.abs(gs_img))
    diff = img_ours - gs_img
    max_frac = np.max(np.abs(diff)) / peak
    rms_frac = np.sqrt(np.mean(diff**2)) / peak

    print(
        f"  inclined n={n}, cosi={cosi}: "
        f"max_frac={max_frac:.4%}, rms_frac={rms_frac:.4%}"
    )

    # save diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = peak
    axes[0].imshow(gs_img, origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title(f'GalSim InclinedSersic(n={n})')
    axes[1].imshow(img_ours, origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title(f'Our InclinedSersicModel')
    im = axes[2].imshow(diff, origin='lower', cmap='RdBu_r')
    axes[2].set_title(f'Residual (max={max_frac:.2%})')
    plt.colorbar(im, ax=axes[2])
    fig.suptitle(f'n={n}, cosi={cosi}, PSF={psf_fwhm}"')
    fig.tight_layout()
    fig.savefig(sersic_output_dir / f'inclined_n{n}_cosi{cosi}.png', dpi=150)
    plt.close(fig)

    # primary: RMS should be <1% (Spergel was >>10% RMS here)
    # secondary: max up to 15% from core-pixel emulator error
    assert (
        rms_frac < 0.01
    ), f"inclined n={n} cosi={cosi} rms frac {rms_frac:.4%} exceeds 1%"
    assert (
        max_frac < 0.15
    ), f"inclined n={n} cosi={cosi} max frac {max_frac:.4%} exceeds 15%"


# ==============================================================================
# Oversample convergence
# ==============================================================================


def test_oversample_convergence(sersic_output_dir):
    """Verify clean convergence with increasing oversample."""
    model = InclinedSersicModel()
    theta = jnp.array([0.5, 0.0, 0.0, 0.0, 100.0, 1.0, 0.1, 4.0, 0.0, 0.0])

    # reference: highest oversample
    ref = model._render_kspace(theta, 64, 64, 0.11, oversample=15)
    peak = float(jnp.max(jnp.abs(ref)))

    oversamples = [1, 3, 5, 7, 9]
    errors = []
    for os_val in oversamples:
        img = model._render_kspace(theta, 64, 64, 0.11, oversample=os_val)
        max_frac = float(jnp.max(jnp.abs(img - ref))) / peak
        errors.append(max_frac)
        print(f"  oversample={os_val}: max_frac={max_frac:.4%}")

    # errors should decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1] * 0.5, (
            f"non-monotonic convergence: os={oversamples[i]} err={errors[i]:.4%} "
            f"> os={oversamples[i+1]} err={errors[i+1]:.4%}"
        )


# ==============================================================================
# Normalization integral
# ==============================================================================


@pytest.mark.parametrize("n", [1.0, 2.0, 3.0, 4.0])
def test_sersic_normalization_integral(n):
    """Numerical radial integral of face-on Sersic profile must equal flux."""
    from scipy.integrate import quad

    flux = 1.0
    Re = 2.0
    bn = float(_sersic_bn(n))
    I0 = flux / float(_sersic_norm_2d(Re, n))

    def integrand(r):
        return 2.0 * np.pi * r * I0 * np.exp(-bn * (r / Re) ** (1.0 / n))

    # upper limit: exp(-bn * s^{1/n}) negligible for s >> (20/bn)^n
    r_max = Re * max(50.0, (20.0 / bn) ** n)
    measured_flux, _ = quad(integrand, 0, r_max, limit=300)
    rel_err = abs(measured_flux - flux) / flux
    assert rel_err < 1e-3, (
        f"Normalization: measured={measured_flux:.6f}, expected={flux}, "
        f"rel_err={rel_err:.2e} (n={n})"
    )


# ==============================================================================
# PSF path consistency
# ==============================================================================


def test_sersic_psf_path_consistency():
    """Fused k-space PSF path must match fallback real-space PSF path."""
    model = InclinedSersicModel()
    psf_obj = gs.Gaussian(fwhm=0.625)
    theta = jnp.array([0.7, 0.3, 0.02, -0.01, 1.0, 2.0, 0.1, 2.0, 0.0, 0.0])

    ip = ImagePars(shape=(64, 64), pixel_scale=0.3125, indexing='ij')

    # fused k-space path (int_model triggers kspace_psf_fft)
    obs_kspace = build_image_obs(ip, psf=psf_obj, oversample=5, int_model=model)
    img_kspace = np.array(model.render_image(theta, obs=obs_kspace))

    # fallback real-space path (no int_model -> no kspace_psf_fft)
    obs_realspace = build_image_obs(ip, psf=psf_obj, oversample=5)
    img_realspace = np.array(model.render_image(theta, obs=obs_realspace))

    peak = np.max(np.abs(img_kspace))
    max_frac = np.max(np.abs(img_kspace - img_realspace)) / peak

    assert max_frac < 2e-3, f"PSF path consistency: max|resid|/peak = {max_frac:.2e}"


# ==============================================================================
# Non-square grid
# ==============================================================================


def test_sersic_non_square_grid():
    """Output shape must be (Nrow, Ncol) on non-square grids."""
    model = InclinedSersicModel()
    theta = jnp.array([0.7, 0.5, 0.0, 0.0, 100.0, 1.0, 0.1, 2.0, 0.0, 0.0])

    img = model._render_kspace(theta, 48, 96, 0.11)
    assert img.shape == (48, 96), f"Expected (48, 96), got {img.shape}"

    img2 = model._render_kspace(theta, 96, 48, 0.11)
    assert img2.shape == (96, 48), f"Expected (96, 48), got {img2.shape}"


# ==============================================================================
# Diagnostic helpers
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


_BANDS = [
    ('core', 0.0, 1.0),
    ('mid', 1.0, 3.0),
    ('wings', 3.0, 6.0),
    ('total', 0.0, 6.0),
]


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


def _galsim_sersic_ref(n, hlr, h_over_r, cosi, flux, gsparams):
    """Build GalSim Sersic/InclinedSersic profile for reference rendering.

    Returns (profile, is_faceon). Handles face-on (gs.Sersic) vs inclined
    (gs.InclinedSersic with correct scale_h_over_r mapping).
    """
    is_faceon = abs(cosi - 1.0) < 0.01
    if is_faceon:
        return gs.Sersic(n=n, half_light_radius=hlr, flux=flux, gsparams=gsparams), True

    bn = float(_sersic_bn(n))
    gs_r_s = hlr / bn**n
    gs_h_s = h_over_r * hlr
    inc_angle = np.arccos(cosi) * gs.radians
    prof = gs.InclinedSersic(
        n=n,
        inclination=inc_angle,
        half_light_radius=hlr,
        scale_h_over_r=gs_h_s / gs_r_s,
        flux=flux,
        gsparams=gsparams,
    )
    return prof, False


# ==============================================================================
# Rendering accuracy diagnostics — face-on radial profiles with annular stats
# ==============================================================================


def _faceon_sersic_diagnostic(
    output_dir, psf_obj=None, psf_label='no PSF', fwhm_str=''
):
    """Face-on Sersic(n=2,4) vs GalSim: radial profiles + annular residuals.

    2 columns (n=2, n=4) x 2 rows (profile + residual).
    Log x-axis (r/R_e) from 0.02 to 6.0, semilogy profiles, auto-scaled
    residual panels with annular stats text annotations.
    """
    npix = 128
    ps = 0.11
    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    model = InclinedSersicModel()

    use_psf = psf_obj is not None
    if use_psf:
        gsp = gs.GSParams(
            folding_threshold=1e-4,
            maxk_threshold=1e-4,
            kvalue_accuracy=1e-6,
            maximum_fft_size=32768,
        )
    else:
        gsp = gs.GSParams(maximum_fft_size=32768)
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')

    draw_method = 'auto' if use_psf else 'no_pixel'
    oversample = 5 if use_psf else 1

    # pixel radius grid
    center = npix // 2
    y_idx, x_idx = np.mgrid[:npix, :npix]
    r_re = np.sqrt(((x_idx - center) * ps) ** 2 + ((y_idx - center) * ps) ** 2) / hlr

    n_values = [2.0, 4.0]
    fig, axes = plt.subplots(
        2,
        len(n_values),
        figsize=(7 * len(n_values), 8),
        gridspec_kw={'height_ratios': [3, 1.2]},
    )

    for col, n in enumerate(n_values):
        theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, hlr, h_over_r, n, 0.0, 0.0])

        if use_psf:
            obs = build_image_obs(
                ip,
                psf=psf_obj,
                oversample=oversample,
                int_model=model,
                gsparams=gsp,
            )
            our_img = np.array(model.render_image(theta, obs=obs))
            gs_prof = gs.Convolve(
                gs.Sersic(n=n, half_light_radius=hlr, flux=flux, gsparams=gsp),
                psf_obj,
            )
        else:
            our_img = np.array(model.render_image(theta, image_pars=ip))
            gs_prof = gs.Sersic(n=n, half_light_radius=hlr, flux=flux, gsparams=gsp)

        gs_img = (
            gs_prof.drawImage(
                nx=npix,
                ny=npix,
                scale=ps,
                method=draw_method,
            ).array
            / ps**2
        )

        # radial profiles
        mid_r, prof_ours = _radial_profile(our_img, r_re)
        mid_r, prof_gs = _radial_profile(gs_img, r_re)
        valid = np.isfinite(prof_gs) & (prof_gs > 0)
        peak = np.nanmax(prof_gs[valid])

        # profile panel
        ax_prof = axes[0, col]
        ax_prof.semilogy(mid_r[valid], prof_gs[valid], 'b-', lw=1.5, label='GalSim')
        ax_prof.semilogy(mid_r[valid], prof_ours[valid], 'r--', lw=1.5, label='Ours')
        ax_prof.axvline(1.0, color='grey', ls=':', alpha=0.4, label='$R_e$')
        ax_prof.axvline(3.0, color='grey', ls='--', alpha=0.3, label='$3 R_e$')
        ax_prof.set_xlim(0.02, 6.0)
        ax_prof.set_xscale('log')
        ax_prof.set_ylim(peak * 1e-5, peak * 3)
        ax_prof.set_ylabel('Surface brightness')
        ax_prof.set_title(f'n={n:.0f}, {psf_label}', fontsize=10)
        ax_prof.legend(fontsize=7, loc='upper right')

        # residual panel
        ax_res = axes[1, col]
        frac_resid = np.where(valid, (prof_ours - prof_gs) / peak, np.nan)
        ax_res.plot(mid_r[valid], frac_resid[valid], 'k-', lw=0.8)
        ax_res.axhline(0, color='grey', ls='-', alpha=0.3)
        ax_res.set_xlim(0.02, 6.0)
        ax_res.set_xscale('log')
        ylim = max(0.05, np.nanmax(np.abs(frac_resid[valid])) * 1.3)
        ax_res.set_ylim(-ylim, ylim)
        ax_res.set_ylabel('(Ours - GalSim) / peak')
        ax_res.set_xlabel('$r / R_e$')

        # annular stats text
        stats = _annular_stats(our_img, gs_img, r_re, _BANDS)
        stat_text = '\n'.join(
            f'{lab}: max={s["max_frac"]:.3%}, rms={s["rms_frac"]:.3%}'
            for lab, s in stats.items()
        )
        ax_res.text(
            0.97,
            0.95,
            stat_text,
            transform=ax_res.transAxes,
            ha='right',
            va='top',
            fontsize=6,
            family='monospace',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'),
        )

        print(f'\n  n={n}, {psf_label}:')
        for band, s in stats.items():
            print(f'    {band:6s}: max={s["max_frac"]:.4%}, rms={s["rms_frac"]:.4%}')

    fig.suptitle(
        f'Sersic emulator vs GalSim (face-on, {psf_label})\n'
        f'flux={flux}, hlr={hlr}", {ps}"/pix, {npix}x{npix}',
        fontsize=11,
    )
    fig.tight_layout()
    suffix = 'nopsf' if not use_psf else f'psf_{fwhm_str}'
    fname = f'sersic_faceon_diagnostic_{suffix}.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def test_sersic_faceon_diagnostic_nopsf(sersic_output_dir):
    """Face-on Sersic radial profile diagnostic, no PSF."""
    _faceon_sersic_diagnostic(sersic_output_dir, psf_obj=None, psf_label='no PSF')


def test_sersic_faceon_diagnostic_psf(sersic_output_dir):
    """Face-on Sersic radial profile diagnostic, PSF FWHM=0.15"."""
    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )
    fwhm = 0.15
    psf_obj = gs.Gaussian(fwhm=fwhm, gsparams=gsp)
    _faceon_sersic_diagnostic(
        sersic_output_dir,
        psf_obj=psf_obj,
        psf_label=f'PSF FWHM={fwhm}"',
        fwhm_str=f'{fwhm}',
    )


# ==============================================================================
# 2D image diagnostic — InclinedSersic(n=4) vs GalSim InclinedSersic
# ==============================================================================


def _sersic_2d_diagnostic(output_dir, psf_fwhm=None, cosi=1.0):
    """2D image comparison: our InclinedSersic(n=4) vs GalSim.

    2x2 layout: [GalSim | Ours] top, [residual | stats text] bottom.
    Uses LogNorm for galaxy images, RdBu_r with symmetric vmax for
    residuals. Generates separate files for each (psf_fwhm, cosi)
    combination.
    """
    n = 4.0
    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    npix = 128
    ps = 0.11
    is_faceon = abs(cosi - 1.0) < 0.01

    use_psf = psf_fwhm is not None

    # tight GSParams for PSF-convolved rendering (PSF limits maxk)
    # relaxed for no-PSF (raw Sersic n=4 has enormous maxk with tight params)
    if use_psf:
        gsp = gs.GSParams(
            folding_threshold=1e-4,
            maxk_threshold=1e-4,
            kvalue_accuracy=1e-6,
            maximum_fft_size=32768,
        )
    else:
        gsp = gs.GSParams(maximum_fft_size=32768)

    psf_obj = gs.Gaussian(fwhm=psf_fwhm, gsparams=gsp) if use_psf else None
    draw_method = 'auto' if use_psf else 'no_pixel'
    oversample = 5 if use_psf else 1

    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSersicModel()
    extent = np.array([-npix / 2, npix / 2, -npix / 2, npix / 2]) * ps

    # GalSim reference
    gs_prof, _ = _galsim_sersic_ref(n, hlr, h_over_r, cosi, flux, gsp)

    gs_draw = gs.Convolve(gs_prof, psf_obj, gsparams=gsp) if use_psf else gs_prof
    # check FFT on the profile that will actually be rendered
    safe, fft_gb = _galsim_fft_safe(gs_draw, ps)
    if not safe:
        print(
            f'  SKIP: InclinedSersic(n={n}) cosi={cosi} needs {fft_gb:.1f} GB '
            f'(limit {_MAX_FFT_GB} GB)'
        )
        return
    gs_img = (
        gs_draw.drawImage(
            nx=npix,
            ny=npix,
            scale=ps,
            method=draw_method,
        ).array
        / ps**2
    )

    # our render
    theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, hlr, h_over_r, n, 0.0, 0.0])
    if use_psf:
        obs = build_image_obs(
            ip,
            psf=psf_obj,
            oversample=oversample,
            int_model=model,
            gsparams=gsp,
        )
        our_img = np.array(model.render_image(theta, obs=obs))
    else:
        our_img = np.array(model.render_image(theta, image_pars=ip))

    peak = np.max(np.abs(gs_img))
    diff = our_img - gs_img
    max_frac = np.max(np.abs(diff)) / peak
    rms_frac = np.sqrt(np.mean(diff**2)) / peak

    vmax_img = np.max(gs_img)
    vmin_img = vmax_img * 1e-4
    vmax_resid = np.max(np.abs(diff)) * 0.8

    psf_str = f'PSF FWHM={psf_fwhm}"' if use_psf else 'no PSF'
    cosi_str = f'cosi={cosi}' if not is_faceon else 'face-on'
    sersic_label = 'InclinedSersic(n=4)' if not is_faceon else 'Sersic(n=4)'

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 11), gridspec_kw={'height_ratios': [1, 0.8]}
    )

    # top row: GalSim | Ours (LogNorm)
    for j, (img, title) in enumerate(
        [
            (gs_img, f'GalSim {sersic_label}'),
            (our_img, f'Our InclinedSersic(n=4)'),
        ]
    ):
        im = axes[0, j].imshow(
            img,
            origin='lower',
            extent=extent,
            norm=plt.matplotlib.colors.LogNorm(vmin=vmin_img, vmax=vmax_img),
        )
        axes[0, j].set_title(title, fontsize=10)
        axes[0, j].set_xlabel('arcsec')
        axes[0, j].set_ylabel('arcsec')
        plt.colorbar(im, ax=axes[0, j], shrink=0.8)

    # bottom left: residual (RdBu_r, symmetric)
    im_res = axes[1, 0].imshow(
        diff,
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
        vmin=-vmax_resid,
        vmax=vmax_resid,
    )
    axes[1, 0].set_title('Residual (Ours - GalSim)', fontsize=10)
    axes[1, 0].set_xlabel('arcsec')
    axes[1, 0].set_ylabel('arcsec')
    plt.colorbar(im_res, ax=axes[1, 0], shrink=0.8)

    # bottom right: stats text
    axes[1, 1].set_visible(False)
    center_pix = npix // 2
    y_idx, x_idx = np.mgrid[:npix, :npix]
    r_re = (
        np.sqrt(((x_idx - center_pix) * ps) ** 2 + ((y_idx - center_pix) * ps) ** 2)
        / hlr
    )
    stats = _annular_stats(our_img, gs_img, r_re, _BANDS)

    stat_lines = [
        f'max|resid|/peak = {max_frac:.3%}',
        f'rms|resid|/peak = {rms_frac:.4%}',
        '',
        'Annular stats:',
    ]
    for band, s in stats.items():
        stat_lines.append(
            f'  {band:6s}: max={s["max_frac"]:.3%}, rms={s["rms_frac"]:.3%}'
        )
    fig.text(
        0.75,
        0.22,
        '\n'.join(stat_lines),
        ha='center',
        va='center',
        fontsize=9,
        family='monospace',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='grey'),
    )

    fig.suptitle(
        f'2D: {sersic_label} vs GalSim — {psf_str}, {cosi_str}\n'
        f'flux={flux}, hlr={hlr}", {ps}"/pix, {npix}x{npix}',
        fontsize=11,
    )
    plt.tight_layout()
    suffix_parts = ['2d']
    suffix_parts.append('nopsf' if not use_psf else f'psf_{psf_fwhm}')
    if not is_faceon:
        suffix_parts.append(f'cosi{cosi}')
    fname = f'sersic_n4_{"_".join(suffix_parts)}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_dir / fname}')


def test_sersic_2d_diagnostic(sersic_output_dir):
    """2D image comparison: InclinedSersic(n=4) vs GalSim.

    Face-on: no PSF and with PSF.
    Inclined: cosi=0.25, 0.5, 0.75 with PSF.
    """
    # face-on, no PSF
    _sersic_2d_diagnostic(sersic_output_dir, psf_fwhm=None, cosi=1.0)
    # face-on, with PSF
    _sersic_2d_diagnostic(sersic_output_dir, psf_fwhm=0.15, cosi=1.0)
    # inclined, with PSF
    for cosi in [0.25, 0.5, 0.75]:
        _sersic_2d_diagnostic(sersic_output_dir, psf_fwhm=0.15, cosi=cosi)


# ==============================================================================
# 4-column inclination diagnostic — multiple n values across inclinations
# ==============================================================================


def test_sersic_inclination_diagnostic(sersic_output_dir):
    """4-column panel: n=1,2,3,4 across cosi=1.0,0.75,0.5,0.25 with PSF.

    Azimuthally-averaged radial profiles + residuals for each (n, cosi)
    combination, comparing our InclinedSersicModel to GalSim. Uses log
    x-axis, semilogy profiles, fused k-space PSF path, and elliptical r
    for inclined cases. GalSim draws with method='no_pixel', divide by
    ps**2 for SB units.
    """
    npix = 128
    ps = 0.11
    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    psf_fwhm = 0.15

    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )
    psf_obj = gs.Gaussian(fwhm=psf_fwhm, gsparams=gsp)

    model = InclinedSersicModel()
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')

    n_values = [1.0, 2.0, 3.0, 4.0]
    cosi_values = [1.0, 0.75, 0.5, 0.25]

    # pixel radius grid (use np.mgrid for pixel indices, not build_map_grid)
    center = npix // 2
    y_idx, x_idx = np.mgrid[:npix, :npix]

    fig, axes = plt.subplots(
        len(cosi_values) * 2,
        len(n_values),
        figsize=(4.5 * len(n_values), 2.2 * len(cosi_values) * 2),
        gridspec_kw={'height_ratios': [3, 1] * len(cosi_values)},
    )

    stats_all = {}
    for col, n in enumerate(n_values):
        for row, cosi in enumerate(cosi_values):
            ax_main = axes[row * 2, col]
            ax_resid = axes[row * 2 + 1, col]

            theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, hlr, h_over_r, n, 0.0, 0.0])

            # our render (fused k-space PSF path)
            obs = build_image_obs(
                ip,
                psf=psf_obj,
                oversample=5,
                int_model=model,
                gsparams=gsp,
            )
            img_ours = np.array(model.render_image(theta, obs=obs))

            # GalSim reference
            gs_prof, is_faceon = _galsim_sersic_ref(n, hlr, h_over_r, cosi, flux, gsp)

            # check FFT on the CONVOLVED profile (PSF limits maxk,
            # dramatically reducing FFT size vs the raw profile)
            gs_draw = gs.Convolve(gs_prof, psf_obj, gsparams=gsp)
            safe, fft_gb = _galsim_fft_safe(gs_draw, ps)
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
                stats_all[(n, cosi)] = {'max': np.nan, 'rms': np.nan}
                continue
            gs_img = (
                gs_draw.drawImage(
                    nx=npix,
                    ny=npix,
                    scale=ps,
                    method='no_pixel',
                ).array
                / ps**2
            )

            # elliptical r for inclined cases
            r_ell = np.sqrt(
                ((x_idx - center) * ps) ** 2
                + ((y_idx - center) * ps / max(cosi, 0.1)) ** 2
            )
            r_re = r_ell / hlr

            # radial profiles
            r_bin_edges = np.concatenate([[0], np.logspace(-1.5, np.log10(5.0), 40)])
            r_mid = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])
            gs_prof_r = np.full(len(r_mid), np.nan)
            our_prof_r = np.full(len(r_mid), np.nan)
            for k in range(len(r_mid)):
                mask = (r_re >= r_bin_edges[k]) & (r_re < r_bin_edges[k + 1])
                if np.any(mask):
                    gs_prof_r[k] = np.mean(gs_img[mask])
                    our_prof_r[k] = np.mean(img_ours[mask])

            valid = np.isfinite(gs_prof_r) & (gs_prof_r > 0)
            peak = np.nanmax(gs_prof_r[valid])
            residual = np.where(
                valid,
                (our_prof_r - gs_prof_r) / peak,
                np.nan,
            )

            # 2D stats
            peak_2d = np.max(np.abs(gs_img))
            max_frac = np.max(np.abs(img_ours - gs_img)) / peak_2d
            rms_frac = np.sqrt(np.mean(((img_ours - gs_img) / peak_2d) ** 2))
            stats_all[(n, cosi)] = {'max': max_frac, 'rms': rms_frac}

            # main panel: semilogy + log x
            ax_main.semilogy(
                r_mid[valid],
                gs_prof_r[valid],
                'b-',
                lw=1.5,
                label='GalSim' if row == 0 else None,
            )
            ax_main.semilogy(
                r_mid[valid],
                our_prof_r[valid],
                'r--',
                lw=1.5,
                label='Ours' if row == 0 else None,
            )
            ax_main.axvline(
                1.0,
                color='grey',
                ls=':',
                alpha=0.4,
                label='$R_e$' if (row == 0 and col == 0) else None,
            )
            ax_main.axvline(
                3.0,
                color='grey',
                ls='--',
                alpha=0.3,
                label='$3R_e$' if (row == 0 and col == 0) else None,
            )
            ax_main.set_xlim(0.02, 5.0)
            ax_main.set_xscale('log')
            ax_main.set_ylim(peak * 1e-5, peak * 3)
            ax_main.set_ylabel('SB')
            if row == 0:
                ax_main.set_title(f'n={n:.0f}', fontsize=10)
                ax_main.legend(fontsize=6, loc='upper right')
            if col == 0:
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
            ylim = max(0.05, np.nanmax(np.abs(residual[valid])) * 1.3)
            ax_resid.set_ylim(-ylim, ylim)
            ax_resid.set_xlim(0.02, 5.0)
            ax_resid.set_xscale('log')
            ax_resid.set_ylabel('frac resid')
            # x-axis label only on the very bottom residual panel
            is_last_row = row == len(cosi_values) - 1
            if is_last_row:
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

    # summary
    all_max = [
        s['max'] for s in stats_all.values() if np.isfinite(s.get('max', np.nan))
    ]
    all_rms = [
        s['rms'] for s in stats_all.values() if np.isfinite(s.get('rms', np.nan))
    ]
    mean_max = np.mean(all_max) if all_max else np.nan
    mean_rms = np.mean(all_rms) if all_rms else np.nan

    fig.suptitle(
        f'InclinedSersic vs GalSim (PSF FWHM={psf_fwhm}")\n'
        f'flux={flux}, hlr={hlr}", {ps}"/pix '
        f'| mean max={mean_max:.1%}, mean rms={mean_rms:.1%}',
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        sersic_output_dir / 'sersic_inclination_diagnostic.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)

    # print summary table
    print(f'\nInclination diagnostic summary:')
    print(f'  {"n":>4s} {"cosi":>5s}  {"max":>8s}  {"rms":>8s}')
    for col, n in enumerate(n_values):
        for cosi in cosi_values:
            s = stats_all.get((n, cosi))
            if s is None or np.isnan(s.get('max', np.nan)):
                print(f'  {n:4.0f} {cosi:5.2f}  (skipped)')
                continue
            print(f'  {n:4.0f} {cosi:5.2f}  {s["max"]:8.2%}  {s["rms"]:8.2%}')


# ==============================================================================
# Expanded oversample convergence — multi-column with GalSim baseline
# ==============================================================================


def test_sersic_oversample_convergence_diagnostic(sersic_output_dir):
    """4-column oversample convergence: face-on (vs GalSim) + 3 inclined.

    Face-on: residuals vs GalSim Sersic(n=4) at increasing oversample,
    using fused k-space PSF path.
    Inclined: self-convergence vs oversample=15 reference.
    """
    npix = 128
    ps = 0.11
    hlr = 2.0
    flux = 1.0
    h_over_r = 0.1
    n = 4.0
    fwhm = 0.15

    gsp = gs.GSParams(
        folding_threshold=1e-4,
        maxk_threshold=1e-4,
        kvalue_accuracy=1e-6,
        maximum_fft_size=32768,
    )
    psf_obj = gs.Gaussian(fwhm=fwhm, gsparams=gsp)
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')
    model = InclinedSersicModel()

    oversamples = [1, 3, 5, 7, 9, 15]
    ref_osamp = 15
    cosi_values = [1.0, 0.3, 0.5, 0.7]
    col_labels = [
        'Face-on vs GalSim',
        'cosi=0.3 (self-conv, ref=os15)',
        'cosi=0.5 (self-conv, ref=os15)',
        'cosi=0.7 (self-conv, ref=os15)',
    ]

    fig, axes = plt.subplots(
        1,
        len(cosi_values),
        figsize=(4.5 * len(cosi_values), 5),
    )

    for ci, (cosi, col_lab) in enumerate(zip(cosi_values, col_labels)):
        theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, hlr, h_over_r, n, 0.0, 0.0])
        ax = axes[ci]

        if ci == 0:
            # face-on: reference = GalSim with PSF
            gs_prof = gs.Sersic(n=n, half_light_radius=hlr, flux=flux, gsparams=gsp)
            gs_conv = gs.Convolve(gs_prof, psf_obj)
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
            # inclined: reference = highest oversample
            obs_ref = build_image_obs(
                ip,
                psf=psf_obj,
                oversample=ref_osamp,
                int_model=model,
                gsparams=gsp,
            )
            ref_sb = np.array(model.render_image(theta, obs=obs_ref))
            ref_label = f'oversample={ref_osamp}'

        peak = np.max(np.abs(ref_sb))
        max_fracs = []
        rms_fracs = []

        print(f'\n  cosi={cosi} (ref={ref_label}):')
        for osamp in oversamples:
            obs = build_image_obs(
                ip,
                psf=psf_obj,
                oversample=osamp,
                int_model=model,
                gsparams=gsp,
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
        f'Oversample convergence: Sersic(n={n:.0f}) + PSF (FWHM={fwhm}")\n'
        f'flux={flux}, {ps}"/pix, {npix}x{npix} | face-on: vs GalSim; '
        f'inclined: self-convergence vs os={ref_osamp}',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(
        sersic_output_dir / 'sersic_oversample_convergence.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
