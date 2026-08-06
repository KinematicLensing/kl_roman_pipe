---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Roman Reference: a full-complexity mock + fit

A terse, reference-quality template for a realistic Roman kinematic-lensing fit:
the two HLWAS medium-tier bands (F129 + F158, each a coadd of several
exposures), two grism roll angles for one emission line, WCS-aware geometry,
the production Roman WFI PSF (`galsim.roman.getPSF`), a bulge + disk composite
shared across the broadbands and the line continuum, and a separate disk for
the emission line. It builds the data vector, shows its components, runs an
optimizer, and leaves the sampler call commented.

This is a worked template, not a tutorial. For the concepts see `quickstart.md`
(the pedagogical version of this same configuration) and `sampling.md`.

```{code-cell} python
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from IPython.display import Image
from astropy.wcs import WCS
import galsim
import galsim.roman

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import BulgeDiskModel, InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_image_obs, build_grism_obs, build_velocity_obs
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.priors import PriorDict, Uniform, Gaussian, TruncatedNormal
from kl_pipe.render import RenderConfig
from kl_pipe.sampling import InferenceTask, NumpyroSamplerConfig, build_sampler
from kl_pipe.noise import add_intensity_noise
from kl_pipe.optimization import multi_start_minimize
from kl_pipe.diagnostics.imaging import plot_parameter_recovery
from kl_pipe.surveys import roman
```

```{code-cell} python
def make_wcs(shape, pixel_scale_arcsec, pa_deg, crval=(150.0, 2.0)):
    """TAN WCS with a sky position angle (cdelt + pc rotation, read via get_pc)."""
    w = WCS(naxis=2)
    cd = pixel_scale_arcsec / 3600.0
    rot = np.deg2rad(pa_deg)
    w.wcs.cdelt = [-cd, cd]
    w.wcs.pc = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.crval = list(crval)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.pixel_shape = (shape[1], shape[0])
    return w
```

```{code-cell} python
# --- Configuration -------------------------------------------------------
Z = 1.0
SHAPE = (32, 32)
PIXEL_SCALE = 0.11                        # arcsec/pixel, Roman native
BANDS = ('F129', 'F158')                  # the HLWAS medium-tier pair
ROLLS = {'roll0': 20.0, 'roll90': 110.0}  # sky position angle per grism roll
N_EXP = 4                                 # exposures coadded per band
SNR_EXP = 50                              # per-exposure broadband SNR
SNR_GRISM = 150                           # whole-stamp grism SNR (note below)
SEED = 42

img_pars = ImagePars(shape=SHAPE, wcs=make_wcs(SHAPE, PIXEL_SCALE, pa_deg=20.0))

# Production Roman WFI PSFs: monochromatic at each band's effective
# wavelength, and at the observed Halpha wavelength for the grism. Each
# takes a few seconds to construct; this is the same PSF the ensemble
# pipeline uses.
psf_band = {
    band: galsim.roman.getPSF(
        10,                                                    # SCA
        None,
        wavelength=roman.BAND_EFFECTIVE_LAMBDA_A[band] / 10.0,  # Angstrom -> nm
        pupil_bin=4,
    )
    for band in BANDS
}
psf_grism = galsim.roman.getPSF(
    10,
    None,
    wavelength=LINE_LAMBDAS['Halpha'] * (1.0 + Z),  # nm
    pupil_bin=4,
)
```

```{code-cell} python
# --- Source model ---------------------------------------------------------
# Bulge+disk composites for the two bands and the line continuum; the
# emission line gets its own disk (line-emitting gas and stellar light need
# not share a scale length). All components share the galaxy geometry.
bd_F129, bd_F158, bd_cont = BulgeDiskModel(), BulgeDiskModel(), BulgeDiskModel()
line_model = InclinedExponentialModel()

source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F129': bd_F129, 'F158': bd_F158},
    emission_lines={
        'Halpha': EmissionLine(intensity=line_model, continuum=bd_cont),
    },
)

# Parameter names without a component prefix (cosi, disk_rscale, ...) are
# shared by every component that uses them; fluxes carry their component's
# prefix. Fluxes here are arbitrary units -- the physical-units section at
# the end shows the microjansky convention.
truth = {
    'cosi': 0.55,
    'theta_int': 0.6,
    'g1': 0.03,
    'g2': -0.02,
    'vel.v0': 5.0,
    'vel.vcirc': 210.0,
    'vel.rscale': 0.35,
    'bulge_frac': 0.3,
    'disk_rscale': 0.35,
    'disk_h_over_r': 0.1,
    'disk_x0': 0.0,
    'disk_y0': 0.0,
    'bulge_hlr': 0.18,
    'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0,
    'bulge_y0': 0.0,
    'F129.total_flux': 120.0,
    'F158.total_flux': 180.0,
    'Halpha.flux': 90.0,
    'Halpha.rscale': 0.28,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 55.0,
    'Halpha.cont.total_flux': 20.0,
    'z': Z,
}
```

```{code-cell} python
# --- Build the data vector ------------------------------------------------
# Broadband: each band is a coadd of N_EXP exposures, so the coadd noise is
# the per-exposure noise / sqrt(N_EXP).
band_exps = {}
image_obs = {}

for i, band in enumerate(BANDS):
    model = source.broadband_models[band]

    render_obs = build_image_obs(
        img_pars,
        psf=psf_band[band],
        int_model=model,
        broadband_key=band,
    )
    clean = np.asarray(source.render_broadband(truth, render_obs, band))

    exps = []
    for k in range(N_EXP):
        noisy, var_exp = add_intensity_noise(
            clean,
            target_snr=SNR_EXP,
            seed=SEED + 10 * i + k,
        )
        exps.append(noisy)
    band_exps[band] = exps

    coadd = np.mean(exps, axis=0)
    var_coadd = var_exp / N_EXP  # uniform per-exposure variance, coadded

    image_obs[band] = build_image_obs(
        img_pars,
        psf=psf_band[band],
        int_model=model,
        broadband_key=band,
        data=jnp.asarray(coadd),
        variance=var_coadd,
    )
```

```{code-cell} python
# Grism: one emission line dispersed at two roll angles; the roll enters
# through the WCS position angle. SNR_GRISM is a whole-stamp SNR for
# brevity -- a physically better label normalizes on the line alone
# (kl_pipe.noise.grism_line_noise; see the pathways section below).
grism_obs = {}

for j, (roll, pa) in enumerate(ROLLS.items()):
    roll_pars = ImagePars(shape=SHAPE, wcs=make_wcs(SHAPE, PIXEL_SCALE, pa_deg=pa))
    gpars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=Z,
        image_pars=roll_pars,
        dispersion=1.1,
    )

    render_obs = build_grism_obs(gpars, z=Z, psf=psf_grism)
    clean = np.asarray(source.render_grism(truth, render_obs))
    noisy, var = add_intensity_noise(
        clean,
        target_snr=SNR_GRISM,
        seed=SEED + 100 + j,
    )

    grism_obs[roll] = build_grism_obs(
        gpars,
        z=Z,
        psf=psf_grism,
        data=jnp.asarray(noisy),
        variance=var,
    )
```

```{code-cell} python
# --- Data-vector components ------------------------------------------------
# Top row: individual F129 exposures + their coadd (noise drops as 1/sqrt(N)).
# Bottom row: the grism rolls (dispersed in different directions) + velocity.
vel_ctx = np.asarray(source.render_velocity(truth, build_velocity_obs(img_pars)))

fig, axes = plt.subplots(2, max(N_EXP, len(ROLLS) + 1), figsize=(15, 7))

for k in range(N_EXP):
    axes[0, k].imshow(band_exps['F129'][k], origin='lower', cmap='viridis')
    axes[0, k].set_title(f'F129 exp {k + 1}', fontsize=9)
axes[0, -1].imshow(np.mean(band_exps['F129'], axis=0), origin='lower', cmap='viridis')
axes[0, -1].set_title(f'F129 coadd (x{N_EXP})', fontsize=9)

for j, roll in enumerate(ROLLS):
    axes[1, j].imshow(np.asarray(grism_obs[roll].data), origin='lower', cmap='magma')
    axes[1, j].set_title(f'grism {roll}', fontsize=9)
vmax = float(np.max(np.abs(vel_ctx)))
axes[1, len(ROLLS)].imshow(vel_ctx, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1, len(ROLLS)].set_title('LOS velocity (truth)', fontsize=9)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    if not ax.images:
        ax.axis('off')
plt.tight_layout(); plt.show()
```

```{code-cell} python
# --- Priors + InferenceTask ------------------------------------------------
# Sample geometry, velocity, shared morphology, per-band flux, line; fix the
# thickness parameters and z. Shear +/- 0.25; TF prior on vcirc (Ubler+2017).
priors = PriorDict({
    'cosi': TruncatedNormal(0.55, 0.15, 0.05, 0.99),
    'theta_int': TruncatedNormal(0.6, 0.3, 0.0, np.pi),
    'g1': Uniform(-0.25, 0.25),
    'g2': Uniform(-0.25, 0.25),
    'vel.v0': Gaussian(5.0, 10.0),
    'vel.vcirc': TruncatedNormal(210.0, 37.0, 80.0, 400.0),
    'vel.rscale': TruncatedNormal(0.35, 0.1, 0.05, 1.0),
    'bulge_frac': TruncatedNormal(0.3, 0.1, 0.0, 0.9),
    'disk_rscale': TruncatedNormal(0.35, 0.08, 0.05, 1.0),
    'disk_h_over_r': 0.1,
    'disk_x0': 0.0,
    'disk_y0': 0.0,
    'bulge_hlr': TruncatedNormal(0.18, 0.05, 0.02, 0.6),
    'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0,
    'bulge_y0': 0.0,
    'F129.total_flux': TruncatedNormal(120.0, 25.0, 30.0, 400.0),
    'F158.total_flux': TruncatedNormal(180.0, 30.0, 30.0, 500.0),
    'Halpha.flux': TruncatedNormal(90.0, 20.0, 20.0, 250.0),
    'Halpha.rscale': TruncatedNormal(0.28, 0.08, 0.05, 1.0),
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': TruncatedNormal(55.0, 20.0, 5.0, 150.0),
    'Halpha.cont.total_flux': TruncatedNormal(20.0, 12.0, 0.0, 150.0),
    'z': Z,
})

task = InferenceTask.from_obs(source, priors, image_obs=image_obs, grism_obs=grism_obs)
theta_truth = jnp.array([truth[n] for n in task.sampled_names])
print(f"sampled dimension: {task.n_params}")
print(f"joint log-posterior at truth: {float(task.log_posterior(theta_truth)):.2f}")
```

```{code-cell} python
# --- Optimizer (MAP) point estimate ----------------------------------------
grad_fn = task.get_log_posterior_and_grad_fn()

def neg_logpost(th):
    val, grad = grad_fn(jnp.asarray(th))
    return -float(val), -np.asarray(grad)

rng = np.random.default_rng(0)
x0 = theta_truth + 0.03 * np.abs(theta_truth) * rng.standard_normal(task.n_params)

# A short illustrative descent (the real fit is the commented sampler below);
# rendering two broadband composites + two grism cubes per step is not cheap.
result = multi_start_minimize(
    neg_logpost,
    x0=np.asarray(x0),
    bounds=task.get_bounds(),
    n_starts=1,
    jac=True,
    options={'maxiter': 60},
)
fit = dict(zip(task.sampled_names, (float(v) for v in result.x)))

res = plot_parameter_recovery(
    true_values={n: truth[n] for n in task.sampled_names},
    recovered_values=fit,
    output_dir=Path(tempfile.mkdtemp()),
    test_name='roman_reference',
)
Image(res['output_path'])
```

The MAP recovers the well-constrained parameters; shear and the inclination-
coupled parameters need the full posterior. Run NUTS to get it (commented;
minutes to tens of minutes at these settings, longer at production sizes):

```{code-cell} python
# config = NumpyroSamplerConfig(
#     n_warmup=50, n_samples=500, n_chains=4, chain_method='vectorized',
#     precondition='laplace', max_tree_depth=10, seed=42, progress=True,
# )
# result = build_sampler('numpyro', task, config).run()
```

Swap the synthetic renders for real coadded broadband images, per-roll grism
frames, and their variance maps, and the same `task` runs unchanged.

## Physical flux units and Roman noise

Everything above is unit-agnostic: the fluxes are arbitrary numbers and you
choose the SNR. For Roman work you can instead carry physical units and let
the noise come from the mission itself. The conventions
(`docs/units_and_conventions.md`):

- broadband fluxes in microjansky, so a rendered image is in uJy/pixel;
- line fluxes in units of 1e-17 erg/s/cm2 (`photometry.CGS_TO_F17`), the
  unit of the grism literature and of the published limit itself;
- published depths, flux limits, and throughput tables live in
  `kl_pipe.surveys.roman`.

There are three noise pathways, named as in the ensemble's `noise_model`
knob and its comparison diagnostic
(`tests/test_ensemble_noise.py::TestNoiseModelComparisonFigure`):

- **bkg** -- a flat per-pixel background solved from the published survey
  depth; no source term. The realized SNR is whatever the survey delivers
  for your galaxy: you measure it, you do not choose it.
- **poisson** -- the bkg level plus the source's own shot noise (a per-pixel
  variance map). The most physical of the three.
- **matched_filter** -- you declare the SNR and a flat noise level is derived
  from the rendered template so the declared SNR is realized exactly. No
  physical units needed. This is the pipeline default.

Where the background level comes from: a published depth is a statement
about one specific reference source -- "a point source at f_lim reaches
5 sigma" for the imaging bands, "a 0.25 arcsec disk at f_lim reaches
5 sigma" for the grism. The reference source is fixed by the publication
and is never a choice you make. Turning that statement into a per-pixel
sigma does require the instrument model (how the PSF and pixel scale spread
the reference's flux), so the anchor is a property of survey + instrument.
For the production instrument model -- the WFI PSFs used throughout this
notebook -- the anchors are pinned and importable:
`roman.SIGMA_BKG_DEFAULT_UJY` (bands) and
`roman.GRISM_SIGMA_BKG_DEFAULT_PER_PASS_F17` (grism). Re-anchoring through
`band_sigma_bkg_ujy` / `grism_sigma_bkg_per_pass` is only needed for a
non-default PSF or pixel scale.

```{code-cell} python
from kl_pipe.photometry import ab_mag_to_ujy
from kl_pipe.noise import physical_variance_map, add_map_noise, matched_filter_snr

# A disk with a real brightness: m_AB = 23 in F129 (~2.3 uJy).
disk = InclinedExponentialModel()
pars = {
    'flux': float(ab_mag_to_ujy(23.0)),
    'rscale': 0.25,
    'h_over_r': 0.1,
    'cosi': 0.55,
    'theta_int': 0.6,
    'g1': 0.0,
    'g2': 0.0,
    'x0': 0.0,
    'y0': 0.0,
}
theta = jnp.array([pars[n] for n in disk.PARAMETER_NAMES])

ip = ImagePars(shape=SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij')
obs_render = build_image_obs(ip, psf=psf_band['F129'], int_model=disk)
clean_ujy = np.asarray(disk.render_image(theta, obs=obs_render))  # uJy/pixel
```

**Pathway 1, bkg.** The production anchor is one import. The re-derivation
below it is only needed off the production instrument model; here it agrees
because this notebook uses the same PSF:

```{code-cell} python
sigma_bkg = roman.SIGMA_BKG_DEFAULT_UJY['F129']  # uJy/pixel

# re-anchoring by hand, for a custom PSF or pixel scale:
kernel = psf_band['F129'].drawImage(scale=PIXEL_SCALE).array
sigma_check = roman.band_sigma_bkg_ujy('F129', float(np.sqrt((kernel**2).sum())))
print(f'pinned {sigma_bkg * 1e3:.3f} nJy/pix; re-derived {sigma_check * 1e3:.3f}')

var_bkg = np.full_like(clean_ujy, sigma_bkg**2)
noisy_bkg = add_map_noise(clean_ujy, var_bkg, seed=7)
snr_bkg = matched_filter_snr(clean_ujy, var_bkg)
print(f'bkg realized SNR = {snr_bkg:.1f}')
```

**Pathway 2, poisson.** The galaxy's own photons add Poisson variance on
top of the flat background, converted through the mission zeropoint
(`ELECTRONS_PER_UJY`; the detector e-/ADU gain never enters). The draw is
the heteroscedastic Gaussian of that variance map -- the high-count limit
of Poisson statistics, and exactly the noise a Gaussian likelihood carrying
the same map describes:

```{code-cell} python
var_map = physical_variance_map(clean_ujy, sigma_bkg, roman.ELECTRONS_PER_UJY['F129'])
noisy = add_map_noise(clean_ujy, var_map, seed=7)

obs_physical = build_image_obs(
    ip,
    psf=psf_band['F129'],
    int_model=disk,
    data=jnp.asarray(noisy),
    variance=jnp.asarray(var_map),
    flux_unit='uJy (band-averaged f_nu)',
)
snr_realized = matched_filter_snr(clean_ujy, var_map)
print(f'realized SNR = {snr_realized:.1f} (background alone: {snr_bkg:.1f}; '
      f'the difference is this galaxy\'s shot noise)')
```

**Pathway 3, matched_filter** (what the rest of this notebook uses). The
logic runs the other way: you declare the SNR and the uniform noise level
is derived from the rendered template, `var = ||T||^2 / SNR^2`. Exact by
construction and unit-free -- the right tool for controlled experiments at
a chosen depth. The poisson pathway is the right tool when the question is
what Roman would actually deliver for this galaxy:

```{code-cell} python
noisy_mf, var_mf = add_intensity_noise(clean_ujy, target_snr=snr_realized, seed=7)
print(f'matched-filter path at target {snr_realized:.1f}: uniform sigma = '
      f'{np.sqrt(var_mf.flat[0]):.4g} uJy/pix vs background {sigma_bkg:.4g}')
```

**The grism channel follows the same recipe with two changes.** First, its
published limit refers to the extended reference source described above, so
the pinned anchor was derived by dispersing that 0.25" disk through the
production grism model. Second, a grism SNR should normalize on the
emission line alone: the continuum dominates the dispersed power but
carries no kinematic signal (`kl_pipe.noise.grism_line_noise` implements
the declared-SNR version of this convention).

```{code-cell} python
from kl_pipe.photometry import CGS_TO_F17, EXP_R50_OVER_RSCALE, HALPHA_REST_A

line_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    emission_lines={
        'Halpha': EmissionLine(
            intensity=InclinedExponentialModel(),
            continuum=InclinedExponentialModel(),
        ),
    },
)

def halpha_truth(f_line_cgs, z, **overrides):
    """Truth for a round Halpha disk; line flux in erg/s/cm2."""
    pars = {
        'cosi': 1.0,
        'theta_int': 0.0,
        'g1': 0.0,
        'g2': 0.0,
        'z': z,
        'vel.v0': 0.0,
        'vel.vcirc': 0.0,
        'vel.rscale': 1.0,
        'Halpha.flux': f_line_cgs * CGS_TO_F17,       # -> 1e-17 erg/s/cm2 units
        'Halpha.rscale': 0.25 / EXP_R50_OVER_RSCALE,  # r50 = 0.25"
        'Halpha.h_over_r': 0.1,
        'Halpha.x0': 0.0,
        'Halpha.y0': 0.0,
        'Halpha.dispersion': roman.GRISM_REF_LINE_WIDTH_KMS,
        'Halpha.cont.flux_per_nm': 0.0,
        'Halpha.cont.rscale': 0.25,
        'Halpha.cont.h_over_r': 0.1,
        'Halpha.cont.x0': 0.0,
        'Halpha.cont.y0': 0.0,
    }
    pars.update(overrides)
    return pars

def grism_render(pars):
    z = pars['z']
    gpars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=z,
        image_pars=ip,
        dispersion=1.1,
    )
    psf_z = galsim.roman.getPSF(
        10,
        None,
        wavelength=LINE_LAMBDAS['Halpha'] * (1.0 + z),
        pupil_bin=4,
    )
    obs = build_grism_obs(
        gpars,
        z=z,
        psf=psf_z,
        render_config=RenderConfig(oversample=3),
    )
    return np.asarray(line_source.render_grism(pars, obs))

# The pinned per-pass anchor, and its re-derivation: render the reference
# disk at the per-pass limit and demand matched-filter SNR = 5. They agree
# here because this notebook matches the production grism setup (stamp,
# dispersion, pixel scale, PSF).
sigma_bkg_grism = roman.GRISM_SIGMA_BKG_DEFAULT_PER_PASS_F17

z_ref = roman.F_LIM_REF_LAMBDA_A / HALPHA_REST_A - 1.0
L_ref = grism_render(halpha_truth(roman.F_LIM_PER_PASS_CGS, z_ref))
sigma_grism_check = roman.grism_sigma_bkg_per_pass(float(np.sqrt((L_ref**2).sum())))
print(f'pinned {sigma_bkg_grism:.4f}; re-derived {sigma_grism_check:.4f} '
      f'(1e-17 erg/s/cm2 per pixel)')
```

```{code-cell} python
# Your galaxy: a catalog line flux at the per-pass limit, inclined, rotating,
# with continuum. Shot noise counts line and continuum photons; the SNR
# convention normalizes on the line alone.
gal = halpha_truth(
    3e-16,
    1.0,
    **{
        'cosi': 0.55,
        'theta_int': 0.6,
        'vel.vcirc': 200.0,
        'Halpha.dispersion': 50.0,
        'Halpha.cont.flux_per_nm': 0.75,
    },
)
clean_grism = grism_render(gal)
g_grism = float(roman.grism_electrons_per_f17_per_pass(
    HALPHA_REST_A * (1.0 + gal['z'])
))  # detected electrons per flux unit, one pass

var_grism = physical_variance_map(clean_grism, sigma_bkg_grism, g_grism)
noisy_grism = add_map_noise(clean_grism, var_grism, seed=8)

line_only = grism_render({**gal, 'Halpha.cont.flux_per_nm': 0.0})
snr_line = matched_filter_snr(line_only, var_grism)
print(f'realized per-pass line SNR = {snr_line:.1f} '
      f'(x2 over the 4-pass coadd; the reference source at this flux reads 5.0)')
```

All of these observations run through `InferenceTask` identically --
per-pixel variance maps are supported everywhere. In the ensemble machinery
the pathways are the `noise_model: matched_filter | poisson` knob on the
observation config (bkg is the poisson pathway without its shot term), which
automates the anchoring above and records the realized `snr_effective` per
channel for every fit.
