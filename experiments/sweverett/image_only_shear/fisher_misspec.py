"""Image-only shear bias from model misspecification, versus the statistical width.

A noise-free F158 image of a truth that breaks one assumption of the fit model is fit by MAP with the pinned model
(round inclined exponential, h/r 0.1). The MAP offset is the bias; it does not depend on SNR, while the Fisher width
scales as 1/SNR, so bias/sigma grows linearly with image SNR.

Truth variants:
  control_exp    the fit model itself (bias must be ~0: closure)
  eint_e1_*      intrinsic face-on ellipticity along the disk major axis (EllipticalInclinedExponentialModel)
  eint_e2_*      intrinsic ellipticity at 45 deg in the disk plane
  h_*            thicker disk (truth h/r 0.15, 0.25; fit pinned 0.1)
  sersic_n*      Sersic profile at matched half-light radius (n = 1 is the emulator-vs-exponential floor)

Scene: Jiachuan fiducial, theta_int pi/4 to the pixel grid, zero shear, Roman F158 PSF, 0.11" pixels.
Prediction: e_int ~ 0.05-0.1 and h off by +0.05-0.15 give |dg+| of order 0.01-0.1, above sigma at SNR 5000.
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'kinematic_regime'))
import run_image_only as rio  # noqa: E402

THETA = math.pi / 4
COSIS = (0.1, 0.3, 0.5, 0.7)
SNRS = (1057.0, 5000.0)
HLR_PER_RSCALE = 1.6783469900166605
VARIANTS = {
    'control_exp': {},
    'eint_e1_0.05': {'e1_int': 0.05},
    'eint_e1_0.10': {'e1_int': 0.10},
    'eint_e2_0.05': {'e2_int': 0.05},
    'eint_e2_0.10': {'e2_int': 0.10},
    'h_0.15': {'h_over_r': 0.15},
    'h_0.25': {'h_over_r': 0.25},
    'sersic_n1.0': {'n_sersic': 1.0},
    'sersic_n0.8': {'n_sersic': 0.8},
    'sersic_n1.5': {'n_sersic': 1.5},
}


def truth_source_and_pars(variant: dict, cosi: float):
    from elliptical_disk import EllipticalInclinedExponentialModel

    from kl_pipe.intensity import InclinedExponentialModel, InclinedSersicModel
    from kl_pipe.source import SourceModel

    band = rio.BAND
    base = {
        'cosi': cosi,
        'theta_int': THETA,
        'g1': 0.0,
        'g2': 0.0,
        f'{band}.x0': 0.0,
        f'{band}.y0': 0.0,
    }
    h = variant.get('h_over_r', rio.H_PINNED)
    if 'n_sersic' in variant:
        hlr = HLR_PER_RSCALE * rio.RSCALE
        pars = dict(base)
        pars.update(
            {
                f'{band}.flux': rio.FLUX,
                f'{band}.hlr': hlr,
                f'{band}.h_over_hlr': h * rio.RSCALE / hlr,
                f'{band}.n_sersic': variant['n_sersic'],
            }
        )
        return SourceModel(broadband_models={band: InclinedSersicModel()}), pars
    pars = dict(base)
    pars.update({f'{band}.flux': rio.FLUX, f'{band}.rscale': rio.RSCALE, 'h_over_r': h})
    if 'e1_int' in variant or 'e2_int' in variant:
        pars[f'{band}.e1_int'] = variant.get('e1_int', 0.0)
        pars[f'{band}.e2_int'] = variant.get('e2_int', 0.0)
        return (
            SourceModel(broadband_models={band: EllipticalInclinedExponentialModel()}),
            pars,
        )
    return SourceModel(broadband_models={band: InclinedExponentialModel()}), pars


def run_case(name, variant, cosi, psf):
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.observation import build_image_obs
    from kl_pipe.parameters import ImagePars
    from kl_pipe.priors import PriorDict, Uniform
    from kl_pipe.render import RenderConfig
    from kl_pipe.sampling import InferenceTask
    from kl_pipe.source import SourceModel

    band = rio.BAND
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')
    rc = RenderConfig(oversample=rio.OVERSAMPLE)
    tsrc, tpars = truth_source_and_pars(variant, cosi)
    tobs = build_image_obs(
        image_pars,
        psf=psf,
        render_config=rc,
        int_model=tsrc.broadband_models[band],
        broadband_key=band,
    )
    data = np.asarray(tsrc.render_broadband(tpars, tobs, band))
    fsrc = SourceModel(broadband_models={band: InclinedExponentialModel()})
    priors = PriorDict(
        {
            'g1': Uniform(-0.5, 0.5),
            'g2': Uniform(-0.5, 0.5),
            'cosi': Uniform(0.001, 0.999),
            'theta_int': Uniform(-math.pi, math.pi),
            f'{band}.flux': Uniform(0.01, 100.0),
            f'{band}.rscale': Uniform(0.001, 2.0),
            'h_over_r': rio.H_PINNED,
            f'{band}.x0': 0.0,
            f'{band}.y0': 0.0,
        }
    )
    out = dict(variant=name, cosi=cosi)
    var5000 = float(np.sum(data**2)) / 5000.0**2
    obs = build_image_obs(
        image_pars,
        psf=psf,
        render_config=rc,
        data=jnp.asarray(data),
        variance=var5000,
        int_model=fsrc.broadband_models[band],
        broadband_key=band,
    )
    task = InferenceTask.from_obs(fsrc, priors, image_obs={band: obs})
    names = list(priors.sampled_names)
    nominal = {
        'cosi': cosi,
        'theta_int': THETA,
        'g1': 0.0,
        'g2': 0.0,
        f'{band}.flux': rio.FLUX,
        f'{band}.rscale': rio.RSCALE,
    }
    x0 = np.array([nominal[n] for n in names])
    f = jax.jit(lambda th: -task.log_likelihood(th))
    g = jax.jit(jax.grad(lambda th: -task.log_likelihood(th)))
    res = minimize(
        lambda x: float(f(jnp.asarray(x))),
        x0,
        jac=lambda x: np.asarray(g(jnp.asarray(x))),
        method='L-BFGS-B',
        bounds=[
            (
                (-0.5, 0.5)
                if n in ('g1', 'g2')
                else (0.001, 0.999) if n == 'cosi' else (None, None)
            )
            for n in names
        ],
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10},
    )
    xm = res.x
    c, s = math.cos(2 * THETA), math.sin(2 * THETA)
    g1, g2 = xm[names.index('g1')], xm[names.index('g2')]
    out['dg_plus'] = g1 * c + g2 * s
    out['dg_cross'] = -g1 * s + g2 * c
    out['dcosi'] = xm[names.index('cosi')] - cosi
    # residual chi2 of the best fit at SNR 5000 (noise-free data, so the expected value is 0 for a correct model)
    best = {n: float(v) for n, v in zip(names, xm)}
    best.update({'h_over_r': rio.H_PINNED, f'{band}.x0': 0.0, f'{band}.y0': 0.0})
    model_img = np.asarray(fsrc.render_broadband(best, obs, band))
    out['chi2_min_snr5000'] = float(np.sum((data - model_img) ** 2) / var5000)
    out['converged'] = bool(res.success)
    # Fisher from the fit model's own noise-free image at the nominal point (residual-free, so the Hessian is J^T C^-1 J)
    clean = np.asarray(
        fsrc.render_broadband(
            {**nominal, 'h_over_r': rio.H_PINNED, f'{band}.x0': 0.0, f'{band}.y0': 0.0},
            obs,
            band,
        )
    )
    fobs = build_image_obs(
        image_pars,
        psf=psf,
        render_config=rc,
        data=jnp.asarray(clean),
        variance=float(np.sum(clean**2)) / 5000.0**2,
        int_model=fsrc.broadband_models[band],
        broadband_key=band,
    )
    ftask = InferenceTask.from_obs(fsrc, priors, image_obs={band: fobs})
    hess = np.asarray(jax.hessian(ftask.log_likelihood)(jnp.asarray(x0)))
    cov = np.linalg.inv(-0.5 * (hess + hess.T))
    i1, i2 = names.index('g1'), names.index('g2')
    rot = np.array([[c, s], [-s, c]])
    cg = rot @ cov[np.ix_([i1, i2], [i1, i2])] @ rot.T
    for snr in SNRS:
        k = 5000.0 / snr
        out[f'sig_gp_snr{snr:g}'] = math.sqrt(cg[0, 0]) * k
        out[f'pull_gp_snr{snr:g}'] = out['dg_plus'] / (math.sqrt(cg[0, 0]) * k)
        out[f'chi2_min_snr{snr:g}'] = out['chi2_min_snr5000'] / k**2
    return out


def main():
    import pandas as pd

    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig

    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[rio.BAND], rio.BAND, rio.Z, mock=True)
    rows = []
    for name, variant in VARIANTS.items():
        for cosi in COSIS:
            r = run_case(name, variant, cosi, psf)
            rows.append(r)
            print(
                {
                    k: (float(f'{v:.4g}') if isinstance(v, float) else v)
                    for k, v in r.items()
                },
                flush=True,
            )
    df = pd.DataFrame(rows)
    (HERE / 'out').mkdir(exist_ok=True)
    df.to_csv(HERE / 'out' / 'fisher_misspec.csv', index=False)
    pd.set_option('display.width', 220)
    cols = [
        'variant',
        'cosi',
        'dg_plus',
        'dg_cross',
        'dcosi',
        'sig_gp_snr5000',
        'pull_gp_snr1057',
        'pull_gp_snr5000',
        'chi2_min_snr1057',
        'chi2_min_snr5000',
        'converged',
    ]
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.4g}'))


if __name__ == '__main__':
    main()
