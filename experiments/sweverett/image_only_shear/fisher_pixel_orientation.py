"""Noise-free Fisher check: does image-only shear information depend on the disk's orientation to the pixel grid?

Hypothesis (from the vista pinned arm, image SNR 5000): edge-on, sigma(g+) is ~2x smaller with the major axis at
45 deg to the pixel rows than aligned with them, because a tilted, unresolved minor axis samples many sub-pixel
phases (slanted-edge effect). Predictions if physical: the gap shrinks with finer pixels (0.055") and with
inclination (resolved minor axis), and does not move with the render oversample (3 -> 7). If the gap moves with
oversample, it is a rendering artifact.

Fisher = -Hessian of the Gaussian log-likelihood at truth on the noise-free image (exact for Gaussian noise),
matched-filter variance at fixed SNR so every configuration has the same total SNR. Free: g1, g2, cos i, theta,
flux, rscale; h/r 0.1 and centroid fixed. Disk-frame (g+, gx) by rotating the shear block by the true theta.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_image_only as rio  # noqa: E402

SNR = 5000.0
THETAS = (0.0, math.pi / 16, math.pi / 8, math.pi / 4)
COSIS = (0.03, 0.1, 0.3, 0.5, 0.7)
# (label, pixel scale arcsec, stamp pix, render oversample)
SETUPS = (
    ('px0.11_os3', 0.11, 32, 3),
    ('px0.11_os7', 0.11, 32, 7),
    ('px0.055_os3', 0.055, 64, 3),
)


def fisher_sigmas(cosi, theta, pixel_scale, stamp, oversample, psf):
    import jax
    import jax.numpy as jnp

    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.observation import build_image_obs
    from kl_pipe.parameters import ImagePars
    from kl_pipe.priors import PriorDict, Uniform
    from kl_pipe.render import RenderConfig
    from kl_pipe.sampling import InferenceTask
    from kl_pipe.source import SourceModel

    band = rio.BAND
    truth = {
        'cosi': cosi,
        'theta_int': theta,
        'g1': 0.0,
        'g2': 0.0,
        f'{band}.flux': rio.FLUX,
        f'{band}.rscale': rio.RSCALE,
        'h_over_r': rio.H_PINNED,
        f'{band}.x0': 0.0,
        f'{band}.y0': 0.0,
    }
    source = SourceModel(broadband_models={band: InclinedExponentialModel()})
    image_pars = ImagePars(shape=(stamp, stamp), pixel_scale=pixel_scale, indexing='ij')
    model = source.broadband_models[band]
    clean = build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
        int_model=model,
        broadband_key=band,
    )
    data = np.asarray(source.render_broadband(truth, clean, band))
    var = float(np.sum(data**2)) / SNR**2
    obs = build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
        data=jnp.asarray(data),
        variance=var,
        int_model=model,
        broadband_key=band,
    )
    # wide boxes: only the likelihood curvature matters here
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
    task = InferenceTask.from_obs(source, priors, image_obs={band: obs})
    names = list(priors.sampled_names)
    th0 = jnp.array([truth[n] for n in names])
    hess = np.asarray(jax.hessian(task.log_likelihood)(th0))
    fisher = -0.5 * (hess + hess.T)
    cov = np.linalg.inv(fisher)
    i1, i2 = names.index('g1'), names.index('g2')
    c, s = math.cos(2 * theta), math.sin(2 * theta)
    rot = np.array([[c, s], [-s, c]])
    cg = rot @ cov[np.ix_([i1, i2], [i1, i2])] @ rot.T
    return {
        'sig_gp': math.sqrt(cg[0, 0]),
        'sig_gx': math.sqrt(cg[1, 1]),
        'sig_cosi': math.sqrt(cov[names.index('cosi'), names.index('cosi')]),
        'min_eig': float(np.linalg.eigvalsh(fisher).min()),
    }


def main():
    import pandas as pd

    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig

    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[rio.BAND], rio.BAND, rio.Z, mock=True)
    rows = []
    for label, px, stamp, os_ in SETUPS:
        for cosi in COSIS:
            for th in THETAS:
                r = fisher_sigmas(cosi, th, px, stamp, os_, psf)
                r.update(setup=label, cosi=cosi, theta_deg=round(math.degrees(th), 2))
                rows.append(r)
                print(r, flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / 'out'
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'fisher_pixel_orientation.csv', index=False)
    pd.set_option('display.width', 200)
    for col in ('sig_gp', 'sig_gx', 'sig_cosi'):
        print(f'\n== {col} (rows setup x cos i, columns theta deg) ==')
        print(
            df.pivot_table(index=['setup', 'cosi'], columns='theta_deg', values=col)
            .round(4)
            .to_string()
        )


if __name__ == '__main__':
    main()
