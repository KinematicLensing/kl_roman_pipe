"""Fisher check of the sampled-thickness arm: image-only sigma(g+) with h/r free under its LN(0.25, 0.14 dex) prior.

The sampled-h MCMC converged in 3-25% of fits per cell, and its near-face-on widths (0.03-0.05 at SNR 5000) disagree
with the misspecification slope (|dg+/dh| ~ 2 at cos i 0.7). This computes the Gaussian-approximation answer at truth
h/r 0.25: likelihood Fisher on the noise-free image plus the prior precision on ln h, theta at 45 deg to the pixels.
A thickness change is exactly absorbed by cos i and shear, so the likelihood Fisher is near singular along that
direction and the shear width should be set by the h prior.
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_image_only as rio  # noqa: E402

H_TRUE = 0.25
SIG_LNH = 0.14 * math.log(10.0)
THETA = math.pi / 4
COSIS = (0.03, 0.1, 0.3, 0.5, 0.7, 0.9)
SNRS = (1057.0, 5000.0)


def main():
    import jax
    import jax.numpy as jnp

    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.observation import build_image_obs
    from kl_pipe.parameters import ImagePars
    from kl_pipe.priors import PriorDict, Uniform
    from kl_pipe.render import RenderConfig
    from kl_pipe.sampling import InferenceTask
    from kl_pipe.source import SourceModel

    band = rio.BAND
    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[band], band, rio.Z, mock=True)
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')
    rc = RenderConfig(oversample=rio.OVERSAMPLE)
    src = SourceModel(broadband_models={band: InclinedExponentialModel()})
    print('cos_i  snr  sig_gp_hfixed  sig_gp_hfree  sig_lnh_post/prior')
    for cosi in COSIS:
        truth = {
            'cosi': cosi,
            'theta_int': THETA,
            'g1': 0.0,
            'g2': 0.0,
            f'{band}.flux': rio.FLUX,
            f'{band}.rscale': rio.RSCALE,
            'h_over_r': H_TRUE,
            f'{band}.x0': 0.0,
            f'{band}.y0': 0.0,
        }
        clean_obs = build_image_obs(
            image_pars,
            psf=psf,
            render_config=rc,
            int_model=src.broadband_models[band],
            broadband_key=band,
        )
        data = np.asarray(src.render_broadband(truth, clean_obs, band))
        priors = PriorDict(
            {
                'g1': Uniform(-0.5, 0.5),
                'g2': Uniform(-0.5, 0.5),
                'cosi': Uniform(0.001, 0.999),
                'theta_int': Uniform(-math.pi, math.pi),
                f'{band}.flux': Uniform(0.01, 100.0),
                f'{band}.rscale': Uniform(0.001, 2.0),
                'h_over_r': Uniform(0.01, 2.0),
                f'{band}.x0': 0.0,
                f'{band}.y0': 0.0,
            }
        )
        names = list(priors.sampled_names)
        th0 = jnp.array([truth[n] for n in names])
        c, s = math.cos(2 * THETA), math.sin(2 * THETA)
        rot = np.array([[c, s], [-s, c]])
        i1, i2, ih = names.index('g1'), names.index('g2'), names.index('h_over_r')
        for snr in SNRS:
            obs = build_image_obs(
                image_pars,
                psf=psf,
                render_config=rc,
                data=jnp.asarray(data),
                variance=float(np.sum(data**2)) / snr**2,
                int_model=src.broadband_models[band],
                broadband_key=band,
            )
            task = InferenceTask.from_obs(src, priors, image_obs={band: obs})
            hess = np.asarray(jax.hessian(task.log_likelihood)(th0))
            fisher = -0.5 * (hess + hess.T)
            keep = [i for i in range(len(names)) if i != ih]
            cov_fixed = np.linalg.inv(fisher[np.ix_(keep, keep)])
            k1, k2 = keep.index(i1), keep.index(i2)
            sig_fixed = math.sqrt(
                (rot @ cov_fixed[np.ix_([k1, k2], [k1, k2])] @ rot.T)[0, 0]
            )
            # LN prior on h: precision 1 / (h sigma_lnh)^2 in h units at the truth
            fisher_h = fisher.copy()
            fisher_h[ih, ih] += 1.0 / (H_TRUE * SIG_LNH) ** 2
            cov = np.linalg.inv(fisher_h)
            sig_free = math.sqrt((rot @ cov[np.ix_([i1, i2], [i1, i2])] @ rot.T)[0, 0])
            lnh_ratio = math.sqrt(cov[ih, ih]) / H_TRUE / SIG_LNH
            print(
                f'{cosi:5.2f} {snr:6.0f} {sig_fixed:13.4f} {sig_free:13.4f} {lnh_ratio:10.3f}',
                flush=True,
            )


if __name__ == '__main__':
    main()
