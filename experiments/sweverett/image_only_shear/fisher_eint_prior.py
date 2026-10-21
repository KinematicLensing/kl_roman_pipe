"""Image-only shear width vs the intrinsic face-on shape prior (and the thickness prior).

Fisher on the noise-free Roman-case image (0.11" pixels, F158 PSF, disk horizontal) of a thick disk (h/r 0.25) with
free g+, gx, cos i, theta, flux, scale length, h/r and intrinsic ellipticity (e1, e2). Priors: h/r lognormal
0.14 dex (or fixed), (e1, e2) Gaussian with per-component width sigma_e (0 = fixed round). Question: how tight must the
intrinsic-shape prior be for the image alone to say anything about shear?
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import explainer as ex  # noqa: E402
import run_image_only as rio  # noqa: E402

H = 0.25
SIG_LNH = 0.14 * math.log(10.0)
COSIS = (0.1, 0.3, 0.5, 0.7, 0.9)
SIG_ES = (0.0, 0.02, 0.06, 0.1)
SNRS = (1000.0, 5000.0)
NAMES = [
    'gp',
    'gx',
    'cosi',
    'theta_int',
    'flux',
    'rscale',
    'h_over_r',
    'e1_int',
    'e2_int',
]


def main():
    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig

    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[rio.BAND], rio.BAND, rio.Z, mock=True)
    print('cos_i   SNR  h_prior  sig_e | sig_gp  sig_gx')
    for cosi in COSIS:
        ex.COSI = cosi
        img, d = ex.derivatives(H, psf, 0.11, 24)
        jac = np.stack([d[n].ravel() for n in NAMES], axis=1)
        for snr in SNRS:
            sig_pix = math.sqrt(np.sum(img**2)) / snr
            fisher = jac.T @ jac / sig_pix**2
            for h_prior in ('fixed', 'LN0.14'):
                for sig_e in SIG_ES:
                    keep = list(range(len(NAMES)))
                    f = fisher.copy()
                    if h_prior == 'fixed':
                        keep.remove(NAMES.index('h_over_r'))
                    else:
                        f[NAMES.index('h_over_r'), NAMES.index('h_over_r')] += (
                            1.0 / (H * SIG_LNH) ** 2
                        )
                    if sig_e == 0.0:
                        keep.remove(NAMES.index('e1_int'))
                        keep.remove(NAMES.index('e2_int'))
                    else:
                        for n in ('e1_int', 'e2_int'):
                            f[NAMES.index(n), NAMES.index(n)] += 1.0 / sig_e**2
                    cov = np.linalg.inv(f[np.ix_(keep, keep)])
                    sp = math.sqrt(cov[keep.index(0), keep.index(0)])
                    sx = math.sqrt(cov[keep.index(1), keep.index(1)])
                    print(
                        f'{cosi:5.2f} {snr:5.0f} {h_prior:>7} {sig_e:6.2f} | {sp:6.3f} {sx:6.3f}',
                        flush=True,
                    )


if __name__ == '__main__':
    main()
