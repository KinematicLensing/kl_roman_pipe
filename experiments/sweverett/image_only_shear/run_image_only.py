"""Image-only (F158, no grism) shear inference on the roll-angle forecast's fiducial galaxy.

Question: how much shear constraint does a perfectly specified inclined thick exponential disk give from
photometry alone, as a function of image SNR, inclination, PA and the disk-thickness assumption?

Scene: Xu Table I fiducial as in configs/ensembles/rollgrid_j_*.yaml (F158 scale length 0.17875", centroid 0,
Roman F158 PSF from hlwas_medium_roman_rotac_rolls_f158 at z = 1, 0.11" pixels, 32 px stamp, render oversample 3,
matched-filter noise var = ||I||^2 / SNR^2). The true shear is |g| = 0.05; each noise draw is fit twice, at +g
and -g with the same unit deviates, so the response R = (ghat(+) - ghat(-)) / 2g is free of additive terms.
Shear prior: per-component flat box U(-0.2, 0.2); narrower priors by post hoc reweighting.

Arms:
  pinned   h/r = 0.1 in truth and model (the forecast's setup)
  sampled  h/r drawn per noise draw from LN(0.25, 0.14 dex), fit with the same distribution as prior

PA configurations (shear angle relative to the disk major axis, spin-2):
  pa0      theta_int 0,    sky shear (g, 0)      -> pure g+ along the axes
  pa22     theta_int pi/8, sky shear (g, 0)      -> 45 deg mixed
  pa45     theta_int pi/4, sky shear (g, 0)      -> pure gx
  pa45rot  theta_int pi/4, sky shear (0, g)      -> pure g+, galaxy at 45 deg to the pixel grid (grid control)

Usage:
  python run_image_only.py list --arm pinned                 # print task count
  python run_image_only.py run --arm pinned --tasks 0:10 --out out/
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

SNRS = (10.0, 47.0, 224.0, 1057.0, 5000.0)
COSIS = (0.03, 0.1, 0.3, 0.5, 0.7, 0.9)
PA_CONFIGS = {
    # name: (theta_int, spin-2 angle of the sky shear)
    'pa0': (0.0, 0.0),
    'pa22': (math.pi / 8, 0.0),
    'pa45': (math.pi / 4, 0.0),
    'pa45rot': (math.pi / 4, math.pi / 4),
}
GRID_CONTROL_SNRS = (224.0, 5000.0)
N_REPS = 4
G_TRUE = 0.05
RSCALE = 0.17875
FLUX = 1.0
H_PINNED = 0.1
# near-thin disk: projection is close to affine, so shear and inclination are degenerate in the image
H_THIN = 0.005
H_MEDIAN, H_SCATTER_DEX = 0.25, 0.14
SHEAR_PRIOR_HALFWIDTH = 0.2
BAND = 'F158'
OBS_CONFIG = ROOT / 'configs/observation/hlwas_medium_roman_rotac_rolls_f158.yaml'
Z = 1.0
OVERSAMPLE = 3


def tasks(arm: str) -> list:
    out = []
    for snr in SNRS:
        for cosi in COSIS:
            for pa in PA_CONFIGS:
                if pa == 'pa45rot' and snr not in GRID_CONTROL_SNRS:
                    continue
                for rep in range(N_REPS):
                    for sign in (+1, -1):
                        out.append(
                            dict(arm=arm, snr=snr, cosi=cosi, pa=pa, rep=rep, sign=sign)
                        )
    return out


def _seed(*parts) -> int:
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def fit_id(t: dict) -> str:
    s = 'p' if t['sign'] > 0 else 'm'
    return f"{t['arm']}_snr{t['snr']:g}_c{t['cosi']:g}_{t['pa']}_r{t['rep']}_{s}"


def truth_for(t: dict) -> dict:
    theta, phi = PA_CONFIGS[t['pa']]
    g = t['sign'] * G_TRUE
    if t['arm'] == 'pinned':
        h = H_PINNED
    elif t['arm'] == 'thin':
        h = H_THIN
    elif t['arm'] == 'sampled':
        # shared across the +/- pair: same seed parts without the sign
        rng = np.random.default_rng(_seed('h', t['snr'], t['cosi'], t['pa'], t['rep']))
        h = float(H_MEDIAN * 10 ** (H_SCATTER_DEX * rng.standard_normal()))
    else:
        raise ValueError(f"unknown arm {t['arm']!r}")
    return {
        'cosi': t['cosi'],
        'theta_int': theta,
        'g1': g * math.cos(2 * phi),
        'g2': g * math.sin(2 * phi),
        f'{BAND}.flux': FLUX,
        f'{BAND}.rscale': RSCALE,
        'h_over_r': h,
        f'{BAND}.x0': 0.0,
        f'{BAND}.y0': 0.0,
    }


def build_priors(arm: str):
    from kl_pipe.priors import CircularUniform, LogNormal, PriorDict, Uniform

    pd_ = {
        'g1': Uniform(-SHEAR_PRIOR_HALFWIDTH, SHEAR_PRIOR_HALFWIDTH),
        'g2': Uniform(-SHEAR_PRIOR_HALFWIDTH, SHEAR_PRIOR_HALFWIDTH),
        'cosi': Uniform(0.001, 0.999),
        'theta_int': CircularUniform(2.0 * math.pi),
        # forecast flat boxes (rollgrid_j prior_overrides)
        f'{BAND}.flux': Uniform(0.1 * FLUX, 10.0 * FLUX),
        f'{BAND}.rscale': Uniform(0.00596, 0.5958),
        f'{BAND}.x0': 0.0,
        f'{BAND}.y0': 0.0,
    }
    if arm == 'pinned':
        pd_['h_over_r'] = H_PINNED
    elif arm == 'thin':
        pd_['h_over_r'] = H_THIN
    else:
        pd_['h_over_r'] = LogNormal(math.log(H_MEDIAN), H_SCATTER_DEX * math.log(10.0))
    return PriorDict(pd_)


def run_one(t: dict, out_dir: Path, n_warmup: int, n_samples: int) -> dict:
    import jax.numpy as jnp

    from kl_pipe.ensemble.mocks import _build_band_psf, _channel_seeds, _make_band_obs
    from kl_pipe.ensemble.spec import ObservationConfig
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.parameters import ImagePars
    from kl_pipe.sampling import InferenceTask
    from kl_pipe.sampling.configs import NumpyroSamplerConfig
    from kl_pipe.sampling.initialization import InitConfig, Initializer
    from kl_pipe.sampling.numpyro import NumpyroSampler
    from kl_pipe.source import SourceModel

    t0 = time.time()
    fid = fit_id(t)
    config = ObservationConfig.from_yaml(OBS_CONFIG)
    truth = truth_for(t)
    source = SourceModel(broadband_models={BAND: InclinedExponentialModel()})
    priors = build_priors(t['arm'])

    image_pars = ImagePars(
        shape=(config.stamp_broadband_pix, config.stamp_broadband_pix),
        pixel_scale=config.pixel_scale_arcsec,
        indexing='ij',
    )
    band_spec = config.band_psf[BAND]
    # the +/- pair shares the noise seed
    noise_seed = _seed('noise', t['arm'], t['snr'], t['cosi'], t['pa'], t['rep'])
    obs, snr_eff = _make_band_obs(
        source,
        truth,
        _build_band_psf(band_spec, BAND, Z, mock=True),
        _build_band_psf(band_spec, BAND, Z),
        image_pars,
        BAND,
        t['snr'],
        int(_channel_seeds(noise_seed, 1)[0]),
        OVERSAMPLE,
    )
    task = InferenceTask.from_obs(source, priors, image_obs={BAND: obs})
    names = list(priors.sampled_names)

    sampler_seed = _seed('sampler', fid)
    init = Initializer(
        task,
        InitConfig(
            map_bounded=True,
            map_polish_steps=8,
            map_polish_basins=3,
            eig_floor_mode='prior',
            eig_floor=0.5,
        ),
        seed=sampler_seed,
        image_obs={BAND: obs},
    ).run()
    t_init = time.time()
    cfg = NumpyroSamplerConfig(
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_chains=4,
        target_accept_prob=0.9,
        precondition='laplace',
        precondition_unconstrained=True,
        precondition_adapt_mass=True,
        seed=sampler_seed,
    )
    result = NumpyroSampler(task, cfg, preconditioner=init.preconditioner).run()
    t_end = time.time()

    samples = np.asarray(result.samples)
    if list(result.param_names) != names:
        raise RuntimeError(f"param order mismatch: {result.param_names} vs {names}")
    d = result.diagnostics
    row = dict(t)
    row.update(
        fit_id=fid,
        n_warmup=n_warmup,
        n_samples=n_samples,
        snr_effective=float(snr_eff),
        n_divergences=int(d['n_divergences']),
        mean_tree_depth=float(d['mean_tree_depth']),
        wall_s=t_end - t0,
        init_s=t_init - t0,
    )
    for k, v in truth.items():
        row[f'truth.{k}'] = v
    for i, n in enumerate(names):
        x = samples[:, i]
        row[f'mean.{n}'] = float(np.mean(x))
        row[f'median.{n}'] = float(np.median(x))
        row[f'std.{n}'] = float(np.std(x))
        row[f'rhat.{n}'] = float(d['r_hat'][n])
        row[f'ess.{n}'] = float(d['ess'][n])
    # single pass: unconverged fits are kept and flagged, not retried
    row['converged'] = converged(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f'{fid}.npz',
        samples=samples,
        names=np.array(names),
        log_prob=np.asarray(result.log_prob),
    )
    with open(out_dir / f'{fid}.json', 'w') as f:
        json.dump(row, f)
    return row


RHAT_MAX, ESS_MIN = 1.05, 100
GATE_PARAMS = ('g1', 'g2', 'cosi')


def converged(row: dict) -> bool:
    return all(
        row[f'rhat.{n}'] <= RHAT_MAX and row[f'ess.{n}'] >= ESS_MIN for n in GATE_PARAMS
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'order', 'run'])
    ap.add_argument('--min-snr', type=float, default=0.0)
    ap.add_argument('--max-snr', type=float, default=math.inf)
    ap.add_argument(
        '--pas',
        default=None,
        help='comma list of PA configs to keep (order subcommand)',
    )
    ap.add_argument('--arm', required=True, choices=['pinned', 'sampled', 'thin'])
    ap.add_argument(
        '--tasks', default=None, help="slice 'a:b' or comma list of indices"
    )
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'out')
    ap.add_argument('--n-warmup', type=int, default=200)
    ap.add_argument('--n-samples', type=int, default=300)
    args = ap.parse_args()

    all_tasks = tasks(args.arm)
    if args.cmd == 'list':
        print(len(all_tasks))
        return
    if args.cmd == 'order':
        # highest SNR first so a capped job keeps the constraining cells
        pas = set(args.pas.split(',')) if args.pas else set(PA_CONFIGS)
        keep = [
            i
            for i, t in enumerate(all_tasks)
            if args.min_snr <= t['snr'] <= args.max_snr
            and t['pa'] in pas
            and t['rep'] <= args.max_rep
        ]
        keep.sort(key=lambda i: -all_tasks[i]['snr'])
        print(','.join(str(i) for i in keep))
        return
    if args.tasks is None:
        idx = range(len(all_tasks))
    elif ':' in args.tasks:
        a, b = args.tasks.split(':')
        idx = range(int(a), min(int(b), len(all_tasks)))
    else:
        idx = [int(i) for i in args.tasks.split(',')]
    out_dir = args.out / args.arm
    for i in idx:
        t = all_tasks[i]
        if (out_dir / f'{fit_id(t)}.json').exists():
            continue
        try:
            row = run_one(t, out_dir, args.n_warmup, args.n_samples)
            print(
                f"[{i}] {row['fit_id']} wall {row['wall_s']:.0f}s div {row['n_divergences']} "
                f"g1 {row['mean.g1']:+.4f}+-{row['std.g1']:.4f} g2 {row['mean.g2']:+.4f}+-{row['std.g2']:.4f} "
                f"cosi {row['mean.cosi']:.3f}+-{row['std.cosi']:.3f} rhat_max "
                f"{max(v for k, v in row.items() if k.startswith('rhat.')):.3f} converged {row['converged']}",
                flush=True,
            )
        except Exception as err:
            # failed fits are recorded loudly and kept out of the results
            print(
                f"[{i}] {fit_id(t)} FAILED: {type(err).__name__}: {err}",
                file=sys.stderr,
                flush=True,
            )
            with open(out_dir / f'{fit_id(t)}.FAILED', 'w') as f:
                f.write(f"{type(err).__name__}: {err}\n")


if __name__ == '__main__':
    main()
