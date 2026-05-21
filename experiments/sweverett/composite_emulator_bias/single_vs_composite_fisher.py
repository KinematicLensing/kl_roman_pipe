"""Compare H_pp[theta_int] between single-component disk-only and full BulgeDisk
at the composite test geometry. Tests whether the composite's larger
theta_int uncertainty is geometric (e.g. disk_rscale, image grid) or
composite-specific (parameter degeneracies, bulge masking).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from tests.test_composite_intensity import (
    _generate_composite_synthetic,
    _IMAGE_PARS,
    _TRUE_PARS_SHARED,
    _TEST_PSF,
)
from kl_pipe.intensity import BulgeDiskModel, InclinedExponentialModel
from kl_pipe.observation import build_image_obs
from kl_pipe.likelihood import create_jitted_likelihood_intensity
from kl_pipe.synthetic import generate_sersic_intensity_2d
from kl_pipe.noise import add_intensity_noise


def composite_fisher_diag():
    """H_pp diagonal for full BulgeDisk model at the composite test geometry."""
    pars = dict(_TRUE_PARS_SHARED)
    model = BulgeDiskModel(shared_centroids=True)
    theta_true = model.pars2theta(pars)

    data_true, _, variance = _generate_composite_synthetic(
        pars, _IMAGE_PARS, snr=1000, seed=42, psf=_TEST_PSF
    )
    obs = build_image_obs(
        _IMAGE_PARS,
        psf=_TEST_PSF,
        oversample=5,
        int_model=model,
        data=data_true,
        variance=variance,
    )
    log_like = create_jitted_likelihood_intensity(model, obs)

    def nll(theta):
        return -log_like(theta)

    H = jax.hessian(nll)(theta_true)
    H_diag = np.diag(np.array(H))

    name_to_idx = {n: i for i, n in enumerate(model.PARAMETER_NAMES)}
    return H_diag, name_to_idx


def single_disk_only_fisher_diag():
    """H_pp diagonal for a pure-disk InclinedExponential matched to the composite
    disk component, on the SAME image_pars as composite."""
    # disk-only true params, total_flux = full composite flux (so noise level matches)
    pars_full = dict(_TRUE_PARS_SHARED)
    pars = {
        'cosi': pars_full['cosi'],
        'theta_int': pars_full['theta_int'],
        'g1': pars_full['g1'],
        'g2': pars_full['g2'],
        'flux': pars_full['total_flux'],  # entire flux into disk
        'int_rscale': pars_full['disk_rscale'],
        'int_h_over_r': pars_full['disk_h_over_r'],
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    model = InclinedExponentialModel()
    theta_true = model.pars2theta(pars)

    # data_true = GalSim render of the disk-only profile (matched physical h_z)
    data_true = generate_sersic_intensity_2d(
        image_pars=_IMAGE_PARS,
        flux=pars['flux'],
        int_rscale=pars['int_rscale'],
        n_sersic=1.0,
        cosi=pars['cosi'],
        theta_int=pars['theta_int'],
        g1=pars['g1'],
        g2=pars['g2'],
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=pars['int_h_over_r'],
        backend='galsim',
        psf=_TEST_PSF,
    )
    _, variance = add_intensity_noise(
        data_true, target_snr=1000, include_poisson=False, seed=42
    )

    obs = build_image_obs(
        _IMAGE_PARS,
        psf=_TEST_PSF,
        oversample=5,
        int_model=model,
        data=jnp.asarray(data_true),
        variance=variance,
    )
    log_like = create_jitted_likelihood_intensity(model, obs)

    def nll(theta):
        return -log_like(theta)

    H = jax.hessian(nll)(theta_true)
    H_diag = np.diag(np.array(H))

    name_to_idx = {n: i for i, n in enumerate(model.PARAMETER_NAMES)}
    return H_diag, name_to_idx


def composite_with_zero_bulge_fisher_diag():
    """Composite model BUT bulge_frac=0 → effectively disk-only via composite path.
    This isolates the marginalization-over-bulge-params effect."""
    pars = dict(_TRUE_PARS_SHARED)
    pars['bulge_frac'] = 0.001  # near-zero, can't go to exactly 0 (numeric)
    model = BulgeDiskModel(shared_centroids=True)
    theta_true = model.pars2theta(pars)

    data_true, _, variance = _generate_composite_synthetic(
        pars, _IMAGE_PARS, snr=1000, seed=42, psf=_TEST_PSF
    )
    obs = build_image_obs(
        _IMAGE_PARS,
        psf=_TEST_PSF,
        oversample=5,
        int_model=model,
        data=data_true,
        variance=variance,
    )
    log_like = create_jitted_likelihood_intensity(model, obs)

    def nll(theta):
        return -log_like(theta)

    H = jax.hessian(nll)(theta_true)
    H_diag = np.diag(np.array(H))

    name_to_idx = {n: i for i, n in enumerate(model.PARAMETER_NAMES)}
    return H_diag, name_to_idx


def main():
    print(
        'Computing H_pp[theta_int] for three configurations at composite geometry...\n'
    )

    print('1) Full BulgeDisk (true bulge_frac=0.25):')
    H1, idx1 = composite_fisher_diag()
    print(f'   H_pp[theta_int] = {H1[idx1["theta_int"]]:.3e}')
    print(
        f'   sigma_diag       = {1.0 / np.sqrt(H1[idx1["theta_int"]]):.4f} '
        f'({1.0/np.sqrt(H1[idx1["theta_int"]])/0.785*100:.2f}% of truth)\n'
    )

    print('2) Pure disk only (no bulge), single-component InclinedExponential:')
    H2, idx2 = single_disk_only_fisher_diag()
    print(f'   H_pp[theta_int] = {H2[idx2["theta_int"]]:.3e}')
    print(
        f'   sigma_diag       = {1.0 / np.sqrt(H2[idx2["theta_int"]]):.4f} '
        f'({1.0/np.sqrt(H2[idx2["theta_int"]])/0.785*100:.2f}% of truth)\n'
    )

    print('3) Composite with bulge_frac=0.001 (composite path, no bulge contribution):')
    H3, idx3 = composite_with_zero_bulge_fisher_diag()
    print(f'   H_pp[theta_int] = {H3[idx3["theta_int"]]:.3e}')
    print(
        f'   sigma_diag       = {1.0 / np.sqrt(H3[idx3["theta_int"]]):.4f} '
        f'({1.0/np.sqrt(H3[idx3["theta_int"]])/0.785*100:.2f}% of truth)\n'
    )

    print('Comparison:')
    h1 = H1[idx1['theta_int']]
    h2 = H2[idx2['theta_int']]
    h3 = H3[idx3['theta_int']]
    print(
        f'   single-disk vs composite:        H ratio = {h2/h1:.2f}x more info in single-disk'
    )
    print(
        f'   composite_no_bulge vs composite: H ratio = {h3/h1:.2f}x more info without bulge'
    )


if __name__ == '__main__':
    main()
