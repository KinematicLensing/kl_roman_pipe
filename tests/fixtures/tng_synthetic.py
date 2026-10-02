"""
Synthetic stand-in for the TNG50 mock data files.

Writes files with the names and internal layout read by ``kl_pipe.tng.loaders``
and ``kl_pipe.tng.data_vectors`` so that code paths (and the TNG tutorial) can
run without the private TNG50 data product. Galaxies are simple rotating
exponential disks; no values derive from the real data.

Run as a script to write the fixture: ``python tests/fixtures/tng_synthetic.py OUT_DIR``.
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import h5py
import numpy as np
from scipy.integrate import trapezoid

BANDS = ('u', 'g', 'r', 'i', 'z')
DEFAULT_SUBHALO_IDS = (561676, 490815)
REQUIRED_FILES = (
    'gas_data_analysis.npz',
    'stellar_data_analysis.npz',
    'subhalo_data_analysis.npz',
    'SDSS_hr_stelib_stellar_photometrics.hdf5',
) + tuple(f'{b}_SDSS.res' for b in BANDS)

L_SUN_ERG_S = 3.826e33
# AB absolute magnitudes of the Sun, matching TNGDataVectorGenerator.M_abs_sun
M_SUN_AB = {'u': 6.39, 'g': 5.11, 'r': 4.65, 'i': 4.53, 'z': 4.50}
# approximate SDSS filter centers and FWHM [Angstrom]
_FILTER_CENTER = {'u': 3550.0, 'g': 4700.0, 'r': 6200.0, 'i': 7500.0, 'z': 8900.0}
_FILTER_FWHM = {'u': 600.0, 'g': 1300.0, 'r': 1200.0, 'i': 1300.0, 'z': 1200.0}
# fraction of stellar light per band, crude young-to-old colour trend below
_BAND_WEIGHT = {'u': 0.05, 'g': 0.15, 'r': 0.2, 'i': 0.2, 'z': 0.15}
_BAND_AGE_SLOPE = {'u': -1.2, 'g': -0.9, 'r': -0.7, 'i': -0.6, 'z': -0.5}

# (inclination [deg], PA [deg], log10 M* [Msun], scale length [kpc], Vmax [km/s], bar)
_GALAXY_SPECS = {
    561676: (55.0, 30.0, 10.6, 3.0, 220.0, True),
    490815: (40.0, 110.0, 10.0, 2.0, 150.0, False),
}
_FALLBACK_SPEC = (50.0, 60.0, 10.3, 2.5, 180.0, False)


def _disk_particles(
    rng: np.random.Generator,
    n: int,
    rd_kpc: float,
    hz_kpc: float,
    vmax: float,
    sigma_v: float,
    bar: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exponential disk in its own frame: positions [kpc], velocities [km/s], R [kpc]."""
    radius = rng.gamma(2.0, rd_kpc, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    x, y = radius * np.cos(phi), radius * np.sin(phi)
    if bar:
        inner = radius < 1.5 * rd_kpc
        x[inner] *= 1.3
        y[inner] /= 1.3
    z = rng.normal(0.0, hz_kpc, n)
    r_cyl = np.hypot(x, y)
    vc = vmax * (2.0 / np.pi) * np.arctan(r_cyl / (0.5 * rd_kpc))
    phi_cyl = np.arctan2(y, x)
    vel = np.column_stack(
        [-vc * np.sin(phi_cyl), vc * np.cos(phi_cyl), np.zeros(n)]
    ) + rng.normal(0.0, sigma_v, (n, 3))
    return np.column_stack([x, y, z]), vel, r_cyl


def _disk_to_sim(inc_deg: float, pa_deg: float) -> np.ndarray:
    """Rotation taking the disk frame to the simulation frame (observer along +z)."""
    i, p = np.radians(inc_deg), np.radians(pa_deg)
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(i), -np.sin(i)], [0.0, np.sin(i), np.cos(i)]]
    )
    rot_z = np.array(
        [[np.cos(p), -np.sin(p), 0.0], [np.sin(p), np.cos(p), 0.0], [0.0, 0.0, 1.0]]
    )
    return rot_z @ rot_x


def _make_galaxy(
    rng: np.random.Generator, subhalo_id: int, n_star: int, n_gas: int
) -> Tuple[Dict, Dict, Dict]:
    """Build the stellar, gas and subhalo dicts for one synthetic galaxy."""
    inc, pa, log_mstar, rd, vmax, bar = _GALAXY_SPECS.get(subhalo_id, _FALLBACK_SPEC)
    rot = _disk_to_sim(inc, pa)
    center = rng.uniform(1.0e4, 3.0e4, 3)
    v_bulk = rng.normal(0.0, 150.0, 3)
    mstar = 10.0**log_mstar
    mgas = 0.3 * mstar

    pos, vel, r_star = _disk_particles(rng, n_star, rd, 0.1 * rd, vmax, 25.0, bar)
    star_masses = np.full(n_star, mstar / n_star)
    ages_gyr = rng.uniform(0.3, 12.0, n_star)
    stellar = {
        'Coordinates': pos @ rot.T + center,
        'Velocities': vel @ rot.T + v_bulk,
        'Masses': star_masses,
        'GFM_InitialMass': 1.3 * star_masses,
        'GFM_Metallicity': np.clip(
            0.015 * 10.0 ** rng.normal(0.0, 0.3, n_star), 2e-4, 0.05
        ),
        'Stellar_age': ages_gyr,
    }
    attenuation = np.exp(-0.5 * np.exp(-r_star / rd))
    for b in BANDS:
        raw = (
            L_SUN_ERG_S
            * star_masses
            * _BAND_WEIGHT[b]
            * (ages_gyr / 1.0) ** _BAND_AGE_SLOPE[b]
        )
        dusted = raw * attenuation
        stellar[f'Raw_Luminosity_{b}'] = raw
        stellar[f'Dusted_Luminosity_{b}'] = dusted
        stellar[f'Absolute_Magnitude_{b}'] = M_SUN_AB[b] - 2.5 * np.log10(
            raw / L_SUN_ERG_S
        )
        stellar[f'Dusted_Absolute_Magnitude_{b}'] = M_SUN_AB[b] - 2.5 * np.log10(
            dusted / L_SUN_ERG_S
        )

    gpos, gvel, r_gas = _disk_particles(
        rng, n_gas, 1.5 * rd, 0.05 * rd, vmax, 10.0, False
    )
    sfr_weight = np.exp(-r_gas / rd) * (r_gas < 3.0 * rd)
    sfr_total = 3.0 * (mstar / 10.0**10.5)
    metals = np.zeros((n_gas, 10))
    metals[:, 0] = 0.75
    metals[:, 1] = 0.24
    metals[:, 2:] = 0.01 / 8.0
    gas = {
        'Coordinates': gpos @ rot.T + center,
        'Velocities': gvel @ rot.T + v_bulk,
        'Masses': np.full(n_gas, mgas / n_gas),
        'Density': 1e7 * np.exp(-r_gas / (1.5 * rd)),
        'ElectronAbundance': np.full(n_gas, 1.1),
        'GFM_Metallicity': np.full(n_gas, 0.015),
        'GFM_Metals': metals,
        'NeutralHydrogenAbundance': np.full(n_gas, 0.5),
        'StarFormationRate': sfr_total * sfr_weight / sfr_weight.sum(),
        'SubfindHsml': 0.3 + 0.2 * r_gas,
        'Temperature': np.full(n_gas, 1.0e4),
    }

    subhalo = {
        'SubhaloID': np.int64(subhalo_id),
        'Redshift': np.float64(0.0),
        'DistanceMpc': np.float64(50.0),
        'Inclination_star': np.float64(inc),
        'Position_Angle_star': np.float64(pa),
        'Inclination_gas': np.float64(inc),
        'Position_Angle_gas': np.float64(pa),
        'SubhaloPosX': np.float64(center[0]),
        'SubhaloPosY': np.float64(center[1]),
        'SubhaloPosZ': np.float64(center[2]),
        'SubhaloStellarMass': np.float64(mstar),
        'SubhaloHalfmassRadStars': np.float64(1.68 * rd),
        'R_e_fit_r': np.float64(1.68 * rd),
        'SubhaloMaxCircVel': np.float64(vmax),
        'SubhaloGasMetallicity': np.float64(0.015),
        'SubhaloSFR': np.float64(sfr_total),
        'Dusted_Luminosity_r': np.float64(stellar['Dusted_Luminosity_r'].sum()),
    }
    return stellar, gas, subhalo


def _write_photometric_table(path: Path) -> None:
    """Small blackbody SSP grid in the BC03 table layout (SEDs in Lsun/A/Msun)."""
    log_age = np.linspace(-3.0, np.log10(14.0), 24)
    metallicity = np.array([1e-4, 4e-3, 0.02, 0.05])
    wave = np.linspace(2800.0, 11500.0, 400)

    age = 10.0**log_age
    temp = np.clip(5500.0 * age ** (-0.15), 3500.0, 30000.0)
    temp = temp[None, :] * (metallicity[:, None] / 0.02) ** (-0.02)
    x = 1.4388e8 / (wave[None, None, :] * temp[:, :, None])
    shape = wave[None, None, :] ** -5 / np.expm1(x)
    shape /= trapezoid(shape, wave, axis=-1)[..., None]
    l_bol = 1.3 * age ** (-0.8)
    seds = l_bol[None, :, None] * shape

    with h5py.File(path, 'w') as f:
        f['LogAgeInGyr_bins'] = log_age.astype(np.float64)
        f['Metallicity_bins'] = metallicity.astype(np.float64)
        f['Wavelengths'] = wave.astype(np.float32)
        f['SEDs'] = seds.astype(np.float32)
        f['N_LogAgeInGyr'] = np.array([len(log_age)], dtype=np.int32)
        f['N_Metallicity'] = np.array([len(metallicity)], dtype=np.int32)
        for b in BANDS:
            in_band = np.abs(wave - _FILTER_CENTER[b]) < 0.5 * _FILTER_FWHM[b]
            l_band = trapezoid(seds[..., in_band], wave[in_band], axis=-1)
            f[f'Magnitude_{b}'] = (M_SUN_AB[b] - 2.5 * np.log10(l_band)).astype(
                np.float64
            )


def _write_filter(path: Path, band: str) -> None:
    """Gaussian throughput curve: two columns, wavelength [A] ascending, throughput."""
    sigma = _FILTER_FWHM[band] / 2.355
    wave = np.linspace(
        _FILTER_CENTER[band] - 2.5 * sigma, _FILTER_CENTER[band] + 2.5 * sigma, 41
    )
    throughput = 0.4 * np.exp(-0.5 * ((wave - _FILTER_CENTER[band]) / sigma) ** 2)
    np.savetxt(path, np.column_stack([wave, throughput]), fmt='%.1f %.5f')


def _save_object_array(path: Path, dicts: Sequence[Dict]) -> None:
    arr = np.empty(len(dicts), dtype=object)
    for i, d in enumerate(dicts):
        arr[i] = d
    np.savez(path, arr)


def write_synthetic_tng(
    out_dir: Path,
    subhalo_ids: Sequence[int] = DEFAULT_SUBHALO_IDS,
    seed: int = 0,
    n_star: int = 3000,
    n_gas: int = 2000,
) -> Path:
    """
    Write a synthetic TNG50 data directory.

    Parameters
    ----------
    out_dir : Path
        Directory to write into (created if missing).
    subhalo_ids : sequence of int, default=(561676, 490815)
        SubhaloIDs of the galaxies to write, in array order. The defaults are
        the IDs used by ``docs/tutorials/tng50_data.md``.
    seed : int, default=0
        Seed; output is deterministic in (seed, subhalo_ids, n_star, n_gas).
    n_star, n_gas : int
        Stellar and gas particles per galaxy.

    Returns
    -------
    Path
        ``out_dir``, containing every file in ``REQUIRED_FILES``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(set(subhalo_ids)) != len(subhalo_ids):
        raise ValueError(f'duplicate subhalo_ids: {subhalo_ids}')

    galaxies = [
        _make_galaxy(np.random.default_rng([seed, sid]), sid, n_star, n_gas)
        for sid in subhalo_ids
    ]
    _save_object_array(out_dir / 'stellar_data_analysis.npz', [g[0] for g in galaxies])
    _save_object_array(out_dir / 'gas_data_analysis.npz', [g[1] for g in galaxies])
    _save_object_array(out_dir / 'subhalo_data_analysis.npz', [g[2] for g in galaxies])
    _write_photometric_table(out_dir / 'SDSS_hr_stelib_stellar_photometrics.hdf5')
    for b in BANDS:
        _write_filter(out_dir / f'{b}_SDSS.res', b)
    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(write_synthetic_tng(args.out_dir, seed=args.seed))
